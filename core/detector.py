"""ONNX Runtime license plate detector."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
import onnxruntime as ort

from .config import DETECTOR_CONF_THRESHOLD, DETECTOR_NMS_IOU_THRESHOLD


class Detection(TypedDict):
    """Public representation of one detected license plate."""

    conf: float
    box: list[int]


class Timing(TypedDict):
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float
    total_ms: float


class DetectionStats(TypedDict):
    raw_detector_candidates: int
    detections_after_threshold: int
    detections_after_nms: int


class DetectorError(RuntimeError):
    """Raised when the model or an inference result is not supported."""


@dataclass(frozen=True)
class _LetterboxInfo:
    scale: float
    pad_left: int
    pad_top: int


class PlateDetector:
    """Detect all license plates from an OpenCV BGR image using ONNX Runtime.

    Supported models emit raw YOLO-style predictions shaped
    ``[1, 4 + classes, candidates]`` (or its transpose): center-x, center-y,
    width, height, then one confidence score per class. The session is created
    once in ``__init__`` and reused by every ``detect``.
    """

    def __init__(
        self,
        model_path: str | Path,
        conf_threshold: float = DETECTOR_CONF_THRESHOLD,
        iou_threshold: float = DETECTOR_NMS_IOU_THRESHOLD,
        providers: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0.0 and 1.0")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0.0 and 1.0")

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise DetectorError(f"Model file does not exist: {self.model_path}")

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        requested_providers = list(providers or ["CPUExecutionProvider"])

        try:
            # This is intentionally done only in the constructor. Inference
            # reuses this session instead of loading the model per image.
            self.session = ort.InferenceSession(
                str(self.model_path),
                providers=requested_providers,
            )
        except Exception as exc:
            raise DetectorError(
                f"ONNX Runtime could not load model '{self.model_path}': {exc}"
            ) from exc

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise DetectorError(
                f"Expected exactly one model input, found {len(inputs)}."
            )
        if not outputs:
            raise DetectorError("The model has no outputs.")

        self._input = inputs[0]
        self._outputs = outputs
        self._input_shape = list(self._input.shape)
        self._input_type = self._input.type
        self._input_name = self._input.name
        self._output_names = [output.name for output in outputs]

        self._input_height, self._input_width, self._channel_first = (
            self._validate_input_metadata()
        )

    @property
    def providers(self) -> list[str]:
        """Return providers actually used by the ONNX Runtime session."""

        return list(self.session.get_providers())

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def input_shape(self) -> list[Any]:
        return list(self._input_shape)

    @property
    def input_type(self) -> str:
        return self._input_type

    @property
    def output_info(self) -> list[dict[str, Any]]:
        return [
            {
                "name": output.name,
                "shape": list(output.shape),
                "type": output.type,
            }
            for output in self._outputs
        ]

    def detect(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Return every valid plate detection in an OpenCV BGR image."""

        detections, _, _ = self._detect(image, conf_threshold=conf_threshold)
        return detections

    def detect_with_timing(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> tuple[list[Detection], Timing]:
        """Run detection and return detections plus stage timings in milliseconds."""

        detections, timings, _ = self._detect(image, conf_threshold=conf_threshold)
        return detections, timings

    def detect_with_stats(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None = None,
    ) -> tuple[list[Detection], DetectionStats]:
        """Run detection and expose development counters without changing output."""

        detections, _, stats = self._detect(image, conf_threshold=conf_threshold)
        return detections, stats

    def _detect(
        self,
        image: np.ndarray,
        *,
        conf_threshold: float | None,
    ) -> tuple[list[Detection], Timing, DetectionStats]:
        threshold = self.conf_threshold if conf_threshold is None else float(conf_threshold)
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0.0 and 1.0")

        self._validate_image(image)
        total_start = time.perf_counter()

        preprocess_start = time.perf_counter()
        tensor, letterbox = self._preprocess(image)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        try:
            outputs = self.session.run(
                self._output_names,
                {self._input_name: tensor},
            )
        except Exception as exc:
            raise DetectorError(f"ONNX Runtime inference failed: {exc}") from exc
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        postprocess_start = time.perf_counter()
        detections, stats = self._postprocess_with_stats(
            outputs,
            image_shape=image.shape,
            letterbox=letterbox,
            conf_threshold=threshold,
        )
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        total_ms = (time.perf_counter() - total_start) * 1000.0

        timings: Timing = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": total_ms,
        }
        return detections, timings, stats

    def _validate_input_metadata(self) -> tuple[int, int, bool]:
        if self._input_type != "tensor(float)":
            raise DetectorError(
                "Unsupported model input type "
                f"{self._input_type!r}; this detector expects tensor(float)."
            )
        if len(self._input_shape) != 4:
            raise DetectorError(
                "Unsupported model input shape "
                f"{self._input_shape!r}; expected a 4D image tensor."
            )

        # The inspected model is NCHW [1, 3, 640, 640]. Accepting NHWC here
        # keeps the failure mode explicit if a compatible model is substituted.
        if self._input_shape[1] == 3:
            channel_first = True
            height_dimension = self._input_shape[2]
            width_dimension = self._input_shape[3]
        elif self._input_shape[3] == 3:
            channel_first = False
            height_dimension = self._input_shape[1]
            width_dimension = self._input_shape[2]
        else:
            raise DetectorError(
                "Unsupported model input layout/shape "
                f"{self._input_shape!r}; expected NCHW or NHWC with 3 channels."
            )

        if not isinstance(height_dimension, int) or not isinstance(width_dimension, int):
            raise DetectorError(
                "Dynamic spatial input shape is not supported without a configured "
                f"input size: {self._input_shape!r}"
            )
        if height_dimension <= 0 or width_dimension <= 0:
            raise DetectorError(f"Invalid model input shape: {self._input_shape!r}")

        batch_dimension = self._input_shape[0]
        if isinstance(batch_dimension, int) and batch_dimension != 1:
            raise DetectorError(
                f"Unsupported batch size {batch_dimension}; expected batch size 1."
            )

        return height_dimension, width_dimension, channel_first

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise DetectorError("image must be a NumPy array returned by OpenCV.")
        if image.ndim != 3 or image.shape[2] != 3:
            raise DetectorError(
                f"Expected a color image with shape [H, W, 3], got {image.shape}."
            )
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise DetectorError(f"Image has invalid dimensions: {image.shape}.")

    def _preprocess(
        self,
        image: np.ndarray,
    ) -> tuple[np.ndarray, _LetterboxInfo]:
        original_height, original_width = image.shape[:2]

        # Letterbox preserves the plate geometry while fitting the image into
        # the fixed model canvas. The same scale and padding undo this later.
        scale = min(
            self._input_width / original_width,
            self._input_height / original_height,
        )
        resized_width = max(1, int(round(original_width * scale)))
        resized_height = max(1, int(round(original_height * scale)))
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

        pad_width = self._input_width - resized_width
        pad_height = self._input_height - resized_height
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # OpenCV supplies BGR; exported vision models conventionally consume
        # RGB. The inspected model has a 3-channel float input.
        model_image = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        model_image = model_image.astype(np.float32) / 255.0
        if self._channel_first:
            model_image = np.transpose(model_image, (2, 0, 1))
        tensor = np.expand_dims(model_image, axis=0).astype(np.float32, copy=False)

        return tensor, _LetterboxInfo(
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
        )

    def _postprocess(
        self,
        outputs: Sequence[np.ndarray],
        image_shape: tuple[int, ...],
        letterbox: _LetterboxInfo,
        conf_threshold: float | None = None,
    ) -> list[Detection]:
        """Compatibility wrapper returning only detections."""

        detections, _ = self._postprocess_with_stats(
            outputs,
            image_shape=image_shape,
            letterbox=letterbox,
            conf_threshold=self.conf_threshold if conf_threshold is None else conf_threshold,
        )
        return detections

    def _postprocess_with_stats(
        self,
        outputs: Sequence[np.ndarray],
        image_shape: tuple[int, ...],
        letterbox: _LetterboxInfo,
        conf_threshold: float,
    ) -> tuple[list[Detection], DetectionStats]:
        if not outputs:
            raise DetectorError("Inference returned no output tensors.")

        # Raw Ultralytics detection output is [batch, 4 + classes, candidates]
        # or its transpose. The smaller axis is the feature axis for normal
        # detector exports (for example 5 features for one class and 6 for two).
        raw_output = np.asarray(outputs[0])
        if raw_output.ndim == 3:
            if raw_output.shape[0] != 1:
                raise DetectorError(
                    "Unsupported output batch shape "
                    f"{list(raw_output.shape)}; expected batch size 1."
                )
            raw_output = raw_output[0]
        if raw_output.ndim != 2:
            raise DetectorError(
                "Unsupported detection output format. "
                f"name={self._outputs[0].name}, shape={list(raw_output.shape)}, "
                f"dtype={raw_output.dtype}, sample values="
                f"{raw_output.reshape(-1)[:10].tolist()}. "
                "Expected [1, 4 + classes, N] or [1, N, 4 + classes]."
            )

        if 5 <= raw_output.shape[0] <= 256 and raw_output.shape[0] < raw_output.shape[1]:
            predictions = raw_output.T
        elif 5 <= raw_output.shape[1] <= 256 and raw_output.shape[1] < raw_output.shape[0]:
            predictions = raw_output
        else:
            sample = raw_output.reshape(-1)[:10].tolist()
            raise DetectorError(
                "Unsupported detection output format. "
                f"name={self._outputs[0].name}, shape={list(raw_output.shape)}, "
                f"dtype={raw_output.dtype}, "
                f"sample values={sample}. Expected 4 box features followed by one or more class scores."
            )

        predictions = np.asarray(predictions, dtype=np.float32)
        raw_candidate_count = len(predictions)
        if predictions.size == 0:
            return [], self._detection_stats(0, 0, 0)

        # Ultralytics exports one score per class after xywh. The public engine
        # treats every supported class as a license plate, so retain the best
        # class confidence while keeping the established output schema.
        class_scores = predictions[:, 4:]
        scores = np.max(class_scores, axis=1)
        valid = np.isfinite(predictions).all(axis=1)
        valid &= scores >= conf_threshold
        threshold_count = int(np.count_nonzero(valid))
        if not np.any(valid):
            return [], self._detection_stats(raw_candidate_count, 0, 0)

        selected = predictions[valid]
        selected_scores = scores[valid]
        center_x = selected[:, 0]
        center_y = selected[:, 1]
        box_width = selected[:, 2]
        box_height = selected[:, 3]
        boxes = np.column_stack(
            (
                center_x - box_width / 2.0,
                center_y - box_height / 2.0,
                center_x + box_width / 2.0,
                center_y + box_height / 2.0,
            )
        )

        # Undo letterbox: model coordinates -> resized image -> original image.
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - letterbox.pad_left) / letterbox.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - letterbox.pad_top) / letterbox.scale

        image_height, image_width = image_shape[:2]
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, image_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, image_height)
        valid_boxes = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        if not np.any(valid_boxes):
            return [], self._detection_stats(raw_candidate_count, threshold_count, 0)
        boxes = boxes[valid_boxes]
        selected_scores = selected_scores[valid_boxes]

        # IoU-based NMS removes duplicate predictions for one plate while
        # retaining nearby plates whose overlap is below the configured limit.
        kept_indices = self._nms(boxes, selected_scores, self.iou_threshold)
        detections: list[Detection] = []
        for index in kept_indices:
            x1, y1, x2, y2 = boxes[index]
            left = max(0, min(image_width - 1, int(round(float(x1)))))
            top = max(0, min(image_height - 1, int(round(float(y1)))))
            right = max(left + 1, min(image_width, int(round(float(x2)))))
            bottom = max(top + 1, min(image_height, int(round(float(y2)))))
            if right <= left or bottom <= top:
                continue
            detections.append(
                {
                    "conf": float(selected_scores[index]),
                    "box": [left, top, right, bottom],
                }
            )

        return detections, self._detection_stats(
            raw_candidate_count,
            threshold_count,
            len(detections),
        )

    @staticmethod
    def _detection_stats(raw: int, after_threshold: int, after_nms: int) -> DetectionStats:
        return {
            "raw_detector_candidates": int(raw),
            "detections_after_threshold": int(after_threshold),
            "detections_after_nms": int(after_nms),
        }

    @staticmethod
    def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
        order = np.argsort(-scores, kind="stable")
        kept: list[int] = []

        while order.size > 0:
            current = int(order[0])
            kept.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            current_box = boxes[current]
            xx1 = np.maximum(current_box[0], boxes[remaining, 0])
            yy1 = np.maximum(current_box[1], boxes[remaining, 1])
            xx2 = np.minimum(current_box[2], boxes[remaining, 2])
            yy2 = np.minimum(current_box[3], boxes[remaining, 3])

            intersection_width = np.maximum(0.0, xx2 - xx1)
            intersection_height = np.maximum(0.0, yy2 - yy1)
            intersection = intersection_width * intersection_height
            current_area = max(0.0, current_box[2] - current_box[0]) * max(
                0.0,
                current_box[3] - current_box[1],
            )
            remaining_area = np.maximum(
                0.0,
                boxes[remaining, 2] - boxes[remaining, 0],
            ) * np.maximum(
                0.0,
                boxes[remaining, 3] - boxes[remaining, 1],
            )
            union = current_area + remaining_area - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]

        return kept
