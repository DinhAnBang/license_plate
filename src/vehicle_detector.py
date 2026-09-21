"""ONNX Runtime vehicle detector for the Phase 1 YOLO26n model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import time

import cv2
import numpy as np
import onnxruntime as ort


VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@dataclass(frozen=True, slots=True)
class VehicleDetection:
    """One vehicle bounding box in coordinates of the original image."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class _LetterboxInfo:
    scale: float
    pad_left: int
    pad_top: int
    original_width: int
    original_height: int


class VehicleDetector:
    """Detect COCO car, motorcycle, bus, and truck classes with ONNX Runtime.

    The ONNX session is created once in ``__init__`` and reused by every call
    to :meth:`detect`.
    """

    def __init__(
        self,
        model_path: str | Path = "models/vehicle/yolo26n.onnx",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        debug: bool = False,
        providers: Sequence[str] | None = None,
    ) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Vehicle model not found: {self.model_path}")

        self.confidence_threshold = float(confidence_threshold)
        self.iou_threshold = float(iou_threshold)
        self.debug = debug

        selected_providers = list(providers) if providers else self._default_providers()
        self.session = ort.InferenceSession(
            str(self.model_path), providers=selected_providers
        )
        self.session_init_count = 1

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) != 1:
            raise ValueError(
                "yolo26n.onnx is expected to have exactly one input and one output; "
                f"got {len(inputs)} input(s) and {len(outputs)} output(s)"
            )

        self.input_info = inputs[0]
        self.output_info = outputs[0]
        self.input_name = self.input_info.name
        self.output_name = self.output_info.name
        self.input_height, self.input_width = self._get_input_size(
            self.input_info.shape
        )
        self._validate_model_metadata()
        self.detect_call_count = 0
        self.total_processing_seconds = 0.0
        self.total_inference_seconds = 0.0
        if self.debug:
            self._print_model_info()

    @staticmethod
    def _default_providers() -> list[str]:
        available = ort.get_available_providers()
        providers: list[str] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    @staticmethod
    def _get_input_size(shape: Sequence[int | str | None]) -> tuple[int, int]:
        if len(shape) != 4 or shape[1] != 3:
            raise ValueError(f"Expected NCHW model input [1, 3, H, W], got {shape}")
        height, width = shape[2], shape[3]
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f"Model has a dynamic spatial input shape: {shape}")
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid model input shape: {shape}")
        return height, width

    def _validate_model_metadata(self) -> None:
        if self.input_info.type != "tensor(float)":
            raise ValueError(
                f"Expected float32 input, got {self.input_info.type}"
            )

        # Inspected model output is [1, 84, 8400]: four xywh values followed
        # by 80 COCO class scores. It has no separate objectness channel.
        shape = self.output_info.shape
        if len(shape) != 3 or shape[1] != 84:
            raise ValueError(
                "Unsupported yolo26n output. Expected [batch, 84, predictions] "
                f"(xywh + 80 class scores), got {shape}"
            )
        if self.output_info.type != "tensor(float)":
            raise ValueError(
                f"Expected float32 output, got {self.output_info.type}"
            )

    def _print_model_info(self) -> None:
        print("[VehicleDetector]")
        print(f"model: {self.model_path}")
        print(f"provider: {self.session.get_providers()[0]}")
        print(f"input_name: {self.input_name}")
        print(f"input_shape: {self.input_info.shape}")
        print(f"output_name: {self.output_name}")
        print(f"output_shape: {self.output_info.shape}")

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, _LetterboxInfo]:
        if not isinstance(image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected a BGR image with shape HxWx3, got {image.shape}")

        original_height, original_width = image.shape[:2]
        if original_height <= 0 or original_width <= 0:
            raise ValueError("Image dimensions must be greater than zero")

        scale = min(
            self.input_width / original_width,
            self.input_height / original_height,
        )
        resized_width = min(self.input_width, round(original_width * scale))
        resized_height = min(self.input_height, round(original_height * scale))

        if (resized_width, resized_height) != (original_width, original_height):
            resized = cv2.resize(
                image,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            resized = image

        horizontal_padding = self.input_width - resized_width
        vertical_padding = self.input_height - resized_height
        pad_left = horizontal_padding // 2
        pad_right = horizontal_padding - pad_left
        pad_top = vertical_padding // 2
        pad_bottom = vertical_padding - pad_top
        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.transpose(2, 0, 1)[None]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32) / 255.0
        info = _LetterboxInfo(
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            original_width=original_width,
            original_height=original_height,
        )
        return tensor, info

    def detect(self, image: np.ndarray) -> list[VehicleDetection]:
        """Return every valid vehicle found in a BGR image/frame."""

        processing_started = time.perf_counter()
        tensor, letterbox = self._preprocess(image)
        inference_started = time.perf_counter()
        raw_output = self.session.run(
            [self.output_name], {self.input_name: tensor}
        )[0]
        self.total_inference_seconds += time.perf_counter() - inference_started
        detections = self._decode(raw_output, letterbox)
        self.detect_call_count += 1
        self.total_processing_seconds += time.perf_counter() - processing_started
        return detections

    def _decode(
        self, raw_output: np.ndarray, letterbox: _LetterboxInfo
    ) -> list[VehicleDetection]:
        if raw_output.ndim != 3 or raw_output.shape[0] != 1:
            raise ValueError(
                f"Expected runtime output [1, 84, N], got {raw_output.shape}"
            )
        if raw_output.shape[1] != 84:
            raise ValueError(
                "Expected 84 output channels (xywh + 80 class scores), "
                f"got {raw_output.shape[1]}"
            )

        predictions = raw_output[0].T
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
        confidences = class_scores[
            np.arange(class_scores.shape[0]), class_ids
        ].astype(np.float32)

        confidence_mask = confidences >= self.confidence_threshold
        after_confidence = int(np.count_nonzero(confidence_mask))
        vehicle_mask = confidence_mask & np.isin(
            class_ids, np.fromiter(VEHICLE_CLASSES, dtype=np.int64)
        )
        after_vehicle_class = int(np.count_nonzero(vehicle_mask))

        boxes_xywh = predictions[vehicle_mask, :4].astype(np.float32, copy=True)
        filtered_confidences = confidences[vehicle_mask]
        filtered_class_ids = class_ids[vehicle_mask]

        if boxes_xywh.shape[0] == 0:
            self._print_debug(len(predictions), after_confidence, 0, 0)
            return []

        boxes_xyxy = self._xywh_to_original_xyxy(boxes_xywh, letterbox)
        valid = (
            np.isfinite(boxes_xyxy).all(axis=1)
            & (boxes_xyxy[:, 2] > boxes_xyxy[:, 0])
            & (boxes_xyxy[:, 3] > boxes_xyxy[:, 1])
        )
        boxes_xyxy = boxes_xyxy[valid]
        filtered_confidences = filtered_confidences[valid]
        filtered_class_ids = filtered_class_ids[valid]

        kept_indices: list[int] = []
        for class_id in VEHICLE_CLASSES:
            class_indices = np.flatnonzero(filtered_class_ids == class_id)
            if class_indices.size == 0:
                continue
            relative_kept = self._nms(
                boxes_xyxy[class_indices],
                filtered_confidences[class_indices],
                self.iou_threshold,
            )
            kept_indices.extend(class_indices[relative_kept].tolist())

        kept_indices.sort(key=lambda index: float(filtered_confidences[index]), reverse=True)
        detections: list[VehicleDetection] = []
        for index in kept_indices:
            class_id = int(filtered_class_ids[index])
            box = self._round_and_clamp_box(boxes_xyxy[index], letterbox)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            detections.append(
                VehicleDetection(
                    class_id=class_id,
                    class_name=VEHICLE_CLASSES[class_id],
                    confidence=float(filtered_confidences[index]),
                    bbox=box,
                )
            )

        self._print_debug(
            len(predictions),
            after_confidence,
            after_vehicle_class,
            len(detections),
        )
        return detections

    @staticmethod
    def _xywh_to_original_xyxy(
        boxes_xywh: np.ndarray, info: _LetterboxInfo
    ) -> np.ndarray:
        boxes = np.empty_like(boxes_xywh, dtype=np.float32)
        boxes[:, 0] = boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2.0
        boxes[:, 1] = boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2.0
        boxes[:, 2] = boxes_xywh[:, 0] + boxes_xywh[:, 2] / 2.0
        boxes[:, 3] = boxes_xywh[:, 1] + boxes_xywh[:, 3] / 2.0

        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - info.pad_left) / info.scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - info.pad_top) / info.scale
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, info.original_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, info.original_height)
        return boxes

    @staticmethod
    def _round_and_clamp_box(
        box: np.ndarray, info: _LetterboxInfo
    ) -> tuple[int, int, int, int]:
        rounded = np.rint(box).astype(np.int64)
        x1 = int(np.clip(rounded[0], 0, info.original_width - 1))
        y1 = int(np.clip(rounded[1], 0, info.original_height - 1))
        x2 = int(np.clip(rounded[2], 0, info.original_width))
        y2 = int(np.clip(rounded[3], 0, info.original_height))
        return x1, y1, x2, y2

    @staticmethod
    def _nms(
        boxes: np.ndarray, scores: np.ndarray, iou_threshold: float
    ) -> np.ndarray:
        if boxes.shape[0] == 0:
            return np.empty(0, dtype=np.int64)

        x1, y1, x2, y2 = boxes.T
        areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
        order = np.argsort(-scores, kind="stable")
        keep: list[int] = []

        while order.size > 0:
            current = int(order[0])
            keep.append(current)
            if order.size == 1:
                break

            remaining = order[1:]
            intersection_x1 = np.maximum(x1[current], x1[remaining])
            intersection_y1 = np.maximum(y1[current], y1[remaining])
            intersection_x2 = np.minimum(x2[current], x2[remaining])
            intersection_y2 = np.minimum(y2[current], y2[remaining])
            intersection_width = np.maximum(0.0, intersection_x2 - intersection_x1)
            intersection_height = np.maximum(0.0, intersection_y2 - intersection_y1)
            intersection = intersection_width * intersection_height
            union = areas[current] + areas[remaining] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]

        return np.asarray(keep, dtype=np.int64)

    def _print_debug(
        self,
        raw_count: int,
        confidence_count: int,
        vehicle_count: int,
        nms_count: int,
    ) -> None:
        if not self.debug:
            return
        print(f"Raw predictions: {raw_count}")
        print(f"After confidence filter: {confidence_count}")
        print(f"After vehicle class filter: {vehicle_count}")
        print(f"After NMS: {nms_count}")
