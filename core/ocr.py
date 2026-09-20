"""MicroCharNet OCR core using ONNX Runtime only.

The exported MicroCharNet model is a character detector, not a CTC
recognizer. Its current ONNX output is the decoded YOLO detection tensor:
``[cx, cy, width, height, class_scores...]`` for each grid location.
"""

from __future__ import annotations

import ast
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import OCR_CONF_THRESHOLD, OCR_NMS_IOU_THRESHOLD


# Verified against the model's ONNX ``names`` metadata and the MicroCharNet
# dataset YAML. Runtime still reads the mapping from ONNX metadata below so a
# missing or incompatible mapping cannot silently produce fake text.
VERIFIED_CHAR_NAMES = tuple("0123456789abcdefghijklmnopqrstuvwxyz")


def _parse_names_metadata(raw_names: str | None) -> tuple[str, ...]:
    """Parse Ultralytics' serialized ``names`` metadata into an ordered tuple."""

    if not raw_names:
        raise ValueError(
            "OCR BLOCKED: CHARACTER CLASS MAPPING NOT VERIFIED; "
            "ONNX metadata does not contain 'names'."
        )

    try:
        parsed = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(
            "OCR BLOCKED: CHARACTER CLASS MAPPING NOT VERIFIED; "
            "ONNX 'names' metadata is not parseable."
        ) from exc

    if isinstance(parsed, dict):
        try:
            ordered = tuple(str(parsed[index]) for index in range(len(parsed)))
        except KeyError as exc:
            raise ValueError(
                "OCR BLOCKED: CHARACTER CLASS MAPPING NOT VERIFIED; "
                "ONNX class IDs are not contiguous."
            ) from exc
    elif isinstance(parsed, (list, tuple)):
        ordered = tuple(str(value) for value in parsed)
    else:
        raise ValueError(
            "OCR BLOCKED: CHARACTER CLASS MAPPING NOT VERIFIED; "
            "ONNX 'names' metadata has an unsupported format."
        )

    if ordered != VERIFIED_CHAR_NAMES:
        raise ValueError(
            "OCR BLOCKED: CHARACTER CLASS MAPPING NOT VERIFIED; "
            f"found {ordered!r}, expected the verified MicroCharNet mapping."
        )

    return ordered


def _dimension_as_int(dimension: Any) -> int | None:
    """Return a static ONNX dimension, or None for a dynamic dimension."""

    return dimension if isinstance(dimension, int) and dimension > 0 else None


def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Calculate IoU between one xyxy box and an array of xyxy boxes."""

    left = np.maximum(box[0], boxes[:, 0])
    top = np.maximum(box[1], boxes[:, 1])
    right = np.minimum(box[2], boxes[:, 2])
    bottom = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    area_box = max(0.0, float(box[2] - box[0])) * max(0.0, float(box[3] - box[1]))
    area_boxes = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = area_box + area_boxes - intersection
    return intersection / np.maximum(union, np.finfo(np.float32).eps)


def _class_aware_nms(
    detections: list[dict[str, Any]], iou_threshold: float
) -> list[dict[str, Any]]:
    """Apply greedy NMS independently for each character class."""

    if not detections:
        return []

    kept: list[dict[str, Any]] = []
    class_ids = sorted({int(item["class_id"]) for item in detections})
    for class_id in class_ids:
        candidates = [item for item in detections if int(item["class_id"]) == class_id]
        candidates.sort(key=lambda item: float(item["conf"]), reverse=True)

        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            if not candidates:
                break

            best_box = np.asarray(best["box"], dtype=np.float32)
            other_boxes = np.asarray([item["box"] for item in candidates], dtype=np.float32)
            overlaps = _iou(best_box, other_boxes)
            candidates = [item for item, overlap in zip(candidates, overlaps) if overlap <= iou_threshold]

    kept.sort(key=lambda item: float(item["conf"]), reverse=True)
    return kept


def group_and_sort_characters(detections: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group character boxes into one or two lines and sort each line by X.

    The largest vertical gap is treated as a line break only when it is large
    relative to the median character height. This avoids a pixel threshold
    tied to a particular crop resolution.
    """

    if not detections:
        return []

    indexed = list(enumerate(detections))
    indexed.sort(key=lambda pair: (float(pair[1]["box"][1] + pair[1]["box"][3]) / 2.0, pair[0]))
    centers_y = np.asarray(
        [(float(item["box"][1]) + float(item["box"][3])) / 2.0 for _, item in indexed],
        dtype=np.float32,
    )
    heights = np.asarray(
        [max(1.0, float(item["box"][3]) - float(item["box"][1])) for _, item in indexed],
        dtype=np.float32,
    )

    if len(indexed) == 1:
        groups = [indexed]
    else:
        gaps = np.diff(centers_y)
        median_height = max(1.0, float(np.median(heights)))
        line_break_threshold = 0.75 * median_height
        split_index = int(np.argmax(gaps)) + 1

        if float(gaps[split_index - 1]) > line_break_threshold:
            groups = [indexed[:split_index], indexed[split_index:]]
        else:
            groups = [indexed]

    sorted_groups: list[list[dict[str, Any]]] = []
    for group in groups:
        line = [item for _, item in group]
        line.sort(key=lambda item: (float(item["box"][0] + item["box"][2]) / 2.0, item["box"][0]))
        sorted_groups.append(line)

    sorted_groups.sort(
        key=lambda line: float(np.mean([(item["box"][1] + item["box"][3]) / 2.0 for item in line]))
    )
    return sorted_groups


def draw_ocr_result(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    """Draw character detections and final text on a copy of ``image``."""

    if image is None or not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")

    canvas = image.copy()
    for detection in result.get("characters", []):
        x1, y1, x2, y2 = [int(value) for value in detection["box"]]
        label = f'{detection["char"]} {float(detection["conf"]):.2f}'
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 0, 0), 1)
        text_y = max(12, y1 - 3)
        cv2.putText(
            canvas,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    final_text = str(result.get("text", "")) or "(no character)"
    text_label = f"OCR: {final_text}"
    confidence_label = f"conf={float(result.get('confidence', 0.0)):.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1
    (base_width, base_height), base_baseline = cv2.getTextSize(text_label, font, 1.0, text_thickness)
    max_text_width = max(1, canvas.shape[1] - 8)
    font_scale = min(0.55, max_text_width / max(1, base_width))
    font_scale = max(0.20, font_scale)
    (text_width, text_height), baseline = cv2.getTextSize(
        text_label, font, font_scale, text_thickness
    )
    (_, confidence_height), confidence_baseline = cv2.getTextSize(
        confidence_label, font, 0.35, 1
    )
    banner_height = text_height + baseline + confidence_height + confidence_baseline + 8
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1] - 1, banner_height), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        text_label,
        (4, text_height + 2),
        font,
        font_scale,
        (0, 255, 0),
        text_thickness,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        confidence_label,
        (4, text_height + confidence_height + baseline + 3),
        font,
        0.35,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


class MicroCharNetOCR:
    """Run MicroCharNet character detection through ONNX Runtime."""

    def __init__(
        self,
        model_path: str | Path = "models/OCR/microcharnet.onnx",
        conf_threshold: float = OCR_CONF_THRESHOLD,
        iou_threshold: float = OCR_NMS_IOU_THRESHOLD,
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be between 0 and 1")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be between 0 and 1")

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OCR model not found: {self.model_path}")

        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=["CPUExecutionProvider"],
        )
        self.session_creation_count = 1

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise ValueError(f"Expected one ONNX input, found {len(inputs)}")
        if not outputs:
            raise ValueError("ONNX model has no outputs")

        self.input_meta = inputs[0]
        self.output_meta = outputs[0]
        self.input_name = self.input_meta.name
        self.output_name = self.output_meta.name
        self.input_shape = list(self.input_meta.shape)
        self.output_shape = list(self.output_meta.shape)
        self.input_dtype = self.input_meta.type
        self.output_dtype = self.output_meta.type

        if len(self.input_shape) != 4:
            raise ValueError(f"Expected NCHW input, found shape {self.input_shape}")
        channels = _dimension_as_int(self.input_shape[1])
        if channels != 3:
            raise ValueError(f"Expected 3 input channels, found {self.input_shape[1]}")

        metadata = self.session.get_modelmeta().custom_metadata_map
        self.class_names = _parse_names_metadata(metadata.get("names"))
        self.num_classes = len(self.class_names)
        self.input_height, self.input_width = self._resolve_input_size(metadata)
        self._validate_output_shape()

    def _resolve_input_size(self, metadata: dict[str, str]) -> tuple[int, int]:
        height = _dimension_as_int(self.input_shape[2])
        width = _dimension_as_int(self.input_shape[3])
        if height is not None and width is not None:
            return height, width

        # Dynamic input models must publish their intended export size in the
        # metadata; otherwise there is no safe size to choose automatically.
        raw_imgsz = metadata.get("imgsz")
        if raw_imgsz:
            try:
                imgsz = ast.literal_eval(raw_imgsz)
            except (SyntaxError, ValueError):
                imgsz = None
            if isinstance(imgsz, (list, tuple)) and len(imgsz) == 2:
                dynamic_height, dynamic_width = (int(imgsz[0]), int(imgsz[1]))
                if dynamic_height > 0 and dynamic_width > 0:
                    return dynamic_height, dynamic_width

        raise ValueError(
            "OCR BLOCKED: dynamic ONNX input dimensions have no verified input size."
        )

    def _validate_output_shape(self) -> None:
        expected_channels = 4 + self.num_classes
        if len(self.output_shape) != 3:
            raise ValueError(f"Expected a 3D detector output, found {self.output_shape}")

        channel_candidates = {
            _dimension_as_int(self.output_shape[1]),
            _dimension_as_int(self.output_shape[2]),
        }
        if expected_channels not in channel_candidates:
            raise ValueError(
                "OCR BLOCKED: unsupported MicroCharNet output shape; "
                f"expected one axis to equal {expected_channels}, found {self.output_shape}."
            )

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
        if not isinstance(image, np.ndarray):
            raise TypeError("recognize(image) expects a numpy.ndarray")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"recognize(image) expects a BGR HWC image, found {image.shape}")
        if image.shape[0] == 0 or image.shape[1] == 0:
            raise ValueError("recognize(image) received an empty image")

        source = image
        if source.dtype != np.uint8:
            source = np.clip(source, 0, 255).astype(np.uint8)

        source_height, source_width = source.shape[:2]
        scale = min(self.input_width / source_width, self.input_height / source_height)
        resized_width = max(1, round(source_width * scale))
        resized_height = max(1, round(source_height * scale))
        resized = cv2.resize(source, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        pad_width = self.input_width - resized_width
        pad_height = self.input_height - resized_height
        left = round(pad_width / 2.0 - 0.1)
        right = round(pad_width / 2.0 + 0.1)
        top = round(pad_height / 2.0 - 0.1)
        bottom = round(pad_height / 2.0 + 0.1)
        letterboxed = cv2.copyMakeBorder(
            resized,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        # Ultralytics receives BGR OpenCV images, flips to RGB, then divides
        # by 255.0 before inference. Replicate that exact preprocessing here.
        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = rgb.astype(np.float32) / 255.0
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        transform = {
            "scale": float(scale),
            "pad_x": float(left),
            "pad_y": float(top),
            "source_width": float(source_width),
            "source_height": float(source_height),
        }
        return tensor, transform

    def _decode(self, output: np.ndarray, transform: dict[str, float]) -> list[dict[str, Any]]:
        if output.ndim != 3 or output.shape[0] != 1:
            raise ValueError(f"Unexpected MicroCharNet output array shape: {output.shape}")

        expected_channels = 4 + self.num_classes
        raw = output[0]
        if raw.shape[0] == expected_channels:
            predictions = raw.T
        elif raw.shape[1] == expected_channels:
            predictions = raw
        else:
            raise ValueError(
                f"Cannot identify output layout for shape {output.shape}; "
                f"expected an axis of {expected_channels}."
            )

        boxes_xywh = predictions[:, :4].astype(np.float32)
        class_scores = predictions[:, 4:].astype(np.float32)
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]

        candidates: list[dict[str, Any]] = []
        scale = transform["scale"]
        pad_x = transform["pad_x"]
        pad_y = transform["pad_y"]
        source_width = transform["source_width"]
        source_height = transform["source_height"]

        for box, class_id, confidence in zip(boxes_xywh, class_ids, confidences):
            confidence = float(confidence)
            if not np.isfinite(confidence) or confidence < self.conf_threshold:
                continue

            center_x, center_y, width, height = [float(value) for value in box]
            if not all(np.isfinite([center_x, center_y, width, height])):
                continue
            if width <= 0.0 or height <= 0.0:
                continue

            # Current ONNX metadata is an Ultralytics Detect head with
            # end2end=False. The first four output values are decoded absolute
            # xywh coordinates in the letterboxed input image; class values are
            # sigmoid probabilities.
            x1 = (center_x - width / 2.0 - pad_x) / scale
            y1 = (center_y - height / 2.0 - pad_y) / scale
            x2 = (center_x + width / 2.0 - pad_x) / scale
            y2 = (center_y + height / 2.0 - pad_y) / scale

            x1 = float(np.clip(x1, 0.0, source_width))
            y1 = float(np.clip(y1, 0.0, source_height))
            x2 = float(np.clip(x2, 0.0, source_width))
            y2 = float(np.clip(y2, 0.0, source_height))
            if x2 <= x1 or y2 <= y1:
                continue

            class_id = int(class_id)
            candidates.append(
                {
                    "char": self.class_names[class_id].upper(),
                    "class_id": class_id,
                    "conf": confidence,
                    "box": [x1, y1, x2, y2],
                }
            )

        kept = _class_aware_nms(candidates, self.iou_threshold)
        formatted: list[dict[str, Any]] = []
        for detection in kept:
            x1, y1, x2, y2 = detection["box"]
            integer_box = [
                int(np.clip(round(x1), 0, int(source_width) - 1)),
                int(np.clip(round(y1), 0, int(source_height) - 1)),
                int(np.clip(round(x2), 0, int(source_width) - 1)),
                int(np.clip(round(y2), 0, int(source_height) - 1)),
            ]
            if integer_box[2] <= integer_box[0] or integer_box[3] <= integer_box[1]:
                continue
            formatted_detection = dict(detection)
            formatted_detection["box"] = integer_box
            formatted.append(formatted_detection)
        return formatted

    @staticmethod
    def _assemble_result(
        characters: list[dict[str, Any]], timing_ms: dict[str, float]
    ) -> dict[str, Any]:
        lines_as_detections = group_and_sort_characters(characters)
        lines = ["".join(item["char"] for item in line) for line in lines_as_detections]
        text = "".join(lines)
        confidence = (
            float(np.mean([float(item["conf"]) for item in characters])) if characters else 0.0
        )
        return {
            "text": text,
            "confidence": confidence,
            "characters": characters,
            "lines": lines,
            "timing_ms": timing_ms,
        }

    def recognize(self, image: np.ndarray) -> dict[str, Any]:
        """Recognize characters from one OpenCV BGR plate crop."""

        preprocess_start = time.perf_counter()
        tensor, transform = self._preprocess(image)
        preprocess_ms = (time.perf_counter() - preprocess_start) * 1000.0

        inference_start = time.perf_counter()
        outputs = self.session.run([self.output_name], {self.input_name: tensor})
        inference_ms = (time.perf_counter() - inference_start) * 1000.0

        postprocess_start = time.perf_counter()
        characters = self._decode(np.asarray(outputs[0]), transform)
        postprocess_ms = (time.perf_counter() - postprocess_start) * 1000.0
        timing_ms = {
            "preprocess_ms": preprocess_ms,
            "inference_ms": inference_ms,
            "postprocess_ms": postprocess_ms,
            "total_ms": preprocess_ms + inference_ms + postprocess_ms,
        }
        return self._assemble_result(characters, timing_ms)
