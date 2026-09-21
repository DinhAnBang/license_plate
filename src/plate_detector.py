"""ONNX plate detection and vehicle-ROI coordinate helpers for V3."""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .tracking.byte_tracker import TrackedVehicle


PLATE_CLASSES = {0: "vuong", 1: "dai"}
# Backward-compatible metadata alias used by early V3 diagnostics.
PLATE_TYPES = PLATE_CLASSES


@dataclass(frozen=True, slots=True)
class PlateDetection:
    """One plate candidate in local vehicle-ROI coordinates."""

    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class TrackedPlateCandidate:
    """One best plate candidate with global vehicle and plate coordinates."""

    frame_index: int
    track_id: int
    vehicle_class_id: int
    vehicle_class_name: str
    vehicle_confidence: float
    vehicle_bbox: tuple[int, int, int, int]
    plate_class_id: int
    plate_class_name: str
    plate_confidence: float
    plate_bbox: tuple[int, int, int, int]


# Keep the V3 import name working while exposing the explicit V3.1 concept.
TrackedPlate = TrackedPlateCandidate


@dataclass(frozen=True, slots=True)
class _LetterboxInfo:
    scale: float
    pad_left: int
    pad_top: int
    original_width: int
    original_height: int


def crop_vehicle_roi(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Clamp a global vehicle bbox and return its non-empty frame ROI."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3:
        raise ValueError("frame must be an HxWxC NumPy array")
    frame_height, frame_width = frame.shape[:2]
    if frame_height <= 0 or frame_width <= 0:
        return None

    x1, y1, x2, y2 = (int(value) for value in bbox)
    x1 = int(np.clip(x1, 0, frame_width))
    y1 = int(np.clip(y1, 0, frame_height))
    x2 = int(np.clip(x2, 0, frame_width))
    y2 = int(np.clip(y2, 0, frame_height))
    if x2 <= x1 or y2 <= y1:
        return None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None
    return roi, (x1, y1, x2, y2)


def local_bbox_to_global(
    local_bbox: tuple[int, int, int, int],
    parent_bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    """Translate a vehicle-local plate bbox and clamp it to the full frame."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be greater than zero")
    local_x1, local_y1, local_x2, local_y2 = local_bbox
    parent_x1, parent_y1, _, _ = parent_bbox
    global_x1 = int(np.clip(parent_x1 + local_x1, 0, frame_width - 1))
    global_y1 = int(np.clip(parent_y1 + local_y1, 0, frame_height - 1))
    global_x2 = int(np.clip(parent_x1 + local_x2, 0, frame_width))
    global_y2 = int(np.clip(parent_y1 + local_y2, 0, frame_height))
    return global_x1, global_y1, global_x2, global_y2


def select_best_plate(
    plates: Sequence[PlateDetection],
) -> PlateDetection | None:
    """Select only within the current frame; no temporal quality logic."""

    return max(plates, key=lambda plate: plate.confidence, default=None)


def detect_tracked_plates(
    frame: np.ndarray,
    tracks: Sequence[TrackedVehicle],
    plate_detector: "PlateDetector",
    frame_index: int,
) -> list[TrackedPlateCandidate]:
    """Detect one best plate for each valid confirmed track ROI."""

    frame_height, frame_width = frame.shape[:2]
    results: list[TrackedPlateCandidate] = []
    for track in tracks:
        cropped = crop_vehicle_roi(frame, track.bbox)
        if cropped is None:
            continue
        vehicle_roi, vehicle_bbox = cropped
        best_plate = select_best_plate(plate_detector.detect(vehicle_roi))
        if best_plate is None:
            continue
        global_bbox = local_bbox_to_global(
            best_plate.bbox,
            vehicle_bbox,
            frame_width,
            frame_height,
        )
        if global_bbox[2] <= global_bbox[0] or global_bbox[3] <= global_bbox[1]:
            continue
        results.append(
            TrackedPlateCandidate(
                frame_index=frame_index,
                track_id=track.track_id,
                vehicle_class_id=track.class_id,
                vehicle_class_name=track.class_name,
                vehicle_confidence=track.confidence,
                vehicle_bbox=vehicle_bbox,
                plate_class_id=best_plate.class_id,
                plate_class_name=best_plate.class_name,
                plate_confidence=best_plate.confidence,
                plate_bbox=global_bbox,
            )
        )
    return results


class PlateDetector:
    """Detect Vietnamese square/long plates inside a BGR vehicle ROI."""

    def __init__(
        self,
        model_path: str | Path = "models/plate/best.onnx",
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
            raise FileNotFoundError(f"Plate model not found: {self.model_path}")
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
                "best.onnx is expected to have one input and one output; "
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
            raise ValueError(f"Expected NCHW input [1, 3, H, W], got {shape}")
        height, width = shape[2], shape[3]
        if not isinstance(height, int) or not isinstance(width, int):
            raise ValueError(f"Plate model has dynamic spatial shape: {shape}")
        if height <= 0 or width <= 0:
            raise ValueError(f"Invalid plate model input shape: {shape}")
        return height, width

    def _validate_model_metadata(self) -> None:
        if self.input_info.type != "tensor(float)":
            raise ValueError(f"Expected float32 input, got {self.input_info.type}")
        shape = self.output_info.shape
        # Inspected best.onnx output is [1, 6, 8400]: xywh followed by
        # class scores for {0: vuong, 1: dai}; there is no objectness channel.
        if len(shape) != 3 or shape[1] != 6:
            raise ValueError(
                "Unsupported best.onnx output. Expected [batch, 6, predictions] "
                f"(xywh + 2 class scores), got {shape}"
            )
        if self.output_info.type != "tensor(float)":
            raise ValueError(f"Expected float32 output, got {self.output_info.type}")

    def _print_model_info(self) -> None:
        print("[PlateDetector]")
        print(f"model: {self.model_path}")
        print(f"provider: {self.session.get_providers()[0]}")
        print(f"input_name: {self.input_name}")
        print(f"input_shape: {self.input_info.shape}")
        print(f"output_name: {self.output_name}")
        print(f"output_shape: {self.output_info.shape}")
        print(f"plate_classes: {PLATE_CLASSES}")

    def _preprocess(self, roi: np.ndarray) -> tuple[np.ndarray, _LetterboxInfo]:
        if not isinstance(roi, np.ndarray):
            raise TypeError("vehicle_roi must be a NumPy array")
        if roi.ndim != 3 or roi.shape[2] != 3:
            raise ValueError(
                f"Expected a BGR vehicle ROI with shape HxWx3, got {roi.shape}"
            )
        original_height, original_width = roi.shape[:2]
        if original_height <= 0 or original_width <= 0:
            raise ValueError("Vehicle ROI dimensions must be greater than zero")

        scale = min(
            self.input_width / original_width,
            self.input_height / original_height,
        )
        resized_width = min(self.input_width, round(original_width * scale))
        resized_height = min(self.input_height, round(original_height * scale))
        resized = cv2.resize(
            roi,
            (resized_width, resized_height),
            interpolation=cv2.INTER_LINEAR,
        )

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
        return tensor, _LetterboxInfo(
            scale=scale,
            pad_left=pad_left,
            pad_top=pad_top,
            original_width=original_width,
            original_height=original_height,
        )

    def detect(self, vehicle_roi: np.ndarray) -> list[PlateDetection]:
        """Return all post-NMS plates in local vehicle-ROI coordinates."""

        processing_started = time.perf_counter()
        tensor, letterbox = self._preprocess(vehicle_roi)
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
    ) -> list[PlateDetection]:
        if raw_output.ndim != 3 or raw_output.shape[0] != 1:
            raise ValueError(f"Expected runtime output [1, 6, N], got {raw_output.shape}")
        if raw_output.shape[1] != 6:
            raise ValueError(
                "Expected 6 output channels (xywh + 2 class scores), "
                f"got {raw_output.shape[1]}"
            )

        predictions = raw_output[0].T
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
        confidences = class_scores[
            np.arange(class_scores.shape[0]), class_ids
        ].astype(np.float32)
        confidence_mask = confidences >= self.confidence_threshold
        boxes_xywh = predictions[confidence_mask, :4].astype(np.float32, copy=True)
        filtered_confidences = confidences[confidence_mask]
        filtered_class_ids = class_ids[confidence_mask]
        if len(boxes_xywh) == 0:
            self._print_debug(len(predictions), 0, 0)
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
        kept = self._nms(boxes_xyxy, filtered_confidences, self.iou_threshold)

        detections: list[PlateDetection] = []
        for index in kept:
            bbox = self._round_and_clamp_box(boxes_xyxy[index], letterbox)
            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            class_id = int(filtered_class_ids[index])
            detections.append(
                PlateDetection(
                    class_id=class_id,
                    class_name=PLATE_CLASSES[class_id],
                    confidence=float(filtered_confidences[index]),
                    bbox=bbox,
                )
            )
        detections.sort(key=lambda item: item.confidence, reverse=True)
        self._print_debug(
            len(predictions), int(np.count_nonzero(confidence_mask)), len(detections)
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
        if len(boxes) == 0:
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
            intersection = np.maximum(0.0, intersection_x2 - intersection_x1) * np.maximum(
                0.0, intersection_y2 - intersection_y1
            )
            union = areas[current] + areas[remaining] - intersection
            iou = np.divide(
                intersection,
                union,
                out=np.zeros_like(intersection),
                where=union > 0,
            )
            order = remaining[iou <= iou_threshold]
        return np.asarray(keep, dtype=np.int64)

    def _print_debug(self, raw_count: int, confidence_count: int, nms_count: int) -> None:
        if not self.debug:
            return
        print(f"Raw plate predictions: {raw_count}")
        print(f"After plate confidence filter: {confidence_count}")
        print(f"After plate NMS: {nms_count}")
