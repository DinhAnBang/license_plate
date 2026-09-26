"""V5.1 MicroCharNet OCR for retained V4 plate crops.

MicroCharNet is a character detector, not a CTC recognizer.  The preferred
ONNX contract is the exported end-to-end/one-to-one output ``[1, K, 6]`` with
``[x1, y1, x2, y2, confidence, class_id]`` rows.  A raw Ultralytics Detect
output ``[1, 4 + classes, N]`` remains supported as a compatibility fallback.

This module deliberately does not perform temporal fusion or Vietnamese plate
post-processing.
"""

from __future__ import annotations

import ast
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

@dataclass(frozen=True, slots=True)
class OCRResult:
    """Raw OCR result for one crop.

    ``confidence`` is the arithmetic mean of the selected character detection
    confidences.  It is not a temporal or plate-format confidence.  For raw
    fallback output the detections are measured after class-agnostic NMS; for
    end-to-end output they are the model's processed rows.
    """

    text: str
    confidence: float
    char_confidences: tuple[float, ...] | None = None
    raw_indices: tuple[int, ...] = ()
    raw_output_shape: tuple[int, ...] = ()
    status: str = "ok"

    def __post_init__(self) -> None:
        if not isinstance(self.text, str):
            raise TypeError("OCRResult.text must be a string")
        if not np.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("OCRResult.confidence must be finite and in [0, 1]")
        if self.char_confidences is not None:
            if any(
                not np.isfinite(value) or not 0.0 <= value <= 1.0
                for value in self.char_confidences
            ):
                raise ValueError("char_confidences must contain values in [0, 1]")
        if any(not isinstance(index, int) or index < 0 for index in self.raw_indices):
            raise ValueError("raw_indices must contain non-negative integers")


@dataclass(frozen=True, slots=True)
class OCRCharacter:
    """One decoded detector character used for debug/contact-sheet output."""

    char: str
    class_id: int
    confidence: float
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True, slots=True)
class OCRTiming:
    preprocess_ms: float
    inference_ms: float
    decode_ms: float
    total_ms: float


class OCRModelError(RuntimeError):
    """Base error for a model that cannot be safely decoded."""


class CharacterMappingError(OCRModelError):
    """Raised when ONNX does not expose a trustworthy class mapping."""


class UnsupportedOCRModelError(OCRModelError):
    """Raised when the output is not the inspected detector architecture."""


class OutputFormat(str, Enum):
    """Supported MicroCharNet ONNX output contracts."""

    END2END = "end2end"
    RAW = "raw"


@dataclass(slots=True)
class _OCRCounters:
    inference_count: int = 0
    failure_count: int = 0
    preprocess_seconds: float = 0.0
    inference_seconds: float = 0.0
    decode_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class _Transform:
    scale: float
    scale_y: float
    pad_x: int
    pad_y: int
    source_width: int
    source_height: int


@dataclass(frozen=True, slots=True)
class _RawCharacter:
    char: str
    class_id: int
    confidence: float
    box: tuple[float, float, float, float]


def _parse_names_metadata(raw_names: str | None) -> tuple[str, ...]:
    """Parse Ultralytics ``names`` metadata without inventing an alphabet."""

    if not raw_names:
        raise CharacterMappingError(
            "Model output đã xác định nhưng chưa tìm được character mapping. "
            "ONNX metadata does not contain 'names'."
        )
    try:
        parsed = ast.literal_eval(raw_names)
    except (SyntaxError, ValueError) as exc:
        raise CharacterMappingError(
            "Model output đã xác định nhưng character mapping trong ONNX "
            "metadata không parse được."
        ) from exc

    if isinstance(parsed, dict):
        try:
            values = tuple(str(parsed[index]) for index in range(len(parsed)))
        except (KeyError, TypeError) as exc:
            raise CharacterMappingError(
                "Model output đã xác định nhưng class IDs trong character mapping "
                "không liên tục từ 0."
            ) from exc
    elif isinstance(parsed, (list, tuple)):
        values = tuple(str(value) for value in parsed)
    else:
        raise CharacterMappingError(
            "Model output đã xác định nhưng character mapping có định dạng không hỗ trợ."
        )

    if not values or any(len(value) != 1 for value in values):
        raise CharacterMappingError(
            "Model output đã xác định nhưng character mapping không phải ký tự đơn."
        )
    return values


def _dimension_as_int(value: Any) -> int | None:
    return value if isinstance(value, int) and value > 0 else None


def _iou(box: Sequence[float], boxes: np.ndarray) -> np.ndarray:
    left = np.maximum(float(box[0]), boxes[:, 0])
    top = np.maximum(float(box[1]), boxes[:, 1])
    right = np.minimum(float(box[2]), boxes[:, 2])
    bottom = np.minimum(float(box[3]), boxes[:, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    box_area = max(0.0, float(box[2]) - float(box[0])) * max(
        0.0, float(box[3]) - float(box[1])
    )
    other_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + other_area - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0.0,
    )


def _class_agnostic_nms(
    detections: Sequence[_RawCharacter], iou_threshold: float
) -> list[_RawCharacter]:
    """Suppress duplicate physical character locations regardless of class.

    A detector location represents one physical glyph.  If the raw head gives
    that location two class hypotheses (for example ``6`` and ``s``), keeping
    both would manufacture an extra character during string assembly.
    """

    remaining = sorted(detections, key=lambda item: item.confidence, reverse=True)
    kept: list[_RawCharacter] = []
    while remaining:
        best = remaining.pop(0)
        kept.append(best)
        if not remaining:
            continue
        boxes = np.asarray([item.box for item in remaining], dtype=np.float32)
        overlaps = _iou(best.box, boxes)
        remaining = [
            item
            for item, overlap in zip(remaining, overlaps)
            if float(overlap) <= iou_threshold
        ]
    return kept


def _group_and_sort_characters(
    detections: Sequence[OCRCharacter],
) -> list[list[OCRCharacter]]:
    """Order detected characters and separate a genuine second row.

    This is geometry required by a detector-style OCR output.  It is not
    Vietnamese plate validation and does not modify any character.
    """

    if not detections:
        return []
    ordered = sorted(
        detections,
        key=lambda item: ((item.bbox[1] + item.bbox[3]) / 2.0, item.bbox[0]),
    )
    if len(ordered) == 1:
        return [ordered]

    centers_y = np.asarray(
        [(item.bbox[1] + item.bbox[3]) / 2.0 for item in ordered], dtype=np.float32
    )
    heights = np.asarray(
        [max(1.0, item.bbox[3] - item.bbox[1]) for item in ordered], dtype=np.float32
    )
    median_height = max(1.0, float(np.median(heights)))

    # The old rule compared the gap between rows with the full character
    # height.  It missed this valid two-line crop because the glyph boxes are
    # tall and slightly tilted: the row-center gap was just below that
    # threshold even though each row was internally tight.  Compare the
    # candidate row centers with their own robust vertical spread instead.
    split_index: int | None = None
    split_score = float("-inf")
    for candidate_index in range(2, len(ordered) - 1):
        upper = centers_y[:candidate_index]
        lower = centers_y[candidate_index:]
        upper_center = float(np.median(upper))
        lower_center = float(np.median(lower))
        center_gap = lower_center - upper_center
        boundary_gap = float(centers_y[candidate_index] - centers_y[candidate_index - 1])
        upper_spread = float(np.median(np.abs(upper - upper_center)))
        lower_spread = float(np.median(np.abs(lower - lower_center)))
        row_spread = max(1.0, upper_spread, lower_spread)
        minimum_center_gap = max(0.35 * median_height, 2.0 * row_spread)
        if center_gap < minimum_center_gap or boundary_gap < 0.25 * median_height:
            continue
        score = center_gap / row_spread
        if score > split_score:
            split_index = candidate_index
            split_score = score

    if split_index is None:
        groups = [ordered]
    else:
        groups = [ordered[:split_index], ordered[split_index:]]

    for group in groups:
        group.sort(key=lambda item: (item.bbox[0] + item.bbox[2], item.bbox[0]))
    groups.sort(
        key=lambda group: float(
            np.mean([(item.bbox[1] + item.bbox[3]) / 2.0 for item in group])
        )
    )
    return groups


def _as_int_bbox(
    box: Sequence[float], source_width: int, source_height: int
) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = box
    integer = (
        int(np.clip(round(x1), 0, source_width - 1)),
        int(np.clip(round(y1), 0, source_height - 1)),
        int(np.clip(round(x2), 1, source_width)),
        int(np.clip(round(y2), 1, source_height)),
    )
    return integer if integer[2] > integer[0] and integer[3] > integer[1] else None


class MicroCharNetOCR:
    """Load MicroCharNet once and decode its actual detector output."""

    session_init_count = 0

    def __init__(
        self,
        model_path: str | Path = "models/OCR/microcharnet.onnx",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.70,
        providers: Sequence[str] | None = None,
        preprocess_mode: str = "letterbox",
    ) -> None:
        if not 0.0 <= conf_threshold <= 1.0:
            raise ValueError("conf_threshold must be in [0, 1]")
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError("iou_threshold must be in [0, 1]")
        if preprocess_mode not in {"letterbox", "direct_resize"}:
            raise ValueError(
                "preprocess_mode must be 'letterbox' or 'direct_resize'"
            )

        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"OCR model not found: {self.model_path}")
        self.conf_threshold = float(conf_threshold)
        self.iou_threshold = float(iou_threshold)
        self.preprocess_mode = preprocess_mode

        selected_providers = list(providers) if providers else self._default_providers()
        try:
            self.session = ort.InferenceSession(
                str(self.model_path), providers=selected_providers
            )
        except Exception:
            # A machine can advertise CUDA while its CUDA runtime is not
            # usable.  Retry only the safe fallback; do not make V5 depend on
            # a GPU installation.
            if "CUDAExecutionProvider" not in selected_providers:
                raise
            self.session = ort.InferenceSession(
                str(self.model_path), providers=["CPUExecutionProvider"]
            )
        self.session_init_count = 1
        type(self).session_init_count += 1
        self._counters = _OCRCounters()
        self._last_timing = OCRTiming(0.0, 0.0, 0.0, 0.0)

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1:
            raise UnsupportedOCRModelError(
                f"Expected one input, found {len(inputs)}"
            )
        if len(outputs) != 1:
            raise UnsupportedOCRModelError(
                f"Expected one output, found {len(outputs)}"
            )
        self.input_info = inputs[0]
        self.output_info = outputs[0]
        self.input_name = self.input_info.name
        self.output_name = self.output_info.name
        self.input_shape = tuple(self.input_info.shape)
        self.output_shape = tuple(self.output_info.shape)
        self.input_dtype = self.input_info.type
        self.output_dtype = self.output_info.type
        self.providers = tuple(self.session.get_providers())
        self.provider = self.providers[0] if self.providers else "unknown"

        if self.input_dtype != "tensor(float)":
            raise UnsupportedOCRModelError(
                f"Expected float32 input, found {self.input_dtype}"
            )
        if self.output_dtype != "tensor(float)":
            raise UnsupportedOCRModelError(
                f"Expected float32 output, found {self.output_dtype}"
            )
        if len(self.input_shape) != 4 or self.input_shape[1] != 3:
            raise UnsupportedOCRModelError(
                f"Expected NCHW 3-channel input, found {self.input_shape}"
            )
        input_height = _dimension_as_int(self.input_shape[2])
        input_width = _dimension_as_int(self.input_shape[3])
        if input_height is None or input_width is None:
            raise UnsupportedOCRModelError(
                "Dynamic OCR input dimensions have no safe preprocessing size."
            )
        self.input_height = input_height
        self.input_width = input_width

        self.metadata = dict(self.session.get_modelmeta().custom_metadata_map)
        self.class_names = _parse_names_metadata(self.metadata.get("names"))
        self.num_classes = len(self.class_names)
        self.output_format = self._detect_output_format()

    @staticmethod
    def _default_providers() -> list[str]:
        available = ort.get_available_providers()
        providers: list[str] = []
        if "CUDAExecutionProvider" in available:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        return providers

    def _detect_output_format(self) -> OutputFormat:
        """Detect the decoder contract from the real ONNX output shape."""

        if len(self.output_shape) != 3 or self.output_shape[0] != 1:
            raise UnsupportedOCRModelError(
                "Expected one batched 3D OCR output, found "
                f"{self.output_shape}"
            )

        # Ultralytics end-to-end Detect postprocess emits [batch, K, 6]:
        # xyxy, confidence, class index. K is model-defined and must not be
        # hardcoded by the application.
        if self.output_shape[2] == 6:
            if _dimension_as_int(self.output_shape[1]) is None:
                raise UnsupportedOCRModelError(
                    f"Dynamic end-to-end detection count is unsupported for output "
                    f"{self.output_shape}"
                )
            return OutputFormat.END2END

        expected_channels = 4 + self.num_classes
        if self.output_shape[1] != expected_channels:
            raise UnsupportedOCRModelError(
                "Model output is not the inspected MicroCharNet detector layout: "
                f"expected [1, {expected_channels}, N], found {self.output_shape}"
            )
        if _dimension_as_int(self.output_shape[2]) is None:
            raise UnsupportedOCRModelError(
                f"Dynamic detector count is unsupported for output {self.output_shape}"
            )
        return OutputFormat.RAW

    @property
    def session_creation_count(self) -> int:
        """Per-instance diagnostic required by the V5 contract."""

        return self.session_init_count

    @property
    def inference_count(self) -> int:
        return self._counters.inference_count

    @property
    def model_info(self) -> dict[str, object]:
        return {
            "path": self.model_path.as_posix(),
            "provider": self.provider,
            "providers": list(self.providers),
            "input_name": self.input_name,
            "input_shape": list(self.input_shape),
            "input_dtype": self.input_dtype,
            "output_name": self.output_name,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "output_format": self.output_format.value,
            "character_mapping": {
                str(index): char for index, char in enumerate(self.class_names)
            },
            "character_mapping_source": "ONNX metadata.names",
            "architecture": "Ultralytics Detect character detector",
            "output_semantics": (
                "processed xyxy boxes, confidence, and class_id; application "
                "class-agnostic NMS removes duplicate glyph hypotheses"
                if self.output_format is OutputFormat.END2END
                else "decoded absolute xywh boxes plus sigmoid class probabilities; "
                "raw detector fallback, not CTC"
            ),
            "preprocessing": (
                (
                    "BGR uint8 crop -> preserve-aspect-ratio letterbox with value 114 "
                    "-> BGR to RGB -> /255 -> NCHW float32"
                    if self.preprocess_mode == "letterbox"
                    else "EXPERIMENTAL: BGR uint8 crop -> direct resize "
                    "-> BGR to RGB -> /255 -> NCHW float32"
                )
            ),
            "preprocessing_status": (
                "Verified against Ultralytics LetterBox with auto=False, center=True, "
                "scaleup=True, and padding value 114."
            ),
            "preprocess_mode": self.preprocess_mode,
        }

    @property
    def last_timing(self) -> OCRTiming:
        return self._last_timing

    @property
    def timing_totals(self) -> dict[str, float | int]:
        calls = self.inference_count
        return {
            "inference_count": calls,
            "session_init_count": self.session_init_count,
            "preprocess_ms_per_crop": self._counters.preprocess_seconds * 1000.0 / calls
            if calls
            else 0.0,
            "inference_ms_per_crop": self._counters.inference_seconds * 1000.0 / calls
            if calls
            else 0.0,
            "decode_ms_per_crop": self._counters.decode_seconds * 1000.0 / calls
            if calls
            else 0.0,
            "total_ms_per_crop": (
                self._counters.preprocess_seconds
                + self._counters.inference_seconds
                + self._counters.decode_seconds
            )
            * 1000.0
            / calls
            if calls
            else 0.0,
            "failure_count": self._counters.failure_count,
        }

    def preprocess_plate(self, crop: np.ndarray) -> tuple[np.ndarray, _Transform]:
        """Apply the verified MicroCharNet input transform to one BGR crop."""

        if not isinstance(crop, np.ndarray):
            raise TypeError("plate crop must be a NumPy array")
        if crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"plate crop must have shape HxWx3, found {crop.shape}")
        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            raise ValueError("plate crop must be non-empty")
        if crop.dtype != np.uint8:
            raise TypeError(f"plate crop must be uint8 BGR, found {crop.dtype}")

        source_height, source_width = crop.shape[:2]
        scale = min(self.input_width / source_width, self.input_height / source_height)
        if self.preprocess_mode == "direct_resize":
            letterboxed = cv2.resize(
                crop,
                (self.input_width, self.input_height),
                interpolation=cv2.INTER_LINEAR,
            )
            scale_y = self.input_height / source_height
            transform_scale_x = self.input_width / source_width
            pad_left = pad_top = 0
        else:
            resized_width = max(1, round(source_width * scale))
            resized_height = max(1, round(source_height * scale))
            resized = cv2.resize(
                crop,
                (resized_width, resized_height),
                interpolation=cv2.INTER_LINEAR,
            )

            pad_width = self.input_width - resized_width
            pad_height = self.input_height - resized_height
            pad_left = round(pad_width / 2.0 - 0.1)
            pad_right = round(pad_width / 2.0 + 0.1)
            pad_top = round(pad_height / 2.0 - 0.1)
            pad_bottom = round(pad_height / 2.0 + 0.1)
            letterboxed = cv2.copyMakeBorder(
                resized,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(114, 114, 114),
            )
            scale_y = scale
            transform_scale_x = scale
        if letterboxed.shape[:2] != (self.input_height, self.input_width):
            raise RuntimeError(
                "MicroCharNet letterbox produced unexpected shape "
                f"{letterboxed.shape}"
            )

        rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
        tensor = np.ascontiguousarray(
            rgb.astype(np.float32) / 255.0
        ).transpose(2, 0, 1)[None, ...]
        tensor = np.ascontiguousarray(tensor, dtype=np.float32)
        if tensor.shape != (1, 3, self.input_height, self.input_width):
            raise RuntimeError(f"Unexpected OCR tensor shape: {tensor.shape}")
        if tensor.dtype != np.float32 or not np.isfinite(tensor).all():
            raise RuntimeError("OCR preprocessing produced invalid float32 values")
        return tensor, _Transform(
            scale=float(transform_scale_x),
            scale_y=float(scale_y),
            pad_x=int(pad_left),
            pad_y=int(pad_top),
            source_width=int(source_width),
            source_height=int(source_height),
        )

    def _decode(
        self, output: np.ndarray, transform: _Transform
    ) -> tuple[OCRResult, tuple[OCRCharacter, ...]]:
        if self.output_format is OutputFormat.END2END:
            return self._decode_end2end(output, transform)
        return self._decode_raw(output, transform)

    def _build_result(
        self,
        characters: Sequence[OCRCharacter],
        output_shape: Sequence[int],
    ) -> tuple[OCRResult, tuple[OCRCharacter, ...]]:
        """Sort decoded characters and build the common result object."""

        lines = _group_and_sort_characters(characters)
        ordered_characters = tuple(character for line in lines for character in line)
        text = "".join(character.char for character in ordered_characters)
        char_confidences = tuple(
            float(character.confidence) for character in ordered_characters
        )
        raw_indices = tuple(character.class_id for character in ordered_characters)
        confidence = float(np.mean(char_confidences)) if char_confidences else 0.0
        result = OCRResult(
            text=text,
            confidence=confidence,
            char_confidences=char_confidences,
            raw_indices=raw_indices,
            raw_output_shape=tuple(int(value) for value in output_shape),
            status="ok" if ordered_characters else "empty",
        )
        return result, ordered_characters

    def _decode_end2end(
        self, output: np.ndarray, transform: _Transform
    ) -> tuple[OCRResult, tuple[OCRCharacter, ...]]:
        """Decode processed rows and suppress duplicate glyph hypotheses.

        End-to-end detector exports may already contain graph-level NMS, but
        that NMS can be class-aware.  A single physical glyph can therefore
        survive as several classes at the same location (for example ``C``
        and ``0``), which would manufacture extra characters in the assembled
        plate.  Apply the same class-agnostic source-space NMS used by the raw
        decoder before converting boxes to integer debug coordinates.
        """

        output = np.asarray(output)
        if output.ndim != 3 or output.shape[0] != 1 or output.shape[2] != 6:
            raise UnsupportedOCRModelError(
                f"Unexpected end-to-end output shape: {output.shape}; "
                "expected [1, K, 6]"
            )
        if not np.isfinite(output).all():
            raise OCRModelError(
                "MicroCharNet end-to-end output contains non-finite values"
            )

        candidates: list[_RawCharacter] = []
        for row in output[0].astype(np.float32, copy=False):
            x1, y1, x2, y2, confidence_raw, class_id_raw = [
                float(value) for value in row
            ]
            confidence = float(confidence_raw)
            if not np.isfinite(confidence) or confidence < self.conf_threshold:
                continue
            if confidence < 0.0 or confidence > 1.0:
                raise OCRModelError(
                    "MicroCharNet end-to-end confidence is outside [0, 1]"
                )

            class_id = int(round(class_id_raw))
            if not np.isclose(class_id_raw, class_id) or not 0 <= class_id < self.num_classes:
                raise OCRModelError(
                    "MicroCharNet end-to-end class_id is outside the character mapping"
                )
            if not all(np.isfinite((x1, y1, x2, y2))):
                continue

            source_box = (
                float(
                    np.clip(
                        (x1 - transform.pad_x) / transform.scale,
                        0.0,
                        transform.source_width,
                    )
                ),
                float(
                    np.clip(
                        (y1 - transform.pad_y) / transform.scale_y,
                        0.0,
                        transform.source_height,
                    )
                ),
                float(
                    np.clip(
                        (x2 - transform.pad_x) / transform.scale,
                        0.0,
                        transform.source_width,
                    )
                ),
                float(
                    np.clip(
                        (y2 - transform.pad_y) / transform.scale_y,
                        0.0,
                        transform.source_height,
                    )
                ),
            )
            if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
                continue
            candidates.append(
                _RawCharacter(
                    char=self.class_names[class_id],
                    class_id=class_id,
                    confidence=confidence,
                    box=source_box,
                )
            )

        kept = _class_agnostic_nms(candidates, self.iou_threshold)
        characters: list[OCRCharacter] = []
        for item in kept:
            integer_box = _as_int_bbox(
                item.box, transform.source_width, transform.source_height
            )
            if integer_box is None:
                continue
            characters.append(
                OCRCharacter(
                    char=item.char,
                    class_id=item.class_id,
                    confidence=item.confidence,
                    bbox=integer_box,
                )
            )
        return self._build_result(characters, output.shape)

    def _decode_raw(
        self, output: np.ndarray, transform: _Transform
    ) -> tuple[OCRResult, tuple[OCRCharacter, ...]]:
        """Decode raw ``xywh + class probabilities`` with class-agnostic NMS."""

        output = np.asarray(output)
        if (
            output.ndim != 3
            or output.shape[0] != 1
            or output.shape[1] != 4 + self.num_classes
        ):
            raise UnsupportedOCRModelError(
                f"Unexpected raw detector output shape: {output.shape}; "
                f"expected [1, {4 + self.num_classes}, N]"
            )
        if not np.isfinite(output).all():
            raise OCRModelError("MicroCharNet raw output contains non-finite values")

        predictions = output[0].T.astype(np.float32, copy=False)
        boxes_xywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        class_ids = np.argmax(class_scores, axis=1).astype(np.int64)
        confidences = class_scores[np.arange(len(class_scores)), class_ids]
        candidates: list[_RawCharacter] = []
        for box, class_id_raw, confidence_raw in zip(
            boxes_xywh, class_ids, confidences
        ):
            confidence = float(confidence_raw)
            if not np.isfinite(confidence) or confidence < self.conf_threshold:
                continue
            if confidence < 0.0 or confidence > 1.0:
                raise OCRModelError(
                    "MicroCharNet class output is outside [0, 1]; graph semantics "
                    "are not the expected sigmoid probabilities."
                )
            center_x, center_y, width, height = [float(value) for value in box]
            if not all(np.isfinite((center_x, center_y, width, height))):
                continue
            if width <= 0.0 or height <= 0.0:
                continue

            x1 = (center_x - width / 2.0 - transform.pad_x) / transform.scale
            y1 = (center_y - height / 2.0 - transform.pad_y) / transform.scale_y
            x2 = (center_x + width / 2.0 - transform.pad_x) / transform.scale
            y2 = (center_y + height / 2.0 - transform.pad_y) / transform.scale_y
            source_box = (
                float(np.clip(x1, 0.0, transform.source_width)),
                float(np.clip(y1, 0.0, transform.source_height)),
                float(np.clip(x2, 0.0, transform.source_width)),
                float(np.clip(y2, 0.0, transform.source_height)),
            )
            if source_box[2] <= source_box[0] or source_box[3] <= source_box[1]:
                continue
            class_id = int(class_id_raw)
            candidates.append(
                _RawCharacter(
                    char=self.class_names[class_id],
                    class_id=class_id,
                    confidence=confidence,
                    box=source_box,
                )
            )

        kept = _class_agnostic_nms(candidates, self.iou_threshold)
        characters: list[OCRCharacter] = []
        for item in kept:
            integer_box = _as_int_bbox(
                item.box, transform.source_width, transform.source_height
            )
            if integer_box is None:
                continue
            characters.append(
                OCRCharacter(
                    char=item.char,
                    class_id=item.class_id,
                    confidence=item.confidence,
                    bbox=integer_box,
                )
            )
        return self._build_result(characters, output.shape)

    def recognize(self, crop: np.ndarray) -> OCRResult:
        """Run one independent OCR inference on one retained plate crop."""

        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        tensor, transform = self.preprocess_plate(crop)
        preprocess_seconds = time.perf_counter() - preprocess_started

        inference_started = time.perf_counter()
        self._counters.inference_count += 1
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        inference_seconds = time.perf_counter() - inference_started

        decode_started = time.perf_counter()
        result, characters = self._decode(np.asarray(output), transform)
        decode_seconds = time.perf_counter() - decode_started
        del characters

        self._counters.preprocess_seconds += preprocess_seconds
        self._counters.inference_seconds += inference_seconds
        self._counters.decode_seconds += decode_seconds
        total_seconds = time.perf_counter() - started
        self._last_timing = OCRTiming(
            preprocess_ms=preprocess_seconds * 1000.0,
            inference_ms=inference_seconds * 1000.0,
            decode_ms=decode_seconds * 1000.0,
            total_ms=total_seconds * 1000.0,
        )
        return result

    def recognize_with_debug(
        self, crop: np.ndarray
    ) -> tuple[OCRResult, tuple[OCRCharacter, ...], OCRTiming]:
        """Return character boxes for debug rendering without changing OCR text."""

        started = time.perf_counter()
        preprocess_started = time.perf_counter()
        tensor, transform = self.preprocess_plate(crop)
        preprocess_seconds = time.perf_counter() - preprocess_started
        inference_started = time.perf_counter()
        self._counters.inference_count += 1
        output = self.session.run([self.output_name], {self.input_name: tensor})[0]
        inference_seconds = time.perf_counter() - inference_started
        decode_started = time.perf_counter()
        result, characters = self._decode(np.asarray(output), transform)
        decode_seconds = time.perf_counter() - decode_started
        self._counters.preprocess_seconds += preprocess_seconds
        self._counters.inference_seconds += inference_seconds
        self._counters.decode_seconds += decode_seconds
        timing = OCRTiming(
            preprocess_seconds * 1000.0,
            inference_seconds * 1000.0,
            decode_seconds * 1000.0,
            (time.perf_counter() - started) * 1000.0,
        )
        self._last_timing = timing
        return result, characters, timing


__all__ = [
    "CharacterMappingError", "MicroCharNetOCR", "OutputFormat", "OCRCharacter",
    "OCRModelError", "OCRResult", "OCRTiming", "UnsupportedOCRModelError",
]
