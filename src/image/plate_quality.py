"""Explainable plate crop quality scoring for the V4 temporal buffer."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class PlateQualityConfig:
    """Starting constants for ranking crops; they are not trained thresholds."""

    confidence_weight: float = 0.30
    sharpness_weight: float = 0.40
    size_weight: float = 0.20
    exposure_weight: float = 0.10
    sharpness_reference: float = 200.0
    target_plate_height: int = 48
    normalized_sharpness_height: int = 64
    dark_clip_value: int = 3
    bright_clip_value: int = 252

    def __post_init__(self) -> None:
        weights = (
            self.confidence_weight,
            self.sharpness_weight,
            self.size_weight,
            self.exposure_weight,
        )
        if not all(np.isfinite(weight) and weight >= 0.0 for weight in weights):
            raise ValueError("quality weights must be finite and non-negative")
        if not np.isclose(sum(weights), 1.0, rtol=0.0, atol=1e-9):
            raise ValueError("quality weights must sum to 1.0")
        if not np.isfinite(self.sharpness_reference) or self.sharpness_reference <= 0:
            raise ValueError("sharpness_reference must be greater than zero")
        if self.target_plate_height <= 0:
            raise ValueError("target_plate_height must be greater than zero")
        if self.normalized_sharpness_height <= 0:
            raise ValueError("normalized_sharpness_height must be greater than zero")
        if not 0 <= self.dark_clip_value <= 255:
            raise ValueError("dark_clip_value must be in [0, 255]")
        if not 0 <= self.bright_clip_value <= 255:
            raise ValueError("bright_clip_value must be in [0, 255]")
        if self.dark_clip_value >= self.bright_clip_value:
            raise ValueError("dark_clip_value must be less than bright_clip_value")


@dataclass(frozen=True, slots=True)
class PlateQualityMetrics:
    plate_confidence: float
    width: int
    height: int
    sharpness_raw: float
    sharpness_score: float
    size_score: float
    exposure_score: float
    total_score: float


def crop_plate_from_frame(
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> np.ndarray | None:
    """Return an exact, copied global-bbox crop, or ``None`` when invalid.

    Bboxes outside the original image are invalid instead of silently changing
    the detector geometry. No padding or model-resized pixels are introduced.
    """

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        return None
    height, width = frame.shape[:2]
    if height <= 0 or width <= 0 or len(bbox) != 4:
        return None
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height:
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
        return None
    return crop.copy()


def score_plate_quality(
    plate_crop: np.ndarray,
    plate_confidence: float,
    config: PlateQualityConfig | None = None,
) -> PlateQualityMetrics:
    """Score one original-frame BGR crop using deterministic V4 features."""

    config = config or PlateQualityConfig()
    if (
        not isinstance(plate_crop, np.ndarray)
        or plate_crop.ndim != 3
        or plate_crop.shape[2] != 3
        or plate_crop.size == 0
    ):
        raise ValueError("plate_crop must be a non-empty HxWx3 NumPy array")
    if not np.isfinite(plate_confidence) or not 0.0 <= plate_confidence <= 1.0:
        raise ValueError("plate_confidence must be finite and in [0, 1]")

    height, width = plate_crop.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("plate_crop dimensions must be greater than zero")
    gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY)

    normalized_height = config.normalized_sharpness_height
    normalized_width = max(1, int(round(width * normalized_height / height)))
    interpolation = cv2.INTER_AREA if normalized_height < height else cv2.INTER_CUBIC
    normalized = cv2.resize(
        gray,
        (normalized_width, normalized_height),
        interpolation=interpolation,
    )
    sharpness_raw = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    sharpness_score = float(
        np.clip(sharpness_raw / config.sharpness_reference, 0.0, 1.0)
    )
    size_score = float(np.clip(height / config.target_plate_height, 0.0, 1.0))

    dark_ratio = float(np.mean(gray <= config.dark_clip_value))
    bright_ratio = float(np.mean(gray >= config.bright_clip_value))
    clipped_ratio = float(np.clip(dark_ratio + bright_ratio, 0.0, 1.0))
    exposure_score = float(np.clip(1.0 - clipped_ratio, 0.0, 1.0))

    total_score = float(
        np.clip(
            config.confidence_weight * float(plate_confidence)
            + config.sharpness_weight * sharpness_score
            + config.size_weight * size_score
            + config.exposure_weight * exposure_score,
            0.0,
            1.0,
        )
    )
    return PlateQualityMetrics(
        plate_confidence=float(plate_confidence),
        width=int(width),
        height=int(height),
        sharpness_raw=sharpness_raw,
        sharpness_score=sharpness_score,
        size_score=size_score,
        exposure_score=exposure_score,
        total_score=total_score,
    )
