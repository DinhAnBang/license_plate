"""Explainable quality scoring for video plate crops."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoPlateQualityConfig:
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
        if self.sharpness_reference <= 0.0:
            raise ValueError("sharpness_reference must be greater than zero")
        if self.target_plate_height <= 0 or self.normalized_sharpness_height <= 0:
            raise ValueError("quality target heights must be positive")
        if not 0 <= self.dark_clip_value < self.bright_clip_value <= 255:
            raise ValueError("invalid exposure clip values")


@dataclass(frozen=True, slots=True)
class VideoPlateQualityMetrics:
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
    """Return an exact copy of a valid frame bbox without adding pixels."""

    if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
        return None
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in bbox)
    if x1 < 0 or y1 < 0 or x2 > width or y2 > height or x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    return crop.copy() if crop.size else None


def score_plate_quality(
    crop: np.ndarray,
    plate_confidence: float,
    config: VideoPlateQualityConfig | None = None,
) -> VideoPlateQualityMetrics:
    """Score confidence, normalized sharpness, size, and exposure."""

    selected = config or VideoPlateQualityConfig()
    if (
        not isinstance(crop, np.ndarray)
        or crop.ndim != 3
        or crop.shape[2] != 3
        or crop.size == 0
    ):
        raise ValueError("crop must be a non-empty BGR image")
    if not np.isfinite(plate_confidence) or not 0.0 <= plate_confidence <= 1.0:
        raise ValueError("plate_confidence must be finite and in [0, 1]")

    height, width = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Measure focus on the original pixels. Upscaling a tiny crop before the
    # Laplacian can amplify borders and compression noise, making a blurry
    # plate look artificially sharp.
    sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = float(np.clip(sharpness_raw / selected.sharpness_reference, 0.0, 1.0))
    size_score = float(np.clip(height / selected.target_plate_height, 0.0, 1.0))
    clipped_ratio = float(
        np.mean(gray <= selected.dark_clip_value)
        + np.mean(gray >= selected.bright_clip_value)
    )
    exposure_score = float(np.clip(1.0 - clipped_ratio, 0.0, 1.0))
    total_score = float(
        np.clip(
            selected.confidence_weight * plate_confidence
            + selected.sharpness_weight * sharpness_score
            + selected.size_weight * size_score
            + selected.exposure_weight * exposure_score,
            0.0,
            1.0,
        )
    )
    return VideoPlateQualityMetrics(
        plate_confidence=float(plate_confidence),
        width=int(width),
        height=int(height),
        sharpness_raw=sharpness_raw,
        sharpness_score=sharpness_score,
        size_score=size_score,
        exposure_score=exposure_score,
        total_score=total_score,
    )


__all__ = [
    "VideoPlateQualityConfig",
    "VideoPlateQualityMetrics",
    "crop_plate_from_frame",
    "score_plate_quality",
]
