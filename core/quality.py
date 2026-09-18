"""Quality metrics for selecting one best license plate crop per track."""

from __future__ import annotations

import math
from typing import Any, TypedDict

import cv2
import numpy as np


class QualityMetrics(TypedDict):
    quality: float
    sharpness: float
    brightness: float
    size: float
    sharpness_raw: float
    brightness_raw: float
    crop_width: int
    crop_height: int
    crop_area: int
    aspect_ratio: float


class PlateQualityEvaluator:
    """Calculate normalized confidence, sharpness, brightness, and size scores."""

    def __init__(
        self,
        confidence_weight: float = 0.30,
        sharpness_weight: float = 0.35,
        brightness_weight: float = 0.15,
        size_weight: float = 0.20,
        sharpness_reference: float = 500.0,
        brightness_target: float = 127.5,
        reference_area: float = 12_000.0,
    ) -> None:
        weights = {
            "confidence": float(confidence_weight),
            "sharpness": float(sharpness_weight),
            "brightness": float(brightness_weight),
            "size": float(size_weight),
        }
        if any(not math.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError("quality weights must be finite and non-negative")
        weight_total = sum(weights.values())
        if weight_total <= 0.0:
            raise ValueError("at least one quality weight must be greater than zero")
        if not math.isfinite(sharpness_reference) or sharpness_reference <= 0.0:
            raise ValueError("sharpness_reference must be greater than zero")
        if not math.isfinite(brightness_target) or not 0.0 <= brightness_target <= 255.0:
            raise ValueError("brightness_target must be between 0.0 and 255.0")
        if not math.isfinite(reference_area) or reference_area <= 0.0:
            raise ValueError("reference_area must be greater than zero")

        self.weights = {name: value / weight_total for name, value in weights.items()}
        self.sharpness_reference = float(sharpness_reference)
        self.brightness_target = float(brightness_target)
        self.reference_area = float(reference_area)

    def evaluate(
        self,
        crop: np.ndarray,
        detection_confidence: float,
    ) -> QualityMetrics:
        """Return normalized quality metrics for one original-frame crop."""

        self._validate_crop(crop)
        try:
            confidence_raw = float(detection_confidence)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("detection_confidence must be numeric") from exc
        if not math.isfinite(confidence_raw):
            raise ValueError("detection_confidence must be finite")

        crop_height, crop_width = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        sharpness_raw = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        sharpness_score = self._clamp01(sharpness_raw / self.sharpness_reference)

        brightness_raw = float(gray.mean())
        max_brightness_distance = max(
            self.brightness_target,
            255.0 - self.brightness_target,
        )
        brightness_score = self._clamp01(
            1.0 - abs(brightness_raw - self.brightness_target) / max_brightness_distance
        )

        crop_area = int(crop_width * crop_height)
        size_score = self._clamp01(crop_area / self.reference_area)
        confidence_score = self._clamp01(confidence_raw)
        quality = self._clamp01(
            self.weights["confidence"] * confidence_score
            + self.weights["sharpness"] * sharpness_score
            + self.weights["brightness"] * brightness_score
            + self.weights["size"] * size_score
        )

        return {
            "quality": quality,
            "sharpness": sharpness_score,
            "brightness": brightness_score,
            "size": size_score,
            "sharpness_raw": sharpness_raw,
            "brightness_raw": brightness_raw,
            "crop_width": int(crop_width),
            "crop_height": int(crop_height),
            "crop_area": crop_area,
            "aspect_ratio": float(crop_width / crop_height),
        }

    @staticmethod
    def _validate_crop(crop: Any) -> None:
        if not isinstance(crop, np.ndarray):
            raise ValueError("crop must be a NumPy array")
        if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
            raise ValueError(f"crop must be a non-empty BGR image, got {crop.shape}")
        if crop.shape[0] <= 0 or crop.shape[1] <= 0:
            raise ValueError(f"crop has invalid dimensions: {crop.shape}")

    @staticmethod
    def _clamp01(value: float) -> float:
        if not math.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, float(value)))
