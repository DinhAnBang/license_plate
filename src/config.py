"""Explicit image and video production configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .core.ocr_fusion import OCRFusionConfig
from .core.plate_quality import PlateQualityConfig, VideoPlateQualityConfig
from .paths import application_directory
from .core.plate_postprocess import VietnamPostprocessConfig


_APP_DIRECTORY = application_directory()


@dataclass(frozen=True, slots=True)
class TemporalPlateOwnershipConfig:
    history_size: int = 20
    min_history_samples: int = 5
    definite_duplicate_iou_threshold: float = 0.80
    ownership_conflict_iou_threshold: float = 0.70
    overlap_over_smaller_threshold: float = 0.85
    temporal_weight: float = 0.40
    tight_parent_weight: float = 0.25
    containment_weight: float = 0.15
    plate_conf_weight: float = 0.15
    vehicle_conf_weight: float = 0.05
    cold_start_temporal_score: float = 0.50
    min_history_update_margin: float = 0.10
    center_x_tolerance: float = 0.20
    center_y_tolerance: float = 0.20
    relative_width_tolerance: float = 0.50
    relative_height_tolerance: float = 0.50
    minimum_center_scale: float = 0.05
    minimum_relative_size_scale: float = 0.05
    min_temporal_accept_score: float = 0.20
    max_center_jump: float = 0.25
    max_relative_size_ratio: float = 2.50

    def __post_init__(self) -> None:
        if self.history_size < 1:
            raise ValueError("history_size must be >= 1")
        if self.min_history_samples < 1:
            raise ValueError("min_history_samples must be >= 1")
        if self.min_history_samples > self.history_size:
            raise ValueError("min_history_samples cannot exceed history_size")
        for name in (
            "definite_duplicate_iou_threshold",
            "ownership_conflict_iou_threshold",
            "overlap_over_smaller_threshold",
        ):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must be in (0, 1]")
        if self.ownership_conflict_iou_threshold > self.definite_duplicate_iou_threshold:
            raise ValueError(
                "ownership_conflict_iou_threshold cannot exceed "
                "definite_duplicate_iou_threshold"
            )
        weights = (
            self.temporal_weight,
            self.tight_parent_weight,
            self.containment_weight,
            self.plate_conf_weight,
            self.vehicle_conf_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("owner score weights cannot be negative")
        if not np.isclose(sum(weights), 1.0):
            raise ValueError("owner score weights must sum to 1")
        for name in (
            "cold_start_temporal_score",
            "min_history_update_margin",
            "center_x_tolerance",
            "center_y_tolerance",
            "relative_width_tolerance",
            "relative_height_tolerance",
            "minimum_center_scale",
            "minimum_relative_size_scale",
            "min_temporal_accept_score",
            "max_center_jump",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} cannot be negative")
        if self.min_temporal_accept_score > 1.0:
            raise ValueError("min_temporal_accept_score must be <= 1")
        if not np.isfinite(self.max_relative_size_ratio) or self.max_relative_size_ratio < 1.0:
            raise ValueError("max_relative_size_ratio must be finite and >= 1")


@dataclass(frozen=True, slots=True)
class VehicleTrackingConfig:
    """Detection bands and matching gates; time limits are FPS aware."""

    low_confidence: float = 0.10
    high_confidence: float = 0.50
    new_track_confidence: float = 0.55
    max_lost_seconds: float = 0.50
    first_match_cost_limit: float = 0.83
    second_match_cost_limit: float = 0.65
    min_iou: float = 0.10
    max_center_distance: float = 0.35
    class_mismatch_penalty: float = 0.10
    duplicate_iou_threshold: float = 0.75
    active_duplicate_iou_threshold: float = 0.90
    active_duplicate_min_frames: int = 2
    # Retained for old callers; appearance matching is disabled.
    archive_seconds: float = 10.0
    reidentification_similarity: float = 0.80
    reidentification_min_confidence: float = 0.50

    def __post_init__(self) -> None:
        if not 0 <= self.low_confidence <= self.high_confidence <= self.new_track_confidence <= 1:
            raise ValueError("tracking confidence thresholds must satisfy low <= high <= new <= 1")
        if self.max_lost_seconds <= 0:
            raise ValueError("max_lost_seconds must be positive")
        if not 0 < self.first_match_cost_limit <= 1 or not 0 < self.second_match_cost_limit <= 1:
            raise ValueError("matching cost limits must be in (0, 1]")
        if not 0 < self.min_iou <= 1 or self.max_center_distance <= 0:
            raise ValueError("invalid association geometry limits")
        if not 0 <= self.class_mismatch_penalty < 1:
            raise ValueError("class_mismatch_penalty must be in [0, 1)")
        if not 0 < self.duplicate_iou_threshold <= 1:
            raise ValueError("duplicate_iou_threshold must be in (0, 1]")
        if not 0 < self.active_duplicate_iou_threshold <= 1 or self.active_duplicate_min_frames < 1:
            raise ValueError("invalid active duplicate limits")


@dataclass(frozen=True, slots=True)
class VehicleValidationConfig:
    """Conservative geometry and confidence checks for video detections."""

    min_confidence: float = 0.50
    min_width_pixels: int = 8
    min_height_pixels: int = 8
    min_area_ratio: float = 0.00002
    min_aspect_ratio: float = 0.03
    max_aspect_ratio: float = 12.0
    duplicate_iou_threshold: float = 0.75
    wide_edge_bbox_ratio: float = 0.98
    wide_edge_min_confidence: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_confidence <= 1.0:
            raise ValueError("min_confidence must be between 0 and 1")
        if self.min_width_pixels < 1 or self.min_height_pixels < 1:
            raise ValueError("minimum bbox dimensions must be positive")
        if self.min_area_ratio < 0.0:
            raise ValueError("min_area_ratio must be non-negative")
        if not 0.0 < self.min_aspect_ratio <= self.max_aspect_ratio:
            raise ValueError("invalid aspect ratio limits")
        if not 0.0 <= self.duplicate_iou_threshold <= 1.0:
            raise ValueError("duplicate_iou_threshold must be between 0 and 1")
        if not 0.0 < self.wide_edge_bbox_ratio <= 1.0:
            raise ValueError("wide_edge_bbox_ratio must be between 0 and 1")
        if not 0.0 <= self.wide_edge_min_confidence <= 1.0:
            raise ValueError("wide_edge_min_confidence must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class VideoPlateBufferConfig:
    top_k: int = 4
    min_frame_gap: int = 5
    min_quality_score: float = 0.60
    min_sharpness_score: float = 0.70
    min_crop_width: int = 40
    min_crop_height: int = 24

    def __post_init__(self) -> None:
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.min_frame_gap < 0:
            raise ValueError("min_frame_gap cannot be negative")
        if not 0.0 <= self.min_quality_score <= 1.0:
            raise ValueError("min_quality_score must be between 0 and 1")
        if not 0.0 <= self.min_sharpness_score <= 1.0:
            raise ValueError("min_sharpness_score must be between 0 and 1")
        if self.min_crop_width < 1 or self.min_crop_height < 1:
            raise ValueError("minimum crop dimensions must be positive")


@dataclass(frozen=True, slots=True)
class VehicleConfig:
    model: Path = _APP_DIRECTORY / "models" / "vehicle" / "yolo26n.onnx"
    confidence: float = 0.10
    image_confidence: float = 0.25
    iou: float = 0.45

    def __post_init__(self) -> None:
        if not 0.0 <= self.image_confidence <= 1.0:
            raise ValueError("image_confidence must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class PlateConfig:
    model: Path = _APP_DIRECTORY / "models" / "plate" / "best.onnx"
    confidence: float = 0.25
    iou: float = 0.45


@dataclass(frozen=True, slots=True)
class OCRConfig:
    model: Path = _APP_DIRECTORY / "models" / "OCR" / "microcharnet.onnx"
    confidence: float = 0.25
    iou: float = 0.70


@dataclass(frozen=True, slots=True)
class ImageConfig:
    quality: PlateQualityConfig = field(default_factory=PlateQualityConfig)
    ownership: TemporalPlateOwnershipConfig = field(default_factory=TemporalPlateOwnershipConfig)


@dataclass(frozen=True, slots=True)
class VideoPlateOwnershipConfig:
    conflict_iou_threshold: float = 0.70
    overlap_over_smaller_threshold: float = 0.85


@dataclass(frozen=True, slots=True)
class VideoConfig:
    validation: VehicleValidationConfig = field(default_factory=VehicleValidationConfig)
    tracking: VehicleTrackingConfig = field(default_factory=VehicleTrackingConfig)
    quality: VideoPlateQualityConfig = field(default_factory=VideoPlateQualityConfig)
    topk: VideoPlateBufferConfig = field(default_factory=VideoPlateBufferConfig)
    ownership: VideoPlateOwnershipConfig = field(default_factory=VideoPlateOwnershipConfig)


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    plate: PlateConfig = field(default_factory=PlateConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    fusion: OCRFusionConfig = field(default_factory=OCRFusionConfig)
    vietnam: VietnamPostprocessConfig = field(default_factory=VietnamPostprocessConfig)
    image: ImageConfig = field(default_factory=ImageConfig)
    video: VideoConfig = field(default_factory=VideoConfig)
    low_confidence_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be in [0, 1]")


__all__ = [
    "VehicleConfig", "PlateConfig", "OCRConfig", "ImageConfig", "VideoConfig",
    "VideoPlateOwnershipConfig", "PipelineConfig",
    "VehicleTrackingConfig", "VehicleValidationConfig", "VideoPlateBufferConfig",
    "PlateQualityConfig", "VideoPlateQualityConfig", "OCRFusionConfig",
    "VietnamPostprocessConfig", "TemporalPlateOwnershipConfig",
]
