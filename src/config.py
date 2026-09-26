"""Production defaults shared by the single entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .ocr_fusion import OCRFusionConfig
from .plate_buffer import PlateBufferConfig
from .plate_quality import PlateQualityConfig
from .plate_ownership_temporal import TemporalPlateOwnershipConfig
from .paths import application_directory
from .vn_plate_postprocessor import VietnamPostprocessConfig


_APP_DIRECTORY = application_directory()


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
class TrackingConfig:
    high_threshold: float = 0.25
    new_track_threshold: float = 0.25
    match_cost_threshold: float = 0.80
    second_match_cost_threshold: float = 0.50
    unconfirmed_match_cost_threshold: float = 0.80
    track_buffer_seconds: float = 1.0
    min_confirmed_hits: int = 2
    duplicate_iou_threshold: float = 0.85
    cross_class_duplicate_iou_threshold: float = 0.90
    cross_class_duplicate_area_ratio_threshold: float = 0.80


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
class PipelineConfig:
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    plate: PlateConfig = field(default_factory=PlateConfig)
    ownership: TemporalPlateOwnershipConfig = field(default_factory=TemporalPlateOwnershipConfig)
    quality: PlateQualityConfig = field(default_factory=PlateQualityConfig)
    buffer: PlateBufferConfig = field(default_factory=PlateBufferConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    fusion: OCRFusionConfig = field(default_factory=OCRFusionConfig)
    vietnam: VietnamPostprocessConfig = field(default_factory=VietnamPostprocessConfig)
    low_confidence_threshold: float = 0.50

    def __post_init__(self) -> None:
        if not 0.0 <= self.low_confidence_threshold <= 1.0:
            raise ValueError("low_confidence_threshold must be in [0, 1]")


__all__ = ["VehicleConfig", "TrackingConfig", "PlateConfig", "OCRConfig", "PipelineConfig"]
