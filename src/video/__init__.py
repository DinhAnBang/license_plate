"""Video source, validation, tracking, and plate-detection stages."""

from .plate_stage import (
    VideoPlateCandidate,
    VideoPlateDecision,
    run_video_plate_detection,
)
from .plate_buffer import VideoBufferedPlate, VideoPlateBuffer, VideoPlateBufferConfig
from .plate_buffer_stage import run_video_plate_buffer
from .plate_quality import (
    VideoPlateQualityConfig,
    VideoPlateQualityMetrics,
)
from .source import VideoFrame, VideoMetadata, VideoReader
from .tracking import TrackedVehicle, VehicleTracker, VehicleTrackingConfig

__all__ = [
    "TrackedVehicle",
    "VehicleTracker",
    "VehicleTrackingConfig",
    "VideoFrame",
    "VideoMetadata",
    "VideoReader",
    "VideoPlateCandidate",
    "VideoPlateDecision",
    "run_video_plate_detection",
    "VideoBufferedPlate",
    "VideoPlateBuffer",
    "VideoPlateBufferConfig",
    "VideoPlateQualityConfig",
    "VideoPlateQualityMetrics",
    "run_video_plate_buffer",
]
