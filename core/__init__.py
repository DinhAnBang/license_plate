"""Core license plate detection and image processing components."""

from .detector import Detection, DetectorError, PlateDetector
from .byte_tracker import ByteTracker, TrackState
from .image_processor import ImageProcessingError, ImageProcessor
from .quality import PlateQualityEvaluator, QualityMetrics
from .result_serializer import ProductionResultError
from .result_writer import VideoResultError, VideoResultWriter
from .sort_tracker import SortTracker
from .tracklet_stitcher import (
    PlateEvent,
    StitchResult,
    Tracklet,
    TrackletCandidate,
    TrackletStitchConfig,
    TrackletStitcher,
    levenshtein_distance,
)
from .tracker import PlateTracker, Track, TrackedDetection, calculate_iou
from .video_processor import VideoProcessingError, VideoProcessor

__all__ = [
    "Detection",
    "ByteTracker",
    "DetectorError",
    "ImageProcessingError",
    "ImageProcessor",
    "PlateDetector",
    "PlateQualityEvaluator",
    "PlateTracker",
    "SortTracker",
    "PlateEvent",
    "QualityMetrics",
    "ProductionResultError",
    "VideoResultError",
    "VideoResultWriter",
    "Track",
    "Tracklet",
    "TrackletCandidate",
    "TrackletStitchConfig",
    "TrackletStitcher",
    "StitchResult",
    "TrackState",
    "TrackedDetection",
    "VideoProcessingError",
    "VideoProcessor",
    "calculate_iou",
    "levenshtein_distance",
]
