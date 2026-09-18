"""Core license plate detection and image processing components."""

from .detector import Detection, DetectorError, PlateDetector
from .image_processor import ImageProcessingError, ImageProcessor
from .quality import PlateQualityEvaluator, QualityMetrics
from .tracker import PlateTracker, Track, TrackedDetection, calculate_iou
from .video_processor import VideoProcessingError, VideoProcessor

__all__ = [
    "Detection",
    "DetectorError",
    "ImageProcessingError",
    "ImageProcessor",
    "PlateDetector",
    "PlateQualityEvaluator",
    "PlateTracker",
    "QualityMetrics",
    "Track",
    "TrackedDetection",
    "VideoProcessingError",
    "VideoProcessor",
    "calculate_iou",
]
