"""Core license plate detection and image processing components."""

from .detector import Detection, DetectorError, PlateDetector
from .image_processor import ImageProcessingError, ImageProcessor
from .video_processor import VideoProcessingError, VideoProcessor

__all__ = [
    "Detection",
    "DetectorError",
    "ImageProcessingError",
    "ImageProcessor",
    "PlateDetector",
    "VideoProcessingError",
    "VideoProcessor",
]
