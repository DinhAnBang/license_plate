"""Core license plate detection and image processing components."""

from .detector import Detection, DetectorError, PlateDetector
from .image_processor import ImageProcessingError, ImageProcessor

__all__ = [
    "Detection",
    "DetectorError",
    "ImageProcessingError",
    "ImageProcessor",
    "PlateDetector",
]
