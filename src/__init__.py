"""Production ALPR components and reusable detection/OCR APIs."""

from .vehicle_detector import VEHICLE_CLASSES, VehicleDetection, VehicleDetector
from .microcharnet_ocr import MicroCharNetOCR, OCRPlateCandidate, OCRResult

__all__ = [
    "VEHICLE_CLASSES",
    "VehicleDetection",
    "VehicleDetector",
    "MicroCharNetOCR",
    "OCRPlateCandidate",
    "OCRResult",
]
