"""Production ALPR components and reusable detection/OCR APIs."""

from .vehicle_detector import VEHICLE_CLASSES, VehicleDetection, VehicleDetector
from .microcharnet_ocr import MicroCharNetOCR, OCRResult
from .ocr_stage import OCRPlateCandidate

__all__ = [
    "VEHICLE_CLASSES",
    "VehicleDetection",
    "VehicleDetector",
    "MicroCharNetOCR",
    "OCRPlateCandidate",
    "OCRResult",
]
