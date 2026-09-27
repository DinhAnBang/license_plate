"""Image plate-crop evidence objects.

Temporal buffering and Top-K selection belong to the video pipeline and will
be rebuilt under ``src/video``. The image pipeline only needs this immutable
crop record to pass a detected plate into OCR.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .plate_quality import PlateQualityMetrics


@dataclass(frozen=True, slots=True)
class BufferedPlateCandidate:
    frame_index: int
    vehicle_index: int
    plate_class_id: int
    plate_class_name: str
    plate_confidence: float
    bbox: tuple[int, int, int, int]
    quality: PlateQualityMetrics
    crop: np.ndarray

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if not 0.0 <= self.plate_confidence <= 1.0:
            raise ValueError("plate_confidence must be in [0, 1]")
        if len(self.bbox) != 4 or self.bbox[2] <= self.bbox[0] or self.bbox[3] <= self.bbox[1]:
            raise ValueError("bbox must be a valid xyxy box")
        if (
            not isinstance(self.crop, np.ndarray)
            or self.crop.ndim != 3
            or self.crop.shape[2] != 3
            or self.crop.size == 0
        ):
            raise ValueError("crop must be a non-empty HxWx3 NumPy array")
        object.__setattr__(self, "crop", np.ascontiguousarray(self.crop.copy()))


__all__ = ["BufferedPlateCandidate"]
