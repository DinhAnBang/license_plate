"""Plate observation exchanged between detection, ownership and buffering."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImagePlateCandidate:
    frame_index: int
    vehicle_index: int
    vehicle_class_id: int
    vehicle_class_name: str
    vehicle_confidence: float
    vehicle_bbox: tuple[int, int, int, int]
    plate_class_id: int
    plate_class_name: str
    plate_confidence: float
    plate_bbox: tuple[int, int, int, int]
