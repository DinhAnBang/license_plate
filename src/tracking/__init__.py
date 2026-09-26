"""ByteTrack-style vehicle tracking components."""

from .byte_tracker import ByteTracker, TrackedVehicle
from .deduplication import (
    DetectionDeduplicationResult,
    deduplicate_vehicle_detections,
)
from .track import TrackRemovalReason, TrackState, VehicleTrack

__all__ = [
    "ByteTracker",
    "TrackedVehicle",
    "TrackState",
    "TrackRemovalReason",
    "VehicleTrack",
    "DetectionDeduplicationResult",
    "deduplicate_vehicle_detections",
]
