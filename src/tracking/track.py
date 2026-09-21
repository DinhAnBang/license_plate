"""Vehicle track state and bounding-box conversions."""

from __future__ import annotations

from enum import Enum, auto

import numpy as np

from src.vehicle_detector import VEHICLE_CLASSES, VehicleDetection

from .kalman_filter import KalmanFilterXYAH


class TrackState(Enum):
    NEW = auto()
    TRACKED = auto()
    LOST = auto()
    REMOVED = auto()


def xyxy_to_xyah(bbox: tuple[int, int, int, int] | np.ndarray) -> np.ndarray:
    """Convert ``(x1, y1, x2, y2)`` to ``(centre_x, centre_y, a, h)``."""

    box = np.asarray(bbox, dtype=np.float64)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise ValueError(f"Invalid xyxy bbox: {box}")
    width = box[2] - box[0]
    height = box[3] - box[1]
    if width <= 0 or height <= 0:
        raise ValueError(f"xyxy bbox must have positive area: {box}")
    return np.array(
        [
            box[0] + width / 2.0,
            box[1] + height / 2.0,
            width / height,
            height,
        ],
        dtype=np.float64,
    )


def xyah_to_xyxy(xyah: np.ndarray) -> np.ndarray:
    """Convert ``(centre_x, centre_y, a, h)`` to ``(x1, y1, x2, y2)``."""

    value = np.asarray(xyah, dtype=np.float64)
    if value.shape != (4,) or not np.isfinite(value).all():
        raise ValueError(f"Invalid xyah bbox: {value}")
    width = max(0.0, value[2] * value[3])
    height = max(0.0, value[3])
    return np.array(
        [
            value[0] - width / 2.0,
            value[1] - height / 2.0,
            value[0] + width / 2.0,
            value[1] + height / 2.0,
        ],
        dtype=np.float64,
    )


class VehicleTrack:
    """Mutable lifecycle object for one physical vehicle."""

    def __init__(self, detection: VehicleDetection) -> None:
        # Validate before retaining the unfiltered public detection.
        xyxy_to_xyah(detection.bbox)
        self.track_id = 0
        # Class is an attribute of a track, not an identity constraint.  Keep
        # the raw detector evidence so a flickering detector label does not
        # flicker the public track label.
        self.class_score_sum: dict[int, float] = {
            class_id: 0.0 for class_id in VEHICLE_CLASSES
        }
        self.class_hit_count: dict[int, int] = {
            class_id: 0 for class_id in VEHICLE_CLASSES
        }
        self._class_last_frame: dict[int, int] = {
            class_id: -1 for class_id in VEHICLE_CLASSES
        }
        self.current_detection_class_id = detection.class_id
        self.current_detection_class_name = detection.class_name
        self._add_class_evidence(detection, frame_index=-1)
        self.class_id = detection.class_id
        self.class_name = detection.class_name
        self.confidence = float(detection.confidence)
        self.mean: np.ndarray | None = None
        self.covariance: np.ndarray | None = None
        self._initial_bbox = np.asarray(detection.bbox, dtype=np.float64)
        self.state = TrackState.NEW
        self.first_frame = -1
        self.last_frame = -1
        self.hits = 0
        self.age = 0
        self.lost_frames = 0
        self.is_activated = False
        self.was_confirmed = False

    @property
    def bbox(self) -> np.ndarray:
        if self.mean is None:
            return self._initial_bbox.copy()
        return xyah_to_xyxy(self.mean[:4])

    @property
    def bbox_as_int(self) -> tuple[int, int, int, int]:
        values = np.rint(self.bbox).astype(np.int64)
        return tuple(int(value) for value in values)

    @property
    def track_length(self) -> int:
        if self.first_frame < 0 or self.last_frame < self.first_frame:
            return 0
        return self.last_frame - self.first_frame + 1

    def activate(
        self,
        kalman_filter: KalmanFilterXYAH,
        frame_index: int,
        track_id: int,
        confirmed: bool = True,
    ) -> None:
        if self.state is not TrackState.NEW or self.track_id != 0:
            raise RuntimeError("Only a NEW track can be activated")
        self.mean, self.covariance = kalman_filter.initiate(
            xyxy_to_xyah(self._initial_bbox)
        )
        self.track_id = track_id
        self.state = TrackState.TRACKED if confirmed else TrackState.NEW
        self.first_frame = frame_index
        self.last_frame = frame_index
        self.hits = 1
        self.age = 1
        self.lost_frames = 0
        self.is_activated = confirmed
        self.was_confirmed = confirmed
        # The constructor already counted the first detection.  Its frame is
        # only known at activation time, so update the timestamp without
        # adding a second vote.
        self._class_last_frame[self.current_detection_class_id] = frame_index
        self._refresh_dominant_class()

    def predict(self, kalman_filter: KalmanFilterXYAH, frame_index: int) -> None:
        if self.mean is None or self.covariance is None:
            raise RuntimeError("Cannot predict an unactivated track")
        predicted_mean = self.mean.copy()
        if self.state is not TrackState.TRACKED:
            predicted_mean[7] = 0.0
        self.mean, self.covariance = kalman_filter.predict(
            predicted_mean, self.covariance
        )
        self.age = max(self.age, frame_index - self.first_frame + 1)
        if self.state is TrackState.LOST:
            self.lost_frames = frame_index - self.last_frame

    def update(
        self,
        kalman_filter: KalmanFilterXYAH,
        detection: VehicleDetection,
        frame_index: int,
        confirmed: bool = True,
    ) -> None:
        if self.mean is None or self.covariance is None:
            raise RuntimeError("Cannot update an unactivated track")
        self.mean, self.covariance = kalman_filter.update(
            self.mean,
            self.covariance,
            xyxy_to_xyah(detection.bbox),
        )
        self.confidence = float(detection.confidence)
        self.state = TrackState.TRACKED if confirmed else TrackState.NEW
        self.is_activated = confirmed
        self.was_confirmed = self.was_confirmed or confirmed
        self.last_frame = frame_index
        self.hits += 1
        self.age = max(self.age, frame_index - self.first_frame + 1)
        self.lost_frames = 0
        self._add_class_evidence(detection, frame_index)

    def re_activate(
        self,
        kalman_filter: KalmanFilterXYAH,
        detection: VehicleDetection,
        frame_index: int,
    ) -> None:
        if self.state is not TrackState.LOST:
            raise RuntimeError("Only a LOST track can be re-activated")
        self.update(kalman_filter, detection, frame_index)

    def mark_lost(self, frame_index: int) -> None:
        if self.state is TrackState.TRACKED:
            self.state = TrackState.LOST
            self.lost_frames = max(1, frame_index - self.last_frame)

    def mark_removed(self) -> None:
        self.state = TrackState.REMOVED
        self.is_activated = False

    def _add_class_evidence(
        self, detection: VehicleDetection, frame_index: int
    ) -> None:
        if detection.class_id not in self.class_score_sum:
            self.class_score_sum[detection.class_id] = 0.0
            self.class_hit_count[detection.class_id] = 0
            self._class_last_frame[detection.class_id] = -1
        self.class_score_sum[detection.class_id] += float(detection.confidence)
        self.class_hit_count[detection.class_id] += 1
        self._class_last_frame[detection.class_id] = frame_index
        self.current_detection_class_id = detection.class_id
        self.current_detection_class_name = detection.class_name
        self._refresh_dominant_class()

    def _refresh_dominant_class(self) -> None:
        """Select a deterministic semantic class from accumulated evidence."""

        dominant_class_id = min(
            self.class_score_sum,
            key=lambda class_id: (
                -self.class_score_sum[class_id],
                -self.class_hit_count[class_id],
                -self._class_last_frame[class_id],
                class_id,
            ),
        )
        self.class_id = dominant_class_id
        self.class_name = VEHICLE_CLASSES.get(
            dominant_class_id, self.current_detection_class_name
        )
