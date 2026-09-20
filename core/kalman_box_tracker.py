"""Seven-state linear Kalman filter used by the SORT tracker."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class KalmanConfig:
    """Centralized covariance and noise scales matching the SORT baseline."""

    initial_covariance: float = 10.0
    initial_velocity_uncertainty: float = 1_000.0
    measurement_scale_noise: float = 10.0
    process_velocity_noise: float = 0.01
    process_scale_velocity_noise: float = 0.01


class LinearKalmanFilter:
    """Small NumPy-only linear Kalman filter for SORT's 7D state."""

    state_dimension = 7
    measurement_dimension = 4

    def __init__(self, measurement: np.ndarray, config: KalmanConfig | None = None) -> None:
        self.config = config or KalmanConfig()
        self.x = np.zeros((7, 1), dtype=np.float64)
        self.x[:4] = measurement.reshape(4, 1)

        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[:4, :4] = np.eye(4, dtype=np.float64)

        self.P = np.eye(7, dtype=np.float64) * self.config.initial_covariance
        self.P[4:, 4:] *= self.config.initial_velocity_uncertainty
        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= self.config.process_velocity_noise
        self.Q[6, 6] *= self.config.process_scale_velocity_noise
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= self.config.measurement_scale_noise

    def predict(self) -> bool:
        """Advance one constant-velocity step, returning whether state is finite."""

        if self.x[2, 0] + self.x[6, 0] <= 0.0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return bool(np.all(np.isfinite(self.x)) and np.all(np.isfinite(self.P)))

    def update(self, measurement: np.ndarray) -> bool:
        """Apply one measurement using a numerically stable covariance update."""

        z = np.asarray(measurement, dtype=np.float64).reshape(4, 1)
        if not np.all(np.isfinite(z)):
            return False
        innovation = z - self.H @ self.x
        innovation_covariance = self.H @ self.P @ self.H.T + self.R
        try:
            gain = np.linalg.solve(innovation_covariance.T, (self.P @ self.H.T).T).T
        except np.linalg.LinAlgError:
            gain = self.P @ self.H.T @ np.linalg.pinv(innovation_covariance)

        next_x = self.x + gain @ innovation
        identity = np.eye(7, dtype=np.float64)
        residual = identity - gain @ self.H
        next_p = residual @ self.P @ residual.T + gain @ self.R @ gain.T
        if not np.all(np.isfinite(next_x)) or not np.all(np.isfinite(next_p)):
            return False
        self.x = next_x
        self.P = next_p
        return True


class KalmanBoxTracker:
    """One SORT track with motion state and bounded scalar metadata."""

    def __init__(
        self,
        detection: Mapping[str, Any],
        track_id: int,
        frame_index: int,
        config: KalmanConfig | None = None,
    ) -> None:
        box = normalize_box(detection.get("box"))
        measurement = bbox_to_measurement(box) if box is not None else None
        confidence = _finite_confidence(detection.get("conf"))
        if measurement is None or confidence is None:
            raise ValueError("KalmanBoxTracker requires a valid detection")

        self.track_id = int(track_id)
        self.kf = LinearKalmanFilter(measurement, config=config)
        self.last_detection_box = box
        self.last_detection_conf = confidence
        self.first_frame = int(frame_index)
        self.last_frame = int(frame_index)
        self.age = 0
        self.hits = 1
        self.hit_streak = 1
        self.time_since_update = 0

    @property
    def box(self) -> list[int]:
        """Compatibility view of the latest valid detector box."""

        return self.last_detection_box.copy()

    @property
    def missed(self) -> int:
        """Compatibility alias used by legacy-oriented diagnostics."""

        return self.time_since_update

    def predict(self) -> np.ndarray | None:
        if self.time_since_update > 0:
            self.hit_streak = 0
        if not self.kf.predict():
            return None
        self.age += 1
        self.time_since_update += 1
        return state_to_bbox(self.kf.x)

    def update(self, detection: Mapping[str, Any], frame_index: int) -> bool:
        box = normalize_box(detection.get("box"))
        measurement = bbox_to_measurement(box) if box is not None else None
        confidence = _finite_confidence(detection.get("conf"))
        if measurement is None or confidence is None or not self.kf.update(measurement):
            return False
        self.last_detection_box = box
        self.last_detection_conf = confidence
        self.last_frame = int(frame_index)
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        return True

    def get_state(self) -> np.ndarray | None:
        return state_to_bbox(self.kf.x)


def normalize_box(box: Any) -> list[int] | None:
    if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
        return None
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in values):
        return None
    x1, y1, x2, y2 = [int(round(value)) for value in values]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_to_measurement(box: Sequence[float] | None) -> np.ndarray | None:
    """Convert ``xyxy`` to SORT measurement ``[u, v, s, r]``."""

    if box is None or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value) for value in box]
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None
    width = x2 - x1
    height = y2 - y1
    area = width * height
    if width <= 0.0 or height <= 0.0 or area <= 0.0:
        return None
    return np.asarray(
        [(x1 + x2) / 2.0, (y1 + y2) / 2.0, area, width / height],
        dtype=np.float64,
    ).reshape(4, 1)


def state_to_bbox(state: np.ndarray) -> np.ndarray | None:
    """Convert SORT state ``[u, v, s, r, ...]`` to a finite ``xyxy`` box."""

    values = np.asarray(state, dtype=np.float64).reshape(-1)
    if values.size < 4 or not np.all(np.isfinite(values[:4])):
        return None
    center_x, center_y, area, aspect_ratio = values[:4]
    if area <= 0.0 or aspect_ratio <= 0.0:
        return None
    width_squared = area * aspect_ratio
    if not math.isfinite(width_squared) or width_squared <= 0.0:
        return None
    width = math.sqrt(width_squared)
    height = area / width
    if not math.isfinite(width) or not math.isfinite(height) or width <= 0.0 or height <= 0.0:
        return None
    box = np.asarray(
        [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ],
        dtype=np.float64,
    )
    return box if np.all(np.isfinite(box)) else None


def _finite_confidence(value: Any) -> float | None:
    try:
        confidence = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return confidence if math.isfinite(confidence) else None
