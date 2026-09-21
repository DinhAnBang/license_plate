"""Kalman filter used by ByteTrack-style bounding-box tracks."""

from __future__ import annotations

import numpy as np


class KalmanFilterXYAH:
    """Constant-velocity Kalman filter with an eight-dimensional state.

    The state is ``[x, y, a, h, vx, vy, va, vh]`` where ``(x, y)`` is the
    box centre, ``a`` is width divided by height, and ``h`` is box height.
    Measurements contain only ``[x, y, a, h]``.
    """

    ndim = 4
    dt = 1.0

    def __init__(self) -> None:
        self._motion_matrix = np.eye(2 * self.ndim, dtype=np.float64)
        for index in range(self.ndim):
            self._motion_matrix[index, self.ndim + index] = self.dt

        self._update_matrix = np.eye(
            self.ndim, 2 * self.ndim, dtype=np.float64
        )
        self._position_weight = 1.0 / 20.0
        self._velocity_weight = 1.0 / 160.0

    @staticmethod
    def _measurement(measurement: np.ndarray) -> np.ndarray:
        value = np.asarray(measurement, dtype=np.float64)
        if value.shape != (4,):
            raise ValueError(f"Expected an xyah measurement with shape (4,), got {value.shape}")
        if not np.isfinite(value).all() or value[3] <= 0 or value[2] <= 0:
            raise ValueError(f"Invalid xyah measurement: {value}")
        return value

    def initiate(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Create a state distribution from an ``xyah`` measurement."""

        measurement = self._measurement(measurement)
        mean = np.r_[measurement, np.zeros(self.ndim, dtype=np.float64)]
        height = measurement[3]
        standard_deviation = np.array(
            [
                2 * self._position_weight * height,
                2 * self._position_weight * height,
                1e-2,
                2 * self._position_weight * height,
                10 * self._velocity_weight * height,
                10 * self._velocity_weight * height,
                1e-5,
                10 * self._velocity_weight * height,
            ],
            dtype=np.float64,
        )
        covariance = np.diag(np.square(standard_deviation))
        return mean, covariance

    def predict(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run one constant-velocity prediction step."""

        height = max(1e-6, float(mean[3]))
        standard_deviation = np.array(
            [
                self._position_weight * height,
                self._position_weight * height,
                1e-2,
                self._position_weight * height,
                self._velocity_weight * height,
                self._velocity_weight * height,
                1e-5,
                self._velocity_weight * height,
            ],
            dtype=np.float64,
        )
        motion_covariance = np.diag(np.square(standard_deviation))
        predicted_mean = self._motion_matrix @ mean
        predicted_covariance = (
            self._motion_matrix @ covariance @ self._motion_matrix.T
            + motion_covariance
        )
        return predicted_mean, predicted_covariance

    def project(
        self, mean: np.ndarray, covariance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Project a state distribution into measurement space."""

        height = max(1e-6, float(mean[3]))
        standard_deviation = np.array(
            [
                self._position_weight * height,
                self._position_weight * height,
                1e-1,
                self._position_weight * height,
            ],
            dtype=np.float64,
        )
        innovation_covariance = np.diag(np.square(standard_deviation))
        projected_mean = self._update_matrix @ mean
        projected_covariance = (
            self._update_matrix @ covariance @ self._update_matrix.T
            + innovation_covariance
        )
        return projected_mean, projected_covariance

    def update(
        self,
        mean: np.ndarray,
        covariance: np.ndarray,
        measurement: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Correct a predicted state with an ``xyah`` measurement."""

        measurement = self._measurement(measurement)
        projected_mean, projected_covariance = self.project(mean, covariance)
        cross_covariance = covariance @ self._update_matrix.T
        kalman_gain = np.linalg.solve(
            projected_covariance, cross_covariance.T
        ).T
        innovation = measurement - projected_mean
        updated_mean = mean + kalman_gain @ innovation
        updated_covariance = (
            covariance
            - kalman_gain @ projected_covariance @ kalman_gain.T
        )
        # Suppress tiny asymmetry introduced by floating-point arithmetic.
        updated_covariance = (updated_covariance + updated_covariance.T) / 2.0
        return updated_mean, updated_covariance
