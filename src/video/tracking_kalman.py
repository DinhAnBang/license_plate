"""Constant velocity Kalman filter for video vehicle boxes.

State: centre x/y, aspect ratio, height, and their four velocities.
Measurements contain only the first four values.
"""

from __future__ import annotations

import numpy as np


def xyxy_to_xyah(box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        raise ValueError("vehicle bbox must have positive area")
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, width / height, height], dtype=np.float64)


def xyah_to_xyxy(value: np.ndarray) -> tuple[float, float, float, float]:
    x, y, aspect, height = value[:4]
    width = max(1.0, aspect * height)
    height = max(1.0, height)
    return (x - width / 2, y - height / 2, x + width / 2, y + height / 2)


class KalmanFilterXYAH:
    """One-frame predict and measured-box update."""

    def __init__(self) -> None:
        self.motion = np.eye(8, dtype=np.float64)
        self.motion[:4, 4:] = np.eye(4)
        self.measurement = np.eye(4, 8, dtype=np.float64)
        self.position_weight = 1.0 / 20.0
        self.velocity_weight = 1.0 / 160.0

    def initiate(self, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = max(1.0, float(observed[3]))
        deviations = np.array([
            2 * self.position_weight * height,
            2 * self.position_weight * height,
            0.01,
            2 * self.position_weight * height,
            10 * self.velocity_weight * height,
            10 * self.velocity_weight * height,
            0.00001,
            10 * self.velocity_weight * height,
        ])
        return np.r_[observed, np.zeros(4)], np.diag(deviations ** 2)

    def predict(self, mean: np.ndarray, covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        height = max(1.0, float(mean[3]))
        deviations = np.array([
            self.position_weight * height,
            self.position_weight * height,
            0.01,
            self.position_weight * height,
            self.velocity_weight * height,
            self.velocity_weight * height,
            0.00001,
            self.velocity_weight * height,
        ])
        return self.motion @ mean, self.motion @ covariance @ self.motion.T + np.diag(deviations ** 2)

    def update(
        self, mean: np.ndarray, covariance: np.ndarray, observed: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        height = max(1.0, float(mean[3]))
        deviations = np.array([
            self.position_weight * height,
            self.position_weight * height,
            0.1,
            self.position_weight * height,
        ])
        projected_covariance = (
            self.measurement @ covariance @ self.measurement.T
            + np.diag(deviations ** 2)
        )
        gain = np.linalg.solve(
            projected_covariance, (covariance @ self.measurement.T).T
        ).T
        updated_mean = mean + gain @ (observed - self.measurement @ mean)
        updated_covariance = covariance - gain @ self.measurement @ covariance
        updated_covariance = (updated_covariance + updated_covariance.T) / 2
        return updated_mean, updated_covariance
