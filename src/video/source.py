"""Video input boundary for the rebuilt video pipeline.

This module deliberately knows nothing about vehicles, tracking, plates or
OCR. It only validates the source and exposes decoded frames with stable
indices and timestamps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    source: Path
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration_seconds(self) -> float:
        return self.frame_count / self.fps if self.fps > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class VideoFrame:
    frame_index: int
    timestamp_seconds: float
    image: np.ndarray


class VideoReader:
    """Decode one video sequentially without applying any ALPR logic."""

    def __init__(self, source: str | Path) -> None:
        self.source = Path(source)
        self._capture: cv2.VideoCapture | None = None
        self._metadata: VideoMetadata | None = None

    def __enter__(self) -> "VideoReader":
        if not self.source.is_file():
            raise FileNotFoundError(f"Video does not exist: {self.source}")
        capture = cv2.VideoCapture(str(self.source))
        if not capture.isOpened():
            capture.release()
            raise ValueError(f"Could not open video: {self.source}")

        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        width = int(round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if not np.isfinite(fps) or fps <= 0.0:
            capture.release()
            raise ValueError(f"Video has invalid FPS: {self.source}")
        if frame_count < 0 or width <= 0 or height <= 0:
            capture.release()
            raise ValueError(f"Video has invalid metadata: {self.source}")

        self._capture = capture
        self._metadata = VideoMetadata(
            source=self.source,
            fps=fps,
            frame_count=frame_count,
            width=width,
            height=height,
        )
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        self.close()

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise RuntimeError("VideoReader must be entered before metadata is available")
        return self._metadata

    def __iter__(self) -> Iterator[VideoFrame]:
        if self._capture is None or self._metadata is None:
            raise RuntimeError("VideoReader must be used as a context manager")
        frame_index = 0
        while True:
            decoded, image = self._capture.read()
            if not decoded:
                break
            if not isinstance(image, np.ndarray) or image.size == 0:
                raise ValueError(
                    f"Video returned an invalid frame at index {frame_index}: {self.source}"
                )
            yield VideoFrame(
                frame_index=frame_index,
                timestamp_seconds=frame_index / self._metadata.fps,
                image=image,
            )
            frame_index += 1

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


__all__ = ["VideoFrame", "VideoMetadata", "VideoReader"]
