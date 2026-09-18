"""Basic video processing built on top of :class:`PlateDetector`."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import DetectorError, PlateDetector


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be opened, processed, or written."""


class VideoProcessor:
    """Process every frame of a video with one already-created detector."""

    def __init__(
        self,
        detector: PlateDetector,
        output_dir: str | Path = "output",
        project_root: str | Path | None = None,
    ) -> None:
        self.detector = detector
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        self.output_dir = self.output_dir.resolve()
        self.videos_dir = self.output_dir / "videos"

    def process(self, source_path: str | Path) -> dict[str, Any]:
        """Detect and draw all plates in every frame, then write an output video."""

        source = Path(source_path)
        if not source.is_absolute():
            source = self.project_root / source
        source = source.resolve()
        if not source.is_file():
            raise VideoProcessingError(f"Input video does not exist: {source}")

        try:
            self.videos_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VideoProcessingError(
                f"Could not create output directory '{self.videos_dir}': {exc}"
            ) from exc

        output_path = self.videos_dir / f"{source.stem}_result.mp4"
        if output_path.resolve() == source:
            raise VideoProcessingError("Output video must not overwrite the input video.")

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            cap.release()
            raise VideoProcessingError(f"Could not open video: {source}")

        writer: cv2.VideoWriter | None = None
        try:
            width = self._read_positive_int(cap, cv2.CAP_PROP_FRAME_WIDTH, "width")
            height = self._read_positive_int(cap, cv2.CAP_PROP_FRAME_HEIGHT, "height")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count_value = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not math.isfinite(fps) or fps <= 0.0:
                raise VideoProcessingError(
                    f"Video FPS is invalid ({fps!r}); cannot create a reliable output."
                )
            if not math.isfinite(frame_count_value) or frame_count_value < 0.0:
                raise VideoProcessingError(
                    f"Video frame count is invalid ({frame_count_value!r})."
                )

            frame_count = int(round(frame_count_value))
            duration = frame_count / fps if frame_count > 0 else None
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                raise VideoProcessingError(
                    f"Could not open VideoWriter for output: {output_path}"
                )

            processed_frames = 0
            total_detections = 0
            total_detect_seconds = 0.0
            last_progress_bucket = 0
            processing_start = time.perf_counter()

            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame is None or frame.size == 0:
                    raise VideoProcessingError(
                        f"Video returned an empty frame at frame {processed_frames + 1}."
                    )
                frame_height, frame_width = frame.shape[:2]
                if frame_width != width or frame_height != height:
                    raise VideoProcessingError(
                        "Frame resolution differs from video metadata: "
                        f"metadata={width}x{height}, frame={frame_width}x{frame_height}."
                    )

                detect_start = time.perf_counter()
                try:
                    detections = self.detector.detect(frame)
                except DetectorError as exc:
                    raise VideoProcessingError(
                        f"Detection failed at frame {processed_frames + 1}: {exc}"
                    ) from exc
                except Exception as exc:
                    raise VideoProcessingError(
                        f"Unexpected detection error at frame {processed_frames + 1}: {exc}"
                    ) from exc
                total_detect_seconds += time.perf_counter() - detect_start

                self._draw_detections(frame, detections)
                writer.write(frame)
                processed_frames += 1
                total_detections += len(detections)
                last_progress_bucket = self._print_progress(
                    processed_frames,
                    frame_count,
                    last_progress_bucket,
                )

            total_processing_seconds = time.perf_counter() - processing_start
            if processed_frames == 0:
                raise VideoProcessingError(f"Video contains no readable frames: {source}")

            avg_detect_ms = (total_detect_seconds / processed_frames) * 1000.0
            detector_fps = (
                processed_frames / total_detect_seconds
                if total_detect_seconds > 0.0
                else 0.0
            )
            processing_fps = (
                processed_frames / total_processing_seconds
                if total_processing_seconds > 0.0
                else 0.0
            )

            return {
                "status": "ok",
                "source": self._project_relative_path(source),
                "output": self._project_relative_path(output_path),
                "width": width,
                "height": height,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "processed_frames": processed_frames,
                "total_detections": total_detections,
                "avg_detect_ms": avg_detect_ms,
                "detector_fps": detector_fps,
                "total_processing_time": total_processing_seconds,
                "processing_fps": processing_fps,
            }
        finally:
            if writer is not None:
                writer.release()
            cap.release()

    @staticmethod
    def _read_positive_int(
        cap: cv2.VideoCapture,
        property_id: int,
        name: str,
    ) -> int:
        value = float(cap.get(property_id))
        if not math.isfinite(value) or value <= 0.0:
            raise VideoProcessingError(f"Video {name} metadata is invalid: {value!r}")
        return int(round(value))

    @staticmethod
    def _print_progress(
        processed_frames: int,
        frame_count: int,
        last_progress_bucket: int,
    ) -> int:
        if frame_count <= 0:
            if processed_frames % 30 == 0:
                print(f"Processing: frame {processed_frames}")
            return last_progress_bucket

        percent = (processed_frames / frame_count) * 100.0
        bucket = min(20, int(percent // 5))
        if bucket > last_progress_bucket or processed_frames == frame_count:
            print(f"Processing: {percent:5.1f}% | {processed_frames}/{frame_count}")
            return bucket
        return last_progress_bucket

    @classmethod
    def _draw_detections(
        cls,
        frame: np.ndarray,
        detections: Sequence[Mapping[str, Any]],
    ) -> None:
        height, width = frame.shape[:2]
        for detection in detections:
            box = detection.get("box")
            if (
                not isinstance(box, Sequence)
                or isinstance(box, (str, bytes))
                or len(box) != 4
            ):
                raise VideoProcessingError(f"Detector returned invalid box: {detection!r}")
            try:
                coordinates = [int(round(float(value))) for value in box]
                confidence = float(detection["conf"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                raise VideoProcessingError(
                    f"Detector returned invalid detection: {detection!r}"
                ) from exc
            if not math.isfinite(confidence):
                raise VideoProcessingError(f"Detector returned invalid confidence: {detection!r}")

            x1, y1, x2, y2 = coordinates
            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(0, min(width, x2))
            y2 = max(0, min(height, y2))
            if x1 >= x2 or y1 >= y2:
                raise VideoProcessingError(f"Detector returned an empty box: {detection!r}")

            label = f"Plate {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            (text_width, text_height), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )
            label_width = text_width + 6
            label_left = x1
            if label_left + label_width > width:
                label_left = max(0, x2 - label_width)
            label_top = y1 - text_height - baseline - 4
            if label_top < 0:
                label_top = min(height - text_height - baseline - 4, y2 + 2)
            label_top = max(0, label_top)
            label_right = min(width, label_left + label_width)
            label_bottom = min(height, label_top + text_height + baseline + 4)
            cv2.rectangle(
                frame,
                (label_left, label_top),
                (label_right, label_bottom),
                (0, 255, 0),
                thickness=-1,
            )
            cv2.putText(
                frame,
                label,
                (label_left + 3, label_top + text_height + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )

    def _project_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()
