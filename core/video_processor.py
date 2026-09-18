"""Basic video processing built on top of :class:`PlateDetector`."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import DetectorError, PlateDetector
from .quality import PlateQualityEvaluator
from .tracker import PlateTracker


class VideoProcessingError(RuntimeError):
    """Raised when a video cannot be opened, processed, or written."""


@dataclass
class _BestCandidate:
    track_id: int
    best_frame: int
    conf: float
    quality: float
    sharpness: float
    sharpness_raw: float
    brightness: float
    brightness_raw: float
    size: float
    crop_width: int
    crop_height: int
    crop_area: int
    aspect_ratio: float
    box: list[int]
    crop: np.ndarray


class VideoProcessor:
    """Process every frame of a video with one already-created detector."""

    def __init__(
        self,
        detector: PlateDetector,
        output_dir: str | Path = "output",
        project_root: str | Path | None = None,
        tracker: PlateTracker | None = None,
        quality_evaluator: PlateQualityEvaluator | None = None,
    ) -> None:
        self.detector = detector
        self.tracker = tracker
        if quality_evaluator is not None and tracker is None:
            raise ValueError("quality_evaluator requires tracker to be enabled")
        self.quality_evaluator = quality_evaluator
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

        output_suffix = "_tracked.mp4" if self.tracker is not None else "_result.mp4"
        output_path = self.videos_dir / f"{source.stem}{output_suffix}"
        if output_path.resolve() == source:
            raise VideoProcessingError("Output video must not overwrite the input video.")

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            cap.release()
            raise VideoProcessingError(f"Could not open video: {source}")

        writer: cv2.VideoWriter | None = None
        try:
            if self.tracker is not None:
                self.tracker.reset()
            if self.quality_evaluator is not None:
                self._prepare_video_crops(source.stem)
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
            total_tracking_seconds = 0.0
            total_quality_seconds = 0.0
            quality_candidates = 0
            invalid_crops = 0
            best_candidates: dict[int, _BestCandidate] = {}
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

                tracked_detections: Sequence[Mapping[str, Any]] = []
                if self.tracker is not None:
                    tracking_start = time.perf_counter()
                    try:
                        tracked_detections = self.tracker.update(
                            detections,
                            frame_index=processed_frames + 1,
                        )
                    except Exception as exc:
                        raise VideoProcessingError(
                            f"Tracking failed at frame {processed_frames + 1}: {exc}"
                        ) from exc
                    total_tracking_seconds += time.perf_counter() - tracking_start

                    if self.quality_evaluator is not None:
                        quality_start = time.perf_counter()
                        for tracked_detection in tracked_detections:
                            crop = self._crop_from_frame(
                                frame,
                                tracked_detection.get("box"),
                            )
                            if crop is None:
                                invalid_crops += 1
                                continue
                            try:
                                metrics = self.quality_evaluator.evaluate(
                                    crop,
                                    detection_confidence=tracked_detection["conf"],
                                )
                            except ValueError:
                                invalid_crops += 1
                                continue
                            quality_candidates += 1
                            self._update_best_candidate(
                                best_candidates,
                                tracked_detection,
                                metrics,
                                frame_index=processed_frames + 1,
                                crop=crop,
                            )
                        total_quality_seconds += time.perf_counter() - quality_start

                    self._draw_detections(frame, tracked_detections)
                else:
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

            track_summaries: list[dict[str, int]] = []
            if self.tracker is not None:
                self.tracker.finalize()
                track_summaries = self.tracker.track_summaries()

            best_crop_summaries: list[dict[str, Any]] = []
            if self.quality_evaluator is not None:
                best_crop_summaries = self._save_best_crops(
                    source_stem=source.stem,
                    fps=fps,
                    track_summaries=track_summaries,
                    best_candidates=best_candidates,
                )

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
            avg_tracking_ms = (
                (total_tracking_seconds / processed_frames) * 1000.0
                if self.tracker is not None
                else 0.0
            )
            avg_quality_ms = (
                (total_quality_seconds / processed_frames) * 1000.0
                if self.quality_evaluator is not None
                else 0.0
            )
            tracks_with_1_hit = sum(track["hits"] == 1 for track in track_summaries)
            tracks_with_le_2_hits = sum(track["hits"] <= 2 for track in track_summaries)
            tracks_with_ge_3_hits = sum(track["hits"] >= 3 for track in track_summaries)
            longest_track = max(
                track_summaries,
                key=lambda track: (track["hits"], -track["track_id"]),
                default=None,
            )
            best_qualities = [item["quality"] for item in best_crop_summaries]

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
                "tracking_enabled": self.tracker is not None,
                "avg_tracking_ms": avg_tracking_ms,
                "total_tracks": len(track_summaries),
                "tracks_with_1_hit": tracks_with_1_hit,
                "tracks_with_le_2_hits": tracks_with_le_2_hits,
                "tracks_with_ge_3_hits": tracks_with_ge_3_hits,
                "longest_track": longest_track,
                "tracks": track_summaries,
                "quality_enabled": self.quality_evaluator is not None,
                "quality_candidates": quality_candidates,
                "invalid_crops": invalid_crops,
                "avg_quality_ms": avg_quality_ms,
                "tracks_with_best_crop": len(best_crop_summaries),
                "tracks_without_valid_crop": len(track_summaries) - len(best_crop_summaries),
                "min_best_quality": min(best_qualities) if best_qualities else None,
                "max_best_quality": max(best_qualities) if best_qualities else None,
                "avg_best_quality": (
                    sum(best_qualities) / len(best_qualities)
                    if best_qualities
                    else None
                ),
                "best_crops": best_crop_summaries,
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

            track_id = detection.get("track_id")
            if track_id is None:
                label = f"Plate {confidence:.2f}"
            else:
                label = f"Plate #{int(track_id)} | {confidence:.2f}"
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

    def _prepare_video_crops(self, source_stem: str) -> None:
        crops_dir = self.output_dir / "crops"
        try:
            crops_dir.mkdir(parents=True, exist_ok=True)
            for old_crop in crops_dir.glob(f"{source_stem}_track_*.jpg"):
                if old_crop.is_file():
                    old_crop.unlink()
        except OSError as exc:
            raise VideoProcessingError(
                f"Could not prepare video crop directory '{crops_dir}': {exc}"
            ) from exc

    @staticmethod
    def _crop_from_frame(
        frame: np.ndarray,
        box: Any,
    ) -> np.ndarray | None:
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            return None
        try:
            coordinates = [int(round(float(value))) for value in box]
        except (TypeError, ValueError, OverflowError):
            return None
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = coordinates
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width, x2))
        y2 = max(0, min(height, y2))
        if x1 >= x2 or y1 >= y2:
            return None
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] <= 0 or crop.shape[1] <= 0:
            return None
        return crop.copy()

    @staticmethod
    def _update_best_candidate(
        best_candidates: dict[int, _BestCandidate],
        detection: Mapping[str, Any],
        metrics: Mapping[str, Any],
        frame_index: int,
        crop: np.ndarray,
    ) -> None:
        track_id = int(detection["track_id"])
        candidate = _BestCandidate(
            track_id=track_id,
            best_frame=frame_index,
            conf=float(detection["conf"]),
            quality=float(metrics["quality"]),
            sharpness=float(metrics["sharpness"]),
            sharpness_raw=float(metrics["sharpness_raw"]),
            brightness=float(metrics["brightness"]),
            brightness_raw=float(metrics["brightness_raw"]),
            size=float(metrics["size"]),
            crop_width=int(metrics["crop_width"]),
            crop_height=int(metrics["crop_height"]),
            crop_area=int(metrics["crop_area"]),
            aspect_ratio=float(metrics["aspect_ratio"]),
            box=[int(value) for value in detection["box"]],
            crop=crop,
        )
        current = best_candidates.get(track_id)
        if current is None or VideoProcessor._is_better_candidate(candidate, current):
            best_candidates[track_id] = candidate

    @staticmethod
    def _is_better_candidate(
        candidate: _BestCandidate,
        current: _BestCandidate,
    ) -> bool:
        # Deterministic tie-break order: quality, sharpness, confidence, then
        # earlier frame. The frame is not used to influence tracking itself.
        candidate_key = (
            candidate.quality,
            candidate.sharpness,
            candidate.conf,
            -candidate.best_frame,
        )
        current_key = (
            current.quality,
            current.sharpness,
            current.conf,
            -current.best_frame,
        )
        return candidate_key > current_key

    def _save_best_crops(
        self,
        source_stem: str,
        fps: float,
        track_summaries: Sequence[Mapping[str, int]],
        best_candidates: Mapping[int, _BestCandidate],
    ) -> list[dict[str, Any]]:
        crops_dir = self.output_dir / "crops"
        saved: list[dict[str, Any]] = []
        for track in track_summaries:
            track_id = int(track["track_id"])
            candidate = best_candidates.get(track_id)
            if candidate is None:
                continue

            crop_path = crops_dir / f"{source_stem}_track_{track_id:04d}.jpg"
            if not cv2.imwrite(str(crop_path), candidate.crop):
                raise VideoProcessingError(f"Could not save best crop: {crop_path}")
            saved_crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if saved_crop is None or saved_crop.size == 0:
                raise VideoProcessingError(f"Saved best crop could not be reopened: {crop_path}")

            saved.append(
                {
                    "track_id": track_id,
                    "first_frame": int(track["first_frame"]),
                    "last_frame": int(track["last_frame"]),
                    "hits": int(track["hits"]),
                    "best_frame": candidate.best_frame,
                    "best_time": (candidate.best_frame - 1) / fps,
                    "conf": candidate.conf,
                    "quality": candidate.quality,
                    "sharpness": candidate.sharpness,
                    "sharpness_raw": candidate.sharpness_raw,
                    "brightness": candidate.brightness,
                    "brightness_raw": candidate.brightness_raw,
                    "size": candidate.size,
                    "crop_width": candidate.crop_width,
                    "crop_height": candidate.crop_height,
                    "crop_area": candidate.crop_area,
                    "aspect_ratio": candidate.aspect_ratio,
                    "box": candidate.box,
                    "crop": self._project_relative_path(crop_path),
                }
            )
        return saved

    def _project_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()
