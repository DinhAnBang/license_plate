"""Basic video processing built on top of :class:`PlateDetector`."""

from __future__ import annotations

import math
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from .detector import DetectorError, PlateDetector
from .byte_tracker import ByteTracker
from .config import (
    BYTE_HIGH_MATCH_IOU_THRESHOLD,
    BYTE_HIGH_THRESHOLD,
    BYTE_LOW_MATCH_IOU_THRESHOLD,
    BYTE_LOW_THRESHOLD,
    BYTE_REFERENCE_FPS,
    BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
    BYTE_UNCONFIRMED_MATCH_IOU_THRESHOLD,
    RECOGNITION_TOP_K,
    SORT_IOU_THRESHOLD,
    SORT_MAX_AGE,
    SORT_MIN_HITS,
)
from .ocr import MicroCharNetOCR
from .ocr_voter import OCRVoter
from .plate_normalizer import PlateNormalizer
from .quality import PlateQualityEvaluator
from .result_writer import VideoResultError, VideoResultWriter
from .sort_tracker import SortTracker
from .tracklet_stitcher import (
    Tracklet,
    TrackletCandidate,
    TrackletStitchConfig,
    TrackletStitcher,
)
from .tracker import PlateTracker

if TYPE_CHECKING:
    from engine.output_manager import RequestOutputPaths


LOGGER = logging.getLogger(__name__)


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


@dataclass
class _TrackObservation:
    first_frame: int
    last_frame: int
    first_box: list[int]
    last_box: list[int]


class VideoProcessor:
    """Process every frame of a video with one already-created detector."""

    def __init__(
        self,
        detector: PlateDetector,
        output_dir: str | Path = "output",
        project_root: str | Path | None = None,
        tracker: PlateTracker | SortTracker | ByteTracker | None = None,
        quality_evaluator: PlateQualityEvaluator | None = None,
        result_writer: VideoResultWriter | None = None,
        ocr: MicroCharNetOCR | None = None,
        top_k: int = RECOGNITION_TOP_K,
        write_json: bool = True,
        tracker_mode: str | None = None,
        sort_iou_threshold: float = SORT_IOU_THRESHOLD,
        sort_max_age: int = SORT_MAX_AGE,
        sort_min_hits: int = SORT_MIN_HITS,
        track_high_thresh: float = BYTE_HIGH_THRESHOLD,
        track_low_thresh: float = BYTE_LOW_THRESHOLD,
        byte_high_match_iou_threshold: float = BYTE_HIGH_MATCH_IOU_THRESHOLD,
        byte_low_match_iou_threshold: float = BYTE_LOW_MATCH_IOU_THRESHOLD,
        byte_unconfirmed_match_iou_threshold: float = BYTE_UNCONFIRMED_MATCH_IOU_THRESHOLD,
        byte_track_buffer: int = BYTE_TRACK_BUFFER_FRAMES_AT_30FPS,
        byte_reference_fps: float = BYTE_REFERENCE_FPS,
        byte_trace_enabled: bool = False,
        stitching_enabled: bool | None = None,
        stitch_config: TrackletStitchConfig | None = None,
    ) -> None:
        self.detector = detector
        if tracker_mode is None:
            if isinstance(tracker, ByteTracker):
                tracker_mode = "byte"
            elif isinstance(tracker, SortTracker):
                tracker_mode = "sort"
            elif isinstance(tracker, PlateTracker):
                tracker_mode = "legacy"
            else:
                tracker_mode = "sort"
        if tracker_mode not in {"legacy", "sort", "byte", "disabled"}:
            raise ValueError("tracker_mode must be 'legacy', 'sort', 'byte', or 'disabled'")
        if tracker_mode == "legacy":
            if tracker is None:
                tracker = PlateTracker(iou_threshold=sort_iou_threshold, max_missed=sort_max_age)
            elif not isinstance(tracker, PlateTracker):
                raise ValueError("tracker_mode='legacy' requires a PlateTracker")
        elif tracker_mode == "sort":
            if tracker is None:
                tracker = SortTracker(
                    iou_threshold=sort_iou_threshold,
                    max_age=sort_max_age,
                    min_hits=sort_min_hits,
                )
            elif not isinstance(tracker, SortTracker) or isinstance(tracker, ByteTracker):
                raise ValueError("tracker_mode='sort' requires a SortTracker")
        elif tracker_mode == "byte":
            if tracker is None:
                tracker = ByteTracker(
                    track_high_thresh=track_high_thresh,
                    track_low_thresh=track_low_thresh,
                    high_match_iou_threshold=byte_high_match_iou_threshold,
                    low_match_iou_threshold=byte_low_match_iou_threshold,
                    unconfirmed_match_iou_threshold=byte_unconfirmed_match_iou_threshold,
                    track_buffer_frames_at_30fps=byte_track_buffer,
                    reference_fps=byte_reference_fps,
                    min_hits=sort_min_hits,
                    trace_enabled=byte_trace_enabled,
                )
            elif not isinstance(tracker, ByteTracker):
                raise ValueError("tracker_mode='byte' requires a ByteTracker")
        elif tracker is not None:
            raise ValueError("tracker_mode='disabled' does not accept a tracker")
        self.tracker_mode = tracker_mode
        self.tracker = tracker
        if quality_evaluator is not None and self.tracker is None:
            raise ValueError("quality_evaluator requires tracker to be enabled")
        if result_writer is not None and quality_evaluator is None:
            raise ValueError("result_writer requires quality_evaluator to be enabled")
        self.quality_evaluator = quality_evaluator
        self.result_writer = result_writer
        self.write_json = write_json
        if stitch_config is None:
            self.stitch_config = (
                TrackletStitchConfig()
                if stitching_enabled is None
                else TrackletStitchConfig(enabled=bool(stitching_enabled))
            )
        elif stitching_enabled is not None and stitch_config.enabled != stitching_enabled:
            raise ValueError("stitching_enabled conflicts with stitch_config.enabled")
        else:
            self.stitch_config = stitch_config
        self.stitching_enabled = self.stitch_config.enabled
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.top_k = top_k
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.ocr = (
            ocr if ocr is not None else MicroCharNetOCR(self.project_root / "models" / "OCR" / "microcharnet.onnx")
        ) if quality_evaluator is not None else None
        self.ocr_calls = 0
        self.ocr_total_ms = 0.0
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        self.output_dir = self.output_dir.resolve()
        self.videos_dir = self.output_dir / "videos"

    def process(
        self, source_path: str | Path, *, output_paths: RequestOutputPaths | None = None
    ) -> dict[str, Any]:
        """Detect and draw all plates in every frame, then write an output video."""

        source = Path(source_path)
        if not source.is_absolute():
            source = self.project_root / source
        source = source.resolve()
        if not source.is_file():
            raise VideoProcessingError(f"Input video does not exist: {source}")
        processing_start = time.perf_counter()

        if output_paths is None:
            try:
                self.videos_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise VideoProcessingError(
                    f"Could not create output directory '{self.videos_dir}': {exc}"
                ) from exc
            # Development-only direct calls retain the Phase 3-10 output names.
            output_suffix = "_tracked.mp4" if self.tracker is not None else "_result.mp4"
            output_path = self.videos_dir / f"{source.stem}{output_suffix}"
        else:
            output_path = output_paths.annotated
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
                if output_paths is None:
                    self._prepare_video_crops(source.stem)
            if self.result_writer is not None:
                self.result_writer.prepare(
                    source, json_path=output_paths.result_json if output_paths is not None else None
                )
            width = self._read_positive_int(cap, cv2.CAP_PROP_FRAME_WIDTH, "width")
            height = self._read_positive_int(cap, cv2.CAP_PROP_FRAME_HEIGHT, "height")
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            frame_count_value = float(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if not math.isfinite(fps) or fps <= 0.0:
                raise VideoProcessingError(
                    f"Video FPS is invalid ({fps!r}); cannot create a reliable output."
                )
            if isinstance(self.tracker, ByteTracker):
                self.tracker.configure_frame_rate(fps)
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
            best_candidates: dict[int, list[_BestCandidate]] = {}
            track_observations: dict[int, _TrackObservation] = {}
            self.ocr_calls = 0
            self.ocr_total_ms = 0.0
            ocr_report: list[dict[str, Any]] = []
            last_progress_bucket = 0
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
                    if isinstance(self.tracker, ByteTracker):
                        detections, detector_stats = self.detector.detect_with_stats(
                            frame,
                            conf_threshold=self.tracker.track_low_thresh,
                        )
                        self.tracker.record_detector_stats(detector_stats)
                    else:
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

                    self._record_track_observations(
                        track_observations,
                        tracked_detections,
                        frame_index=processed_frames + 1,
                    )

                    if self.quality_evaluator is not None:
                        quality_start = time.perf_counter()
                        for tracked_detection in tracked_detections:
                            if not bool(tracked_detection.get("top_k_eligible", True)):
                                continue
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

            if processed_frames == 0:
                raise VideoProcessingError(f"Video contains no readable frames: {source}")

            track_summaries: list[dict[str, int]] = []
            if self.tracker is not None:
                self.tracker.finalize()
                track_summaries = self.tracker.track_summaries()

            best_crop_summaries: list[dict[str, Any]] = []
            stitching_diagnostics: dict[str, Any] = {
                "enabled": self.stitching_enabled,
                "metrics": {
                    "raw_track_count": len(track_summaries),
                    "recognized_tracklets": 0,
                    "final_plate_event_count": 0,
                    "number_of_merges": 0,
                    "exact_ocr_merges": 0,
                    "fuzzy_distance1_merges": 0,
                    "rejected_time": 0,
                    "rejected_spatial": 0,
                    "rejected_overlap": 0,
                    "rejected_ocr": 0,
                    "ambiguous_cases": 0,
                    "stitching_ms": 0.0,
                    "final_crop_count": 0,
                },
                "events": [],
                "decisions": [],
            }
            if self.quality_evaluator is not None:
                best_crop_summaries, stitching_diagnostics = self._save_best_crops(
                    source_stem=source.stem,
                    output_paths=output_paths,
                    fps=fps,
                    track_summaries=track_summaries,
                    best_candidates=best_candidates,
                    track_observations=track_observations,
                    ocr_report=ocr_report,
                )

            total_processing_seconds = time.perf_counter() - processing_start
            average_frame_ms = (total_processing_seconds / processed_frames) * 1000.0

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

            result: dict[str, Any] = {
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
                "tracks_without_valid_crop": len(track_summaries)
                - int(
                    stitching_diagnostics["metrics"].get(
                        "tracklets_with_candidates", 0
                    )
                ),
                "min_best_quality": min(best_qualities) if best_qualities else None,
                "max_best_quality": max(best_qualities) if best_qualities else None,
                "avg_best_quality": (
                    sum(best_qualities) / len(best_qualities)
                    if best_qualities
                    else None
                ),
                "best_crops": best_crop_summaries,
                "ocr_report": ocr_report,
                "ocr_calls": self.ocr_calls,
                "avg_ocr_ms": self.ocr_total_ms / self.ocr_calls if self.ocr_calls else 0.0,
                "stitching": stitching_diagnostics,
            }
            if self.result_writer is not None:
                try:
                    official_result = self.result_writer.build_result(
                        source_path=source,
                        width=width,
                        height=height,
                        fps=fps,
                        frames=processed_frames,
                        plates=best_crop_summaries,
                        request_id=output_paths.request_id if output_paths is not None else None,
                        output_video=output_path,
                        processing_total_ms=total_processing_seconds * 1000.0,
                        average_frame_ms=average_frame_ms,
                    )
                    result["official_result"] = official_result
                    if self.write_json:
                        result["json"] = self.result_writer.write_built_result(
                            source_path=source,
                            result=official_result,
                            json_path=output_paths.result_json if output_paths is not None else None,
                        )
                except VideoResultError as exc:
                    raise VideoProcessingError(f"Could not create official video JSON: {exc}") from exc
            return result
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

    def _update_best_candidate(
        self,
        best_candidates: dict[int, list[_BestCandidate]],
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
        retained = best_candidates.setdefault(track_id, [])
        retained.append(candidate)
        retained.sort(
            key=lambda item: (item.quality, item.sharpness, item.conf, -item.best_frame),
            reverse=True,
        )
        del retained[self.top_k:]

    @staticmethod
    def _record_track_observations(
        observations: dict[int, _TrackObservation],
        detections: Sequence[Mapping[str, Any]],
        frame_index: int,
    ) -> None:
        """Record detector-backed boundary boxes, never Kalman-only predictions."""

        for detection in detections:
            try:
                track_id = int(detection["track_id"])
                box = [int(round(float(value))) for value in detection["box"]]
            except (KeyError, TypeError, ValueError, OverflowError):
                continue
            if len(box) != 4 or box[0] >= box[2] or box[1] >= box[3]:
                continue
            current = observations.get(track_id)
            if current is None:
                observations[track_id] = _TrackObservation(
                    first_frame=frame_index,
                    last_frame=frame_index,
                    first_box=box.copy(),
                    last_box=box.copy(),
                )
            else:
                current.last_frame = frame_index
                current.last_box = box.copy()

    def _save_best_crops(
        self,
        source_stem: str,
        output_paths: RequestOutputPaths | None,
        fps: float,
        track_summaries: Sequence[Mapping[str, int]],
        best_candidates: Mapping[int, list[_BestCandidate]],
        track_observations: Mapping[int, _TrackObservation],
        ocr_report: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        crops_dir = output_paths.crops_dir if output_paths is not None else self.output_dir / "crops"
        tracklets: list[Tracklet] = []
        for track in track_summaries:
            track_id = int(track["track_id"])
            retained = best_candidates.get(track_id, [])
            if not retained:
                continue

            voted_candidates: list[dict[str, Any]] = []
            recognized_candidates: list[TrackletCandidate] = []
            for candidate in retained:
                raw_text = ""
                ocr_conf = 0.0
                if self.ocr is not None:
                    ocr_start = time.perf_counter()
                    self.ocr_calls += 1
                    try:
                        ocr_result = self.ocr.recognize(candidate.crop)
                        raw_text = str(ocr_result["text"])
                        ocr_conf = float(ocr_result["confidence"])
                    except Exception as exc:
                        LOGGER.warning("OCR failed for track %d frame %d: %s", track_id, candidate.best_frame, exc)
                    finally:
                        self.ocr_total_ms += (time.perf_counter() - ocr_start) * 1000.0
                normalized_text = PlateNormalizer.normalize(raw_text)
                if not normalized_text:
                    ocr_conf = 0.0
                vote_input = {
                    "raw_text": raw_text,
                    "text": normalized_text,
                    "ocr_conf": ocr_conf,
                    "quality": candidate.quality,
                }
                voted_candidates.append(vote_input)
                recognized_candidates.append(
                    TrackletCandidate(
                        track_id=track_id,
                        frame_index=candidate.best_frame,
                        detection_confidence=candidate.conf,
                        quality=candidate.quality,
                        sharpness=candidate.sharpness,
                        sharpness_raw=candidate.sharpness_raw,
                        brightness=candidate.brightness,
                        brightness_raw=candidate.brightness_raw,
                        size=candidate.size,
                        crop_width=candidate.crop_width,
                        crop_height=candidate.crop_height,
                        crop_area=candidate.crop_area,
                        aspect_ratio=candidate.aspect_ratio,
                        box=tuple(candidate.box),
                        crop=candidate.crop,
                        raw_text=raw_text,
                        plate_text=normalized_text,
                        ocr_confidence=ocr_conf,
                    )
                )

            vote = OCRVoter.vote(voted_candidates)
            winner_index = vote["winner_index"]
            # If every OCR is empty, keep the highest-quality crop and track.
            selected_index = winner_index if winner_index is not None else 0
            candidate = recognized_candidates[selected_index]
            ocr_report.append({
                "track_id": track_id,
                "candidates_retained": len(retained),
                "candidates": [
                    {
                        "frame": item.best_frame,
                        "quality": item.quality,
                        "det_conf": item.conf,
                        **ocr_item,
                        "vote_weight": OCRVoter.weight(ocr_item) if ocr_item["text"] else 0.0,
                    }
                    for item, ocr_item in zip(retained, voted_candidates)
                ],
                "winner": {
                    "text": vote["text"],
                    "frame": candidate.frame_index,
                    "weight": vote["vote_weight"],
                },
            })
            observation = track_observations.get(track_id)
            if observation is None:
                ordered_candidates = sorted(
                    recognized_candidates, key=lambda item: item.frame_index
                )
                first_box = ordered_candidates[0].box
                last_box = ordered_candidates[-1].box
            else:
                first_box = tuple(observation.first_box)
                last_box = tuple(observation.last_box)
            tracklets.append(
                Tracklet(
                    track_id=track_id,
                    first_detected_frame=int(track["first_frame"]),
                    last_detected_frame=int(track["last_frame"]),
                    first_box=first_box,
                    last_box=last_box,
                    hits=int(track["hits"]),
                    plate_text=str(vote["text"]),
                    ocr_confidence=float(vote["ocr_conf"]),
                    candidates=tuple(recognized_candidates),
                    selected_candidate=candidate,
                )
            )

        stitch_result = TrackletStitcher(self.stitch_config).stitch(tracklets, fps)
        saved: list[dict[str, Any]] = []
        event_diagnostics: list[dict[str, Any]] = []
        for event in stitch_result.events:
            candidate = event.best_candidate
            crop_path = (
                crops_dir / f"track_{candidate.track_id:04d}.jpg"
                if output_paths is not None
                else crops_dir / f"{source_stem}_track_{candidate.track_id:04d}.jpg"
            )
            if not cv2.imwrite(str(crop_path), candidate.crop):
                raise VideoProcessingError(f"Could not save best crop: {crop_path}")
            saved_crop = cv2.imread(str(crop_path), cv2.IMREAD_COLOR)
            if saved_crop is None or saved_crop.size == 0:
                raise VideoProcessingError(f"Saved best crop could not be reopened: {crop_path}")

            event_hits = sum(tracklet.hits for tracklet in event.tracklets)
            saved.append(
                {
                    # Compatibility diagnostic: the final crop's source track.
                    "track_id": candidate.track_id,
                    "event_id": event.event_id,
                    "member_track_ids": event.member_track_ids,
                    "first_frame": event.first_detected_frame,
                    "last_frame": event.last_detected_frame,
                    "hits": event_hits,
                    "best_frame": candidate.frame_index,
                    "best_time": (candidate.frame_index - 1) / fps,
                    "conf": candidate.detection_confidence,
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
                    "box": list(candidate.box),
                    "crop": self._project_relative_path(crop_path),
                    "raw_text": candidate.raw_text,
                    "text": event.canonical_plate_text,
                    # Same final candidate as crop/frame/detection confidence.
                    "ocr_conf": candidate.ocr_confidence,
                }
            )
            event_diagnostics.append(
                {
                    "event_id": event.event_id,
                    "member_track_ids": event.member_track_ids,
                    "first_detected_frame": event.first_detected_frame,
                    "last_detected_frame": event.last_detected_frame,
                    "canonical_plate_text": event.canonical_plate_text,
                    "best_track_id": candidate.track_id,
                    "best_frame_index": candidate.frame_index,
                }
            )

        metrics = dict(stitch_result.metrics)
        metrics["raw_track_count"] = len(track_summaries)
        metrics["tracklets_with_candidates"] = len(tracklets)
        metrics["final_crop_count"] = len(saved)
        diagnostics = {
            "enabled": self.stitching_enabled,
            "config": {
                "max_gap_sec": self.stitch_config.max_gap_sec,
                "max_edit_distance": self.stitch_config.max_edit_distance,
                "min_fuzzy_text_length": self.stitch_config.min_fuzzy_text_length,
                "max_center_distance_ratio": self.stitch_config.max_center_distance_ratio,
            },
            "metrics": metrics,
            "events": event_diagnostics,
            "decisions": [dict(item) for item in stitch_result.decisions],
        }
        return saved, diagnostics

    def _project_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()
