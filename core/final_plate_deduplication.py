"""Evidence-backed final duplicate removal for T5.

This layer runs after T4.  It never changes frame-level tracking and never
creates a crop.  It only decides whether two internal ``PlateEvent`` objects
describe one physical plate, then lets the caller save the surviving event's
real best candidate once.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

from .config import (
    DUPLICATE_CENTER_DISTANCE_THRESHOLD,
    DUPLICATE_MAX_CENTER_DISTANCE_RATIO,
    DUPLICATE_MAX_GAP_SEC,
    DUPLICATE_MEAN_IOU_THRESHOLD,
    DUPLICATE_MIN_SHARED_FRAMES,
    FINAL_DUPLICATE_MERGE_ENABLED,
    OVERLAP_DUPLICATE_MERGE_ENABLED,
)
from .tracklet_stitcher import (
    PlateEvent,
    _box_iou,
    _center_distance_ratio,
    levenshtein_distance,
    refresh_plate_event,
)


@dataclass(frozen=True)
class FinalPlateDedupConfig:
    enabled: bool = FINAL_DUPLICATE_MERGE_ENABLED
    overlap_enabled: bool = OVERLAP_DUPLICATE_MERGE_ENABLED
    min_shared_frames: int = DUPLICATE_MIN_SHARED_FRAMES
    mean_iou_threshold: float = DUPLICATE_MEAN_IOU_THRESHOLD
    center_distance_threshold: float = DUPLICATE_CENTER_DISTANCE_THRESHOLD
    max_gap_sec: float = DUPLICATE_MAX_GAP_SEC
    max_center_distance_ratio: float = DUPLICATE_MAX_CENTER_DISTANCE_RATIO

    def __post_init__(self) -> None:
        if self.min_shared_frames < 1:
            raise ValueError("min_shared_frames must be positive")
        for name, value in (
            ("mean_iou_threshold", self.mean_iou_threshold),
            ("center_distance_threshold", self.center_distance_threshold),
            ("max_gap_sec", self.max_gap_sec),
            ("max_center_distance_ratio", self.max_center_distance_ratio),
        ):
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.mean_iou_threshold > 1.0:
            raise ValueError("mean_iou_threshold must be <= 1")


@dataclass(frozen=True)
class FinalPlateDeduplicationResult:
    events: tuple[PlateEvent, ...]
    decisions: tuple[dict[str, Any], ...]
    metrics: dict[str, int | float]


class FinalPlateDeduplicator:
    """Merge only events with text, time, and physical-position evidence."""

    def __init__(self, config: FinalPlateDedupConfig | None = None) -> None:
        self.config = config or FinalPlateDedupConfig()

    def deduplicate(
        self, events: Iterable[PlateEvent], fps: float
    ) -> FinalPlateDeduplicationResult:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be a positive finite number")
        started = time.perf_counter()
        ordered = sorted(
            events,
            key=lambda event: (
                event.first_detected_frame,
                event.last_detected_frame,
                event.event_id,
                tuple(event.member_track_ids),
            ),
        )
        if not self.config.enabled:
            metrics = self._metrics(
                ordered,
                merges=0,
                sequential=0,
                overlap=0,
                started=started,
            )
            metrics["duplicate_ms"] = 0.0
            return FinalPlateDeduplicationResult(tuple(ordered), (), metrics)

        output: list[PlateEvent] = []
        decisions: list[dict[str, Any]] = []
        sequential_merges = 0
        overlap_merges = 0

        for incoming in ordered:
            eligible: list[tuple[tuple[float, ...], PlateEvent, dict[str, Any]]] = []
            for existing in output:
                diagnostic = self._evaluate(existing, incoming, fps)
                decisions.append(diagnostic)
                if diagnostic["eligible"]:
                    eligible.append(
                        (
                            (
                                -float(diagnostic["evidence_rank"]),
                                0.0 if diagnostic["overlap"] else float(diagnostic["gap_sec"]),
                                float(diagnostic["mean_center_distance_ratio"]),
                                float(existing.event_id),
                            ),
                            existing,
                            diagnostic,
                        )
                    )
            if not eligible:
                output.append(incoming)
                continue

            _, chosen, diagnostic = min(eligible, key=lambda item: item[0])
            chosen.tracklets.extend(incoming.tracklets)
            chosen.tracklets.sort(
                key=lambda tracklet: (
                    tracklet.first_detected_frame,
                    tracklet.last_detected_frame,
                    tracklet.track_id,
                )
            )
            chosen.merge_history.append(dict(diagnostic))
            refresh_plate_event(chosen)
            if diagnostic["overlap"]:
                overlap_merges += 1
            else:
                sequential_merges += 1
            diagnostic["decision"] = "MERGED"
            diagnostic["reason"] = self._merge_reason(diagnostic)

        output.sort(
            key=lambda event: (
                event.first_detected_frame,
                event.best_candidate.frame_index,
                event.event_id,
            )
        )
        merges = len(ordered) - len(output)
        metrics = self._metrics(
            output,
            merges=merges,
            sequential=sequential_merges,
            overlap=overlap_merges,
            started=started,
        )
        return FinalPlateDeduplicationResult(tuple(output), tuple(decisions), metrics)

    def _evaluate(self, left: PlateEvent, right: PlateEvent, fps: float) -> dict[str, Any]:
        left_texts = self._candidate_texts(left)
        right_texts = self._candidate_texts(right)
        common_texts = sorted(left_texts & right_texts)
        left_text = left.canonical_plate_text
        right_text = right.canonical_plate_text
        edit_distance = levenshtein_distance(left_text, right_text)
        exact_text = bool(left_text and left_text == right_text)
        candidate_consensus = bool(common_texts)
        fuzzy_text = (
            not exact_text
            and not candidate_consensus
            and bool(left_text and right_text)
            and edit_distance == 1
            and min(len(left_text), len(right_text)) >= 6
        )
        evidence_rank = 3 if exact_text else 2 if candidate_consensus else 1 if fuzzy_text else 0
        overlap = right.first_detected_frame <= left.last_detected_frame
        gap_frames = right.first_detected_frame - left.last_detected_frame - 1
        gap_sec = (right.first_detected_frame - left.last_detected_frame) / fps
        shared_frames, ious, centers = self._overlap_stats(left, right)
        mean_iou = sum(ious) / len(ious) if ious else 0.0
        mean_center = sum(centers) / len(centers) if centers else float("inf")
        close_iou_ratio = (
            sum(value >= self.config.mean_iou_threshold for value in ious) / len(ious)
            if ious
            else 0.0
        )
        close_center_ratio = (
            sum(value <= self.config.center_distance_threshold for value in centers) / len(centers)
            if centers
            else 0.0
        )
        physical_overlap = bool(
            shared_frames
            and (
                (mean_iou >= self.config.mean_iou_threshold and close_iou_ratio >= 0.6)
                or (mean_center <= self.config.center_distance_threshold and close_center_ratio >= 0.8)
            )
        )
        temporal_ok = (not overlap and gap_sec <= self.config.max_gap_sec) or (
            overlap and self.config.overlap_enabled
        )
        sequential_spatial_ok = False
        boundary_center = float("inf")
        boundary_ratio = float("inf")
        if not overlap:
            boundary_center, boundary_ratio = _center_distance_ratio(left.last_box, right.first_box)
            sequential_spatial_ok = boundary_ratio <= self.config.max_center_distance_ratio

        eligible = False
        reason = "REJECT_OCR_EVIDENCE"
        if self._invalid_event(left) or self._invalid_event(right):
            reason = "REJECT_INVALID_EVENT"
        elif evidence_rank == 0:
            reason = "REJECT_OCR_EVIDENCE"
        elif overlap:
            if not self.config.overlap_enabled:
                reason = "REJECT_OVERLAP_DISABLED"
            elif evidence_rank < 2:
                # Overlap is the highest-risk case: a one-character OCR
                # distance alone is not enough to identify two nearby plates.
                reason = "REJECT_OCR_EVIDENCE"
            elif not physical_overlap:
                reason = "REJECT_OVERLAP_PHYSICAL_IDENTITY"
            else:
                eligible = True
                reason = "ELIGIBLE_OVERLAP"
        elif gap_sec > self.config.max_gap_sec:
            reason = "REJECT_TIME_GAP"
        elif not sequential_spatial_ok:
            reason = "REJECT_SPATIAL"
        else:
            eligible = True
            reason = "ELIGIBLE_SEQUENTIAL"

        return {
            "event_a": left.event_id,
            "event_b": right.event_id,
            "member_track_ids_a": left.member_track_ids,
            "member_track_ids_b": right.member_track_ids,
            "ocr_final_a": left_text,
            "ocr_final_b": right_text,
            "normalized_a": left_text,
            "normalized_b": right_text,
            "common_candidate_texts": common_texts,
            "edit_distance": edit_distance,
            "frame_range_a": [left.first_detected_frame, left.last_detected_frame],
            "frame_range_b": [right.first_detected_frame, right.last_detected_frame],
            "gap_frames": gap_frames,
            "gap_sec": gap_sec,
            "overlap": overlap,
            "shared_frames": shared_frames,
            "shared_frame_count": len(shared_frames),
            "mean_iou": mean_iou,
            "max_iou": max(ious, default=0.0),
            "mean_center_distance_ratio": mean_center,
            "boundary_center_distance": boundary_center,
            "boundary_center_distance_ratio": boundary_ratio,
            "evidence_rank": evidence_rank,
            "eligible": eligible,
            "decision": "REJECTED",
            "reason": reason,
        }

    @staticmethod
    def _candidate_texts(event: PlateEvent) -> set[str]:
        return {
            candidate.plate_text
            for tracklet in event.tracklets
            for candidate in tracklet.candidates
            if candidate.plate_text
        }

    @staticmethod
    def _invalid_event(event: PlateEvent) -> bool:
        return event.canonical_validation_status == "INVALID"

    @staticmethod
    def _event_boxes(event: PlateEvent) -> dict[int, tuple[int, int, int, int]]:
        boxes: dict[int, tuple[int, int, int, int]] = {}
        for tracklet in event.tracklets:
            for frame, box in tracklet.observation_history:
                boxes[int(frame)] = tuple(int(value) for value in box)
        return boxes

    def _overlap_stats(
        self, left: PlateEvent, right: PlateEvent
    ) -> tuple[list[int], list[float], list[float]]:
        left_boxes = self._event_boxes(left)
        right_boxes = self._event_boxes(right)
        shared = sorted(set(left_boxes) & set(right_boxes))
        ious = [_box_iou(left_boxes[frame], right_boxes[frame]) for frame in shared]
        centers = [
            _center_distance_ratio(left_boxes[frame], right_boxes[frame])[1]
            for frame in shared
        ]
        return shared, ious, centers

    @staticmethod
    def _merge_reason(diagnostic: dict[str, Any]) -> str:
        parts = []
        if diagnostic["normalized_a"] and diagnostic["normalized_a"] == diagnostic["normalized_b"]:
            parts.append("same normalized Vietnamese plate")
        elif diagnostic["common_candidate_texts"]:
            parts.append("strong Top-K OCR consensus")
        else:
            parts.append("one-character OCR distance")
        if diagnostic["overlap"]:
            parts.append("multi-frame physical bbox overlap")
        else:
            parts.append("spatial continuity")
            parts.append("small temporal gap")
        return " + ".join(parts)

    @staticmethod
    def _metrics(
        events: Iterable[PlateEvent],
        *,
        merges: int,
        sequential: int,
        overlap: int,
        started: float,
    ) -> dict[str, int | float]:
        result = list(events)
        return {
            "input_event_count": len(result) + merges,
            "final_event_count": len(result),
            "number_of_merges": merges,
            "sequential_duplicate_merges": sequential,
            "overlapping_duplicate_merges": overlap,
            "duplicate_ms": (time.perf_counter() - started) * 1000.0,
        }


def deduplicate_plate_events(
    events: Iterable[PlateEvent],
    fps: float,
    config: FinalPlateDedupConfig | None = None,
) -> FinalPlateDeduplicationResult:
    """Functional convenience wrapper for tests and benchmark tooling."""

    return FinalPlateDeduplicator(config).deduplicate(events, fps)


__all__ = [
    "FinalPlateDedupConfig",
    "FinalPlateDeduplicationResult",
    "FinalPlateDeduplicator",
    "deduplicate_plate_events",
]
