"""Conservative post-tracking stitching of sequential plate tracklets.

This module runs after a tracker has finalized.  It never participates in
frame-level association and intentionally keeps all stitch diagnostics out of
the production JSON boundary.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

from .config import (
    STITCHING_ENABLED,
    STITCH_MAX_CENTER_DISTANCE_RATIO,
    STITCH_MAX_EDIT_DISTANCE,
    STITCH_MAX_GAP_SEC,
    STITCH_MIN_FUZZY_TEXT_LENGTH,
)
from .ocr_voter import OCRVoter
from .plate_normalizer import PlateNormalizer


@dataclass(frozen=True)
class TrackletStitchConfig:
    """Centralized, conservative T4 gates."""

    enabled: bool = STITCHING_ENABLED
    max_gap_sec: float = STITCH_MAX_GAP_SEC
    max_edit_distance: int = STITCH_MAX_EDIT_DISTANCE
    min_fuzzy_text_length: int = STITCH_MIN_FUZZY_TEXT_LENGTH
    max_center_distance_ratio: float = STITCH_MAX_CENTER_DISTANCE_RATIO

    def __post_init__(self) -> None:
        if not math.isfinite(self.max_gap_sec) or self.max_gap_sec < 0.0:
            raise ValueError("max_gap_sec must be finite and non-negative")
        if self.max_edit_distance != 1:
            raise ValueError("T4 baseline requires max_edit_distance == 1")
        if self.min_fuzzy_text_length < 1:
            raise ValueError("min_fuzzy_text_length must be positive")
        if (
            not math.isfinite(self.max_center_distance_ratio)
            or self.max_center_distance_ratio < 0.0
        ):
            raise ValueError(
                "max_center_distance_ratio must be finite and non-negative"
            )


@dataclass(frozen=True)
class TrackletCandidate:
    """One already-scored, already-recognized crop candidate."""

    track_id: int
    frame_index: int
    detection_confidence: float
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
    box: tuple[int, int, int, int]
    crop: Any = field(repr=False, compare=False)
    raw_text: str = ""
    plate_text: str = ""
    ocr_confidence: float = 0.0
    validation_status: str = ""
    validation_score: float = 0.0
    corrections: tuple[str, ...] = ()
    validation_reasons: tuple[str, ...] = ()

    def vote_input(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "text": self.plate_text,
            "ocr_conf": self.ocr_confidence,
            "quality": self.quality,
            "validation_status": self.validation_status,
            "validation_score": self.validation_score,
        }


@dataclass(frozen=True)
class Tracklet:
    """Finalized tracker fragment enriched with real observations and OCR."""

    track_id: int
    first_detected_frame: int
    last_detected_frame: int
    first_box: tuple[int, int, int, int]
    last_box: tuple[int, int, int, int]
    hits: int
    plate_text: str
    ocr_confidence: float
    candidates: tuple[TrackletCandidate, ...]
    selected_candidate: TrackletCandidate
    observation_history: tuple[tuple[int, tuple[int, int, int, int]], ...] = ()


@dataclass
class PlateEvent:
    """Internal final plate event composed of one or more tracklets."""

    event_id: int
    tracklets: list[Tracklet]
    canonical_plate_text: str
    canonical_ocr_confidence: float
    best_candidate: TrackletCandidate
    merge_history: list[dict[str, Any]] = field(default_factory=list)
    canonical_validation_status: str = ""
    canonical_validation_score: float = 0.0

    @property
    def member_track_ids(self) -> list[int]:
        return [tracklet.track_id for tracklet in self.tracklets]

    @property
    def first_detected_frame(self) -> int:
        return self.tracklets[0].first_detected_frame

    @property
    def last_detected_frame(self) -> int:
        return self.tracklets[-1].last_detected_frame

    @property
    def first_box(self) -> tuple[int, int, int, int]:
        return self.tracklets[0].first_box

    @property
    def last_box(self) -> tuple[int, int, int, int]:
        return self.tracklets[-1].last_box

    @property
    def latest_tracklet(self) -> Tracklet:
        return self.tracklets[-1]


@dataclass(frozen=True)
class StitchResult:
    events: tuple[PlateEvent, ...]
    decisions: tuple[dict[str, Any], ...]
    metrics: dict[str, int | float]


def levenshtein_distance(left: str, right: str) -> int:
    """Return full-string edit distance using O(min(n, m)) memory."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


class TrackletStitcher:
    """Build deterministic final plate events from sequential tracklets."""

    def __init__(self, config: TrackletStitchConfig | None = None) -> None:
        self.config = config or TrackletStitchConfig()

    def stitch(self, tracklets: list[Tracklet], fps: float) -> StitchResult:
        if not math.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be a positive finite number")
        started = time.perf_counter()
        ordered = sorted(
            tracklets,
            key=lambda item: (
                item.first_detected_frame,
                item.last_detected_frame,
                item.track_id,
            ),
        )
        events: list[PlateEvent] = []
        decisions: list[dict[str, Any]] = []
        exact_merges = 0
        fuzzy_merges = 0
        rejected_time = 0
        rejected_spatial = 0
        rejected_overlap = 0
        rejected_ocr = 0
        ambiguous_cases = 0

        for tracklet in ordered:
            if not self.config.enabled:
                events.append(self._new_event(len(events) + 1, tracklet))
                continue

            eligible: list[tuple[tuple[float, ...], PlateEvent, dict[str, Any]]] = []
            for event in events:
                diagnostic = self._evaluate(event, tracklet, fps)
                reason = diagnostic["reason"]
                if reason == "REJECT_TIME_GAP":
                    rejected_time += 1
                elif reason == "REJECT_SPATIAL":
                    rejected_spatial += 1
                elif reason == "REJECT_OVERLAP":
                    rejected_overlap += 1
                elif reason == "REJECT_OCR_DISTANCE":
                    rejected_ocr += 1

                if not diagnostic["eligible"]:
                    decisions.append(diagnostic)
                    continue
                rank = (
                    float(diagnostic["edit_distance"]),
                    float(diagnostic["gap_frames"]),
                    float(diagnostic["normalized_center_distance"]),
                    -float(event.canonical_ocr_confidence),
                    float(event.event_id),
                )
                eligible.append((rank, event, diagnostic))

            if not eligible:
                events.append(self._new_event(len(events) + 1, tracklet))
                continue

            eligible.sort(key=lambda item: item[0])
            _, chosen_event, chosen = eligible[0]
            if len(eligible) > 1:
                ambiguous_cases += 1
                decisions.append(
                    {
                        "reason": "AMBIGUOUS_CANDIDATES",
                        "track_id": tracklet.track_id,
                        "candidate_event_ids": [item[1].event_id for item in eligible],
                        "chosen_event_id": chosen_event.event_id,
                    }
                )

            merge_reason = (
                "MERGED_EXACT_OCR"
                if chosen["edit_distance"] == 0
                and chosen["latest_edit_distance"] == 0
                else "MERGED_FUZZY_OCR_DISTANCE_1"
            )
            chosen["reason"] = merge_reason
            chosen["eligible"] = True
            decisions.append(chosen)
            chosen_event.merge_history.append(dict(chosen))
            chosen_event.tracklets.append(tracklet)
            self._refresh_event(chosen_event)
            if merge_reason == "MERGED_EXACT_OCR":
                exact_merges += 1
            else:
                fuzzy_merges += 1

        events.sort(
            key=lambda event: (
                event.first_detected_frame,
                event.best_candidate.frame_index,
                event.event_id,
            )
        )
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics: dict[str, int | float] = {
            "raw_track_count": len(ordered),
            "recognized_tracklets": sum(bool(item.plate_text) for item in ordered),
            "final_plate_event_count": len(events),
            "number_of_merges": len(ordered) - len(events),
            "exact_ocr_merges": exact_merges,
            "fuzzy_distance1_merges": fuzzy_merges,
            "rejected_time": rejected_time,
            "rejected_spatial": rejected_spatial,
            "rejected_overlap": rejected_overlap,
            "rejected_ocr": rejected_ocr,
            "ambiguous_cases": ambiguous_cases,
            "stitching_ms": elapsed_ms,
        }
        return StitchResult(tuple(events), tuple(decisions), metrics)

    @staticmethod
    def _new_event(event_id: int, tracklet: Tracklet) -> PlateEvent:
        event = PlateEvent(
            event_id=event_id,
            tracklets=[tracklet],
            canonical_plate_text=tracklet.plate_text,
            canonical_ocr_confidence=tracklet.ocr_confidence,
            best_candidate=tracklet.selected_candidate,
        )
        refresh_plate_event(event)
        return event

    def _evaluate(
        self, event: PlateEvent, tracklet: Tracklet, fps: float
    ) -> dict[str, Any]:
        latest = event.latest_tracklet
        gap_frames = tracklet.first_detected_frame - latest.last_detected_frame - 1
        gap_sec = (
            tracklet.first_detected_frame - latest.last_detected_frame
        ) / fps
        center_distance, normalized_distance = _center_distance_ratio(
            latest.last_box, tracklet.first_box
        )
        boundary_iou = _box_iou(latest.last_box, tracklet.first_box)
        canonical_distance, canonical_similarity, canonical_pass = self._ocr_gate(
            event.canonical_plate_text, tracklet.plate_text
        )
        latest_distance, latest_similarity, latest_pass = self._ocr_gate(
            latest.plate_text, tracklet.plate_text
        )
        diagnostic: dict[str, Any] = {
            "event_id": event.event_id,
            "member_track_ids": event.member_track_ids,
            "track_id": tracklet.track_id,
            "event_ocr": event.canonical_plate_text,
            "latest_track_ocr": latest.plate_text,
            "tracklet_ocr": tracklet.plate_text,
            "edit_distance": canonical_distance,
            "latest_edit_distance": latest_distance,
            "similarity": canonical_similarity,
            "latest_similarity": latest_similarity,
            "gap_frames": gap_frames,
            "gap_sec": gap_sec,
            "last_box": list(latest.last_box),
            "first_box": list(tracklet.first_box),
            "center_distance": center_distance,
            "normalized_center_distance": normalized_distance,
            "iou": boundary_iou,
            "eligible": False,
        }
        if tracklet.first_detected_frame <= latest.last_detected_frame:
            diagnostic["reason"] = "REJECT_OVERLAP"
        elif gap_sec > self.config.max_gap_sec:
            diagnostic["reason"] = "REJECT_TIME_GAP"
        elif not canonical_pass or not latest_pass:
            diagnostic["reason"] = "REJECT_OCR_DISTANCE"
        elif normalized_distance > self.config.max_center_distance_ratio:
            diagnostic["reason"] = "REJECT_SPATIAL"
        else:
            diagnostic["eligible"] = True
            diagnostic["reason"] = "ELIGIBLE"
        return diagnostic

    def _ocr_gate(self, left: str, right: str) -> tuple[int, float, bool]:
        left = PlateNormalizer.normalize(left)
        right = PlateNormalizer.normalize(right)
        distance = levenshtein_distance(left, right)
        max_length = max(len(left), len(right))
        similarity = 1.0 - distance / max_length if max_length else 0.0
        if not left or not right:
            return distance, similarity, False
        if distance == 0:
            return distance, similarity, True
        fuzzy_length_ok = min(len(left), len(right)) >= self.config.min_fuzzy_text_length
        return (
            distance,
            similarity,
            distance <= self.config.max_edit_distance and fuzzy_length_ok,
        )

    @staticmethod
    def _refresh_event(event: PlateEvent) -> None:
        refresh_plate_event(event)


def refresh_plate_event(event: PlateEvent) -> None:
    """Refresh internal event consensus after T4 or T5 membership changes."""

    candidates = [
        candidate
        for tracklet in event.tracklets
        for candidate in tracklet.candidates
    ]
    vote = OCRVoter.vote([candidate.vote_input() for candidate in candidates])
    event.canonical_plate_text = str(vote["text"])
    event.canonical_ocr_confidence = float(vote["ocr_conf"])
    supporting = [
        candidate
        for candidate in candidates
        if candidate.plate_text == event.canonical_plate_text
    ]
    pool = supporting or candidates
    if pool:
        event.best_candidate = max(pool, key=_candidate_quality_key)

    matching_statuses = [
        candidate.validation_status
        for candidate in supporting
        if candidate.validation_status
    ]
    all_statuses = [candidate.validation_status for candidate in candidates if candidate.validation_status]
    if "VALID" in matching_statuses:
        event.canonical_validation_status = "VALID"
    elif "UNCERTAIN" in matching_statuses:
        event.canonical_validation_status = "UNCERTAIN"
    elif "INVALID" in matching_statuses:
        event.canonical_validation_status = "INVALID"
    elif all_statuses and all(status == "INVALID" for status in all_statuses):
        event.canonical_validation_status = "INVALID"
    else:
        event.canonical_validation_status = str(vote.get("validation_status", ""))
    event.canonical_validation_score = float(vote.get("validation_score", 0.0))


def _candidate_quality_key(candidate: TrackletCandidate) -> tuple[float, float, float, int]:
    return (
        candidate.quality,
        candidate.sharpness,
        candidate.detection_confidence,
        -candidate.frame_index,
    )


def _center_distance_ratio(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> tuple[float, float]:
    left_center = ((left[0] + left[2]) / 2.0, (left[1] + left[3]) / 2.0)
    right_center = ((right[0] + right[2]) / 2.0, (right[1] + right[3]) / 2.0)
    center_distance = math.hypot(
        right_center[0] - left_center[0], right_center[1] - left_center[1]
    )
    left_diag = math.hypot(left[2] - left[0], left[3] - left[1])
    right_diag = math.hypot(right[2] - right[0], right[3] - right[1])
    reference_diag = max(left_diag, right_diag, 1e-9)
    return center_distance, center_distance / reference_diag


def _box_iou(
    left: tuple[int, int, int, int], right: tuple[int, int, int, int]
) -> float:
    intersection_width = max(0.0, min(left[2], right[2]) - max(left[0], right[0]))
    intersection_height = max(0.0, min(left[3], right[3]) - max(left[1], right[1]))
    intersection = intersection_width * intersection_height
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0
