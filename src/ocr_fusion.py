"""V6 temporal fusion for raw MicroCharNet OCR candidates.

V6 combines the independent OCR observations already retained by V4/V5.  It
does not change OCR detections, apply plate-format rules, or normalize the
result into a Vietnamese plate representation.  The output remains raw text.
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


GAP = "<gap>"


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, float(value)))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if result == result and abs(result) != float("inf") else default


def _as_char_confidences(value: Any) -> tuple[float, ...] | None:
    if value is None:
        return None
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    return tuple(_clamp(_finite_float(item)) for item in value)


@dataclass(frozen=True, slots=True)
class OCRFusionCandidate:
    """The small evidence object consumed by V6."""

    track_id: int
    frame_index: int
    rank: int
    raw_text: str
    ocr_confidence: float
    quality_score: float
    char_confidences: tuple[float, ...] | None = None
    plate_confidence: float | None = None
    status: str = "ok"

    def __post_init__(self) -> None:
        if not isinstance(self.raw_text, str):
            raise TypeError("raw_text must be a string")
        if self.frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if self.rank < 0:
            raise ValueError("rank must be non-negative")
        object.__setattr__(self, "ocr_confidence", _clamp(self.ocr_confidence))
        object.__setattr__(self, "quality_score", _clamp(self.quality_score))
        if self.plate_confidence is not None:
            object.__setattr__(
                self, "plate_confidence", _clamp(self.plate_confidence)
            )

    @property
    def weight(self) -> float:
        """The default V6 evidence weight: OCR confidence times quality."""

        if self.raw_text == "":
            return 0.0
        return _clamp(self.ocr_confidence * self.quality_score)


@dataclass(frozen=True, slots=True)
class OCRFusionConfig:
    """Configurable V6 constants; none encode a plate-format rule."""

    exact_consensus_threshold: float = 0.65
    include_plate_confidence: bool = False
    confidence_consensus_weight: float = 0.50
    confidence_ocr_weight: float = 0.25
    confidence_quality_weight: float = 0.25
    single_candidate_denominator: float = 3.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.exact_consensus_threshold <= 1.0:
            raise ValueError("exact_consensus_threshold must be in [0, 1]")
        weights = (
            self.confidence_consensus_weight,
            self.confidence_ocr_weight,
            self.confidence_quality_weight,
        )
        if any(value < 0.0 for value in weights) or abs(sum(weights) - 1.0) > 1e-9:
            raise ValueError("fusion confidence weights must sum to 1.0")
        if self.single_candidate_denominator <= 0.0:
            raise ValueError("single_candidate_denominator must be positive")


@dataclass(frozen=True, slots=True)
class ExactVote:
    raw_text: str
    count: int
    weight: float


@dataclass(frozen=True, slots=True)
class FusedOCRResult:
    """Compact production result plus deterministic diagnostic fields."""

    track_id: int
    raw_text: str
    confidence: float
    method: str
    support_count: int
    valid_candidate_count: int
    total_weight: float
    winner_weight: float
    consensus_ratio: float
    supporting_frames: tuple[int, ...]
    reference_text: str | None = None
    exact_votes: tuple[ExactVote, ...] = ()
    mean_support_ocr_confidence: float = 0.0
    mean_support_quality: float = 0.0
    alignment_rows: tuple[tuple[int, str, tuple[str, ...]], ...] = ()
    alignment_reference_row: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OCRFusionReport:
    results: tuple[FusedOCRResult, ...]
    tracks_total: int
    tracks_with_ocr: int
    tracks_with_exact_consensus: int
    tracks_using_alignment: int
    single_candidate_tracks: int
    no_valid_ocr_tracks: int
    mean_fusion_confidence: float
    mean_fusion_confidence_all_tracks: float
    exact_vote_ms: float
    alignment_ms: float
    total_ms: float
    config: OCRFusionConfig

    @property
    def exact_vote_ms_per_track(self) -> float:
        return self.exact_vote_ms / self.tracks_total if self.tracks_total else 0.0

    @property
    def alignment_ms_per_alignment_track(self) -> float:
        return (
            self.alignment_ms / self.tracks_using_alignment
            if self.tracks_using_alignment
            else 0.0
        )

    @property
    def total_ms_per_track(self) -> float:
        return self.total_ms / self.tracks_total if self.tracks_total else 0.0


@dataclass(frozen=True, slots=True)
class _AlignedCandidate:
    reference_chars: dict[int, tuple[str, int]]
    insertions: dict[int, tuple[tuple[str, int], ...]]


def candidate_weight(
    candidate: OCRFusionCandidate, include_plate_confidence: bool = False
) -> float:
    """Return a clamped evidence weight without using rank."""

    if candidate.raw_text == "":
        return 0.0
    weight = candidate.ocr_confidence * candidate.quality_score
    if include_plate_confidence and candidate.plate_confidence is not None:
        weight *= candidate.plate_confidence
    return _clamp(weight)


def _candidate_from_object(
    value: OCRFusionCandidate | Mapping[str, Any] | Any,
    track_id: int | None = None,
) -> OCRFusionCandidate:
    if isinstance(value, OCRFusionCandidate):
        return value

    if isinstance(value, Mapping):
        data = value
        ocr = data.get("ocr")
        ocr_data = ocr if isinstance(ocr, Mapping) else {}
        resolved_track = data.get("track_id", track_id)
        raw_text = data.get("raw_text", ocr_data.get("raw_text", ""))
        ocr_confidence = data.get(
            "ocr_confidence", data.get("confidence", ocr_data.get("confidence", 0.0))
        )
        char_confidences = data.get(
            "char_confidences", ocr_data.get("char_confidences")
        )
        status = data.get("status", ocr_data.get("status", "ok"))
        return OCRFusionCandidate(
            track_id=int(resolved_track if resolved_track is not None else 0),
            frame_index=int(data.get("frame_index", 0)),
            rank=int(data.get("rank", 0)),
            raw_text=str(raw_text or ""),
            ocr_confidence=_finite_float(ocr_confidence),
            quality_score=_finite_float(data.get("quality_score", 0.0)),
            char_confidences=_as_char_confidences(char_confidences),
            plate_confidence=(
                _finite_float(data["plate_confidence"])
                if data.get("plate_confidence") is not None
                else None
            ),
            status=str(status),
        )

    resolved_track = getattr(value, "track_id", track_id)
    return OCRFusionCandidate(
        track_id=int(resolved_track if resolved_track is not None else 0),
        frame_index=int(getattr(value, "frame_index", 0)),
        rank=int(getattr(value, "rank", 0)),
        raw_text=str(getattr(value, "raw_text", "") or ""),
        ocr_confidence=_finite_float(
            getattr(value, "ocr_confidence", getattr(value, "confidence", 0.0))
        ),
        quality_score=_finite_float(getattr(value, "quality_score", 0.0)),
        char_confidences=_as_char_confidences(
            getattr(value, "char_confidences", None)
        ),
        plate_confidence=(
            _finite_float(getattr(value, "plate_confidence"))
            if getattr(value, "plate_confidence", None) is not None
            else None
        ),
        status=str(getattr(value, "status", "ok")),
    )


def levenshtein_distance(left: str, right: str) -> int:
    """Return the standard insertion/deletion/substitution distance."""

    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_char in enumerate(left, 1):
        current = [left_index]
        for right_index, right_char in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_char != right_char),
                )
            )
        previous = current
    return previous[-1]


def normalized_levenshtein_distance(left: str, right: str) -> float:
    return levenshtein_distance(left, right) / max(len(left), len(right), 1)


def _exact_votes(
    candidates: Sequence[OCRFusionCandidate], config: OCRFusionConfig
) -> tuple[ExactVote, ...]:
    grouped: dict[str, list[OCRFusionCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.raw_text != "":
            grouped[candidate.raw_text].append(candidate)
    votes = [
        ExactVote(
            raw_text=text,
            count=len(items),
            weight=sum(candidate_weight(item, config.include_plate_confidence) for item in items),
        )
        for text, items in grouped.items()
    ]
    first_frame = {
        text: min(item.frame_index for item in items)
        for text, items in grouped.items()
    }
    return tuple(
        sorted(
            votes,
            key=lambda vote: (
                -vote.weight,
                -vote.count,
                first_frame[vote.raw_text],
                vote.raw_text,
            ),
        )
    )


def _weighted_medoid(
    candidates: Sequence[OCRFusionCandidate],
    votes: Sequence[ExactVote],
    config: OCRFusionConfig,
) -> str:
    texts = tuple(dict.fromkeys(item.raw_text for item in candidates if item.raw_text))
    if not texts:
        return ""
    vote_weights = {vote.raw_text: vote.weight for vote in votes}
    max_ocr = {
        text: max(item.ocr_confidence for item in candidates if item.raw_text == text)
        for text in texts
    }
    max_quality = {
        text: max(item.quality_score for item in candidates if item.raw_text == text)
        for text in texts
    }
    first_frame = {
        text: min(item.frame_index for item in candidates if item.raw_text == text)
        for text in texts
    }

    def medoid_key(text: str) -> tuple[float, float, float, float, int, str]:
        score = sum(
            candidate_weight(item, config.include_plate_confidence)
            * normalized_levenshtein_distance(text, item.raw_text)
            for item in candidates
            if item.raw_text
        )
        return (
            score,
            -vote_weights.get(text, 0.0),
            -max_ocr[text],
            -max_quality[text],
            first_frame[text],
            text,
        )

    return min(texts, key=medoid_key)


def align_pair(reference: str, candidate: str) -> tuple[tuple[str, str], ...]:
    """Needleman-Wunsch-style deterministic global alignment."""

    rows = len(reference) + 1
    columns = len(candidate) + 1
    costs = [[0] * columns for _ in range(rows)]
    parents: list[list[str | None]] = [[None] * columns for _ in range(rows)]
    for row in range(1, rows):
        costs[row][0] = row
        parents[row][0] = "up"
    for column in range(1, columns):
        costs[0][column] = column
        parents[0][column] = "left"

    for row in range(1, rows):
        for column in range(1, columns):
            diagonal = costs[row - 1][column - 1] + (
                reference[row - 1] != candidate[column - 1]
            )
            up = costs[row - 1][column] + 1
            left = costs[row][column - 1] + 1
            best = min(diagonal, up, left)
            costs[row][column] = best
            # Prefer a real character match/substitution, then a reference
            # deletion, then a candidate insertion. This is deterministic and
            # keeps equal strings aligned without inventing a gap.
            if diagonal == best:
                parents[row][column] = "diag"
            elif up == best:
                parents[row][column] = "up"
            else:
                parents[row][column] = "left"

    aligned: list[tuple[str, str]] = []
    row = len(reference)
    column = len(candidate)
    while row or column:
        operation = parents[row][column]
        if operation == "diag":
            aligned.append((reference[row - 1], candidate[column - 1]))
            row -= 1
            column -= 1
        elif operation == "up":
            aligned.append((reference[row - 1], GAP))
            row -= 1
        elif operation == "left":
            aligned.append((GAP, candidate[column - 1]))
            column -= 1
        else:  # pragma: no cover - guarded by the initialized matrix
            raise RuntimeError("alignment backtracking reached an uninitialized cell")
    aligned.reverse()
    return tuple(aligned)


def _align_candidate(reference: str, candidate: OCRFusionCandidate) -> _AlignedCandidate:
    reference_chars: dict[int, tuple[str, int]] = {}
    insertions: dict[int, list[tuple[str, int]]] = defaultdict(list)
    reference_index = 0
    candidate_index = 0
    for reference_token, candidate_token in align_pair(reference, candidate.raw_text):
        if reference_token == GAP:
            insertions[reference_index].append((candidate_token, candidate_index))
            candidate_index += 1
        else:
            if candidate_token == GAP:
                reference_chars[reference_index] = (GAP, -1)
            else:
                reference_chars[reference_index] = (candidate_token, candidate_index)
                candidate_index += 1
            reference_index += 1
    return _AlignedCandidate(
        reference_chars=reference_chars,
        insertions={key: tuple(value) for key, value in insertions.items()},
    )


def _character_weight(candidate: OCRFusionCandidate, character_index: int, config: OCRFusionConfig) -> float:
    weight = candidate_weight(candidate, config.include_plate_confidence)
    if candidate.char_confidences is not None and 0 <= character_index < len(candidate.char_confidences):
        weight *= candidate.char_confidences[character_index]
    return _clamp(weight)


def _choose_character(
    character_weights: Mapping[str, float],
    reference_character: str | None,
) -> str:
    if not character_weights:
        return GAP
    best_weight = max(character_weights.values())
    best = [char for char, weight in character_weights.items() if abs(weight - best_weight) <= 1e-12]
    if reference_character in best:
        return reference_character  # type: ignore[return-value]
    return min(best)


def _alignment_fusion(
    track_id: int,
    candidates: Sequence[OCRFusionCandidate],
    reference: str,
    config: OCRFusionConfig,
) -> tuple[str, int, tuple[int, ...], tuple[tuple[int, str, tuple[str, ...]], ...], tuple[str, ...]]:
    aligned = [_align_candidate(reference, candidate) for candidate in candidates]
    insertion_widths = {
        slot: max(len(item.insertions.get(slot, ())) for item in aligned)
        for slot in range(len(reference) + 1)
    }
    reference_row: list[str] = []
    rows: list[list[str]] = [[] for _ in candidates]
    fused: list[str] = []
    for slot in range(len(reference) + 1):
        for insertion_index in range(insertion_widths[slot]):
            reference_row.append(GAP)
            character_weights: dict[str, float] = defaultdict(float)
            gap_weight = 0.0
            for candidate_index, candidate in enumerate(candidates):
                pair = aligned[candidate_index].insertions.get(slot, ())
                if insertion_index < len(pair):
                    character, index = pair[insertion_index]
                    character_weights[character] += _character_weight(candidate, index, config)
                    rows[candidate_index].append(character)
                else:
                    gap_weight += candidate_weight(candidate, config.include_plate_confidence)
                    rows[candidate_index].append(GAP)
            selected = _choose_character(character_weights, None)
            best_character_weight = max(character_weights.values(), default=0.0)
            if gap_weight > best_character_weight:
                selected = GAP
            if selected != GAP:
                fused.append(selected)
        if slot < len(reference):
            reference_character = reference[slot]
            reference_row.append(reference_character)
            character_weights: dict[str, float] = defaultdict(float)
            gap_weight = 0.0
            for candidate_index, candidate in enumerate(candidates):
                character, index = aligned[candidate_index].reference_chars.get(
                    slot, (GAP, -1)
                )
                if character == GAP:
                    gap_weight += candidate_weight(candidate, config.include_plate_confidence)
                    rows[candidate_index].append(GAP)
                else:
                    character_weights[character] += _character_weight(candidate, index, config)
                    rows[candidate_index].append(character)
            selected = _choose_character(character_weights, reference_character)
            best_character_weight = max(character_weights.values(), default=0.0)
            if gap_weight > best_character_weight:
                selected = GAP
            if selected != GAP:
                fused.append(selected)

    alignment_rows = tuple(
        (candidate.frame_index, candidate.raw_text, tuple(rows[index]))
        for index, candidate in enumerate(candidates)
    )
    support_frames = tuple(candidate.frame_index for candidate in candidates)
    return "".join(fused), len(candidates), support_frames, alignment_rows, tuple(reference_row)


def _mean(values: Iterable[float]) -> float:
    values = tuple(values)
    return sum(values) / len(values) if values else 0.0


def _confidence(
    consensus_ratio: float,
    support_candidates: Sequence[OCRFusionCandidate],
    valid_count: int,
    config: OCRFusionConfig,
) -> tuple[float, float, float]:
    mean_ocr = _mean(item.ocr_confidence for item in support_candidates)
    mean_quality = _mean(item.quality_score for item in support_candidates)
    confidence = (
        config.confidence_consensus_weight * consensus_ratio
        + config.confidence_ocr_weight * mean_ocr
        + config.confidence_quality_weight * mean_quality
    )
    if valid_count == 1:
        evidence_factor = min(valid_count / config.single_candidate_denominator, 1.0)
        confidence *= 0.5 + 0.5 * evidence_factor
    return _clamp(confidence), mean_ocr, mean_quality


def fuse_track(
    track_id: int,
    candidates: Iterable[OCRFusionCandidate | Mapping[str, Any] | Any],
    config: OCRFusionConfig | None = None,
) -> FusedOCRResult:
    """Fuse all V5 candidates belonging to one vehicle track."""

    config = config or OCRFusionConfig()
    normalized = tuple(
        sorted(
            (_candidate_from_object(item, track_id=track_id) for item in candidates),
            key=lambda item: (item.rank, item.frame_index),
        )
    )
    valid = tuple(item for item in normalized if item.raw_text != "")
    valid_count = len(valid)
    if not valid:
        return FusedOCRResult(
            track_id=int(track_id),
            raw_text="",
            confidence=0.0,
            method="no_valid_ocr",
            support_count=0,
            valid_candidate_count=0,
            total_weight=0.0,
            winner_weight=0.0,
            consensus_ratio=0.0,
            supporting_frames=(),
        )

    weights = {
        id(item): candidate_weight(item, config.include_plate_confidence)
        for item in valid
    }
    total_weight = sum(weights.values())
    votes = _exact_votes(valid, config)
    winner = votes[0]
    consensus_ratio = winner.weight / total_weight if total_weight > 0.0 else 0.0
    winner_candidates = tuple(item for item in valid if item.raw_text == winner.raw_text)

    if valid_count == 1:
        confidence, mean_ocr, mean_quality = _confidence(
            1.0, winner_candidates, valid_count, config
        )
        return FusedOCRResult(
            track_id=int(track_id),
            raw_text=winner.raw_text,
            confidence=confidence,
            method="single_candidate",
            support_count=1,
            valid_candidate_count=1,
            total_weight=total_weight,
            winner_weight=winner.weight,
            consensus_ratio=1.0,
            supporting_frames=(winner_candidates[0].frame_index,),
            reference_text=winner.raw_text,
            exact_votes=votes,
            mean_support_ocr_confidence=mean_ocr,
            mean_support_quality=mean_quality,
        )

    if total_weight > 0.0 and consensus_ratio >= config.exact_consensus_threshold:
        confidence, mean_ocr, mean_quality = _confidence(
            consensus_ratio, winner_candidates, valid_count, config
        )
        return FusedOCRResult(
            track_id=int(track_id),
            raw_text=winner.raw_text,
            confidence=confidence,
            method="weighted_exact_vote",
            support_count=winner.count,
            valid_candidate_count=valid_count,
            total_weight=total_weight,
            winner_weight=winner.weight,
            consensus_ratio=consensus_ratio,
            supporting_frames=tuple(item.frame_index for item in winner_candidates),
            reference_text=winner.raw_text,
            exact_votes=votes,
            mean_support_ocr_confidence=mean_ocr,
            mean_support_quality=mean_quality,
        )

    reference = _weighted_medoid(valid, votes, config)
    fused_text, support_count, support_frames, alignment_rows, reference_row = _alignment_fusion(
        int(track_id), valid, reference, config
    )
    confidence, mean_ocr, mean_quality = _confidence(
        consensus_ratio, valid, valid_count, config
    )
    return FusedOCRResult(
        track_id=int(track_id),
        raw_text=fused_text,
        confidence=confidence,
        method="sequence_alignment",
        support_count=support_count,
        valid_candidate_count=valid_count,
        total_weight=total_weight,
        winner_weight=winner.weight,
        consensus_ratio=consensus_ratio,
        supporting_frames=support_frames,
        reference_text=reference,
        exact_votes=votes,
        mean_support_ocr_confidence=mean_ocr,
        mean_support_quality=mean_quality,
        alignment_rows=alignment_rows,
        alignment_reference_row=reference_row,
    )


def fuse_candidates(
    candidates: Iterable[OCRFusionCandidate | Mapping[str, Any] | Any],
    track_ids: Iterable[int] | None = None,
    config: OCRFusionConfig | None = None,
) -> OCRFusionReport:
    """Fuse candidates grouped by track and return V6 metrics/timings."""

    config = config or OCRFusionConfig()
    started = time.perf_counter()
    grouped: dict[int, list[Any]] = defaultdict(list)
    for item in candidates:
        normalized = _candidate_from_object(item)
        grouped[normalized.track_id].append(normalized)
    ids = set(grouped)
    if track_ids is not None:
        ids.update(int(track_id) for track_id in track_ids)

    results: list[FusedOCRResult] = []
    exact_seconds = 0.0
    alignment_seconds = 0.0
    for track_id in sorted(ids):
        track_started = time.perf_counter()
        result = fuse_track(track_id, grouped.get(track_id, ()), config=config)
        track_elapsed = time.perf_counter() - track_started
        if result.method == "sequence_alignment":
            alignment_seconds += track_elapsed
        else:
            exact_seconds += track_elapsed
        results.append(result)
    total_seconds = time.perf_counter() - started
    frozen_results = tuple(results)
    return OCRFusionReport(
        results=frozen_results,
        tracks_total=len(frozen_results),
        tracks_with_ocr=sum(item.valid_candidate_count > 0 for item in frozen_results),
        tracks_with_exact_consensus=sum(
            item.method == "weighted_exact_vote" for item in frozen_results
        ),
        tracks_using_alignment=sum(
            item.method == "sequence_alignment" for item in frozen_results
        ),
        single_candidate_tracks=sum(
            item.method == "single_candidate" for item in frozen_results
        ),
        no_valid_ocr_tracks=sum(
            item.method == "no_valid_ocr" for item in frozen_results
        ),
        mean_fusion_confidence=_mean(
            item.confidence for item in frozen_results if item.valid_candidate_count > 0
        ),
        mean_fusion_confidence_all_tracks=_mean(
            item.confidence for item in frozen_results
        ),
        exact_vote_ms=exact_seconds * 1000.0,
        alignment_ms=alignment_seconds * 1000.0,
        total_ms=total_seconds * 1000.0,
        config=config,
    )


def _result_to_json(result: FusedOCRResult) -> dict[str, object]:
    payload: dict[str, object] = {
        "raw_text": result.raw_text,
        "confidence": round(result.confidence, 6),
        "method": result.method,
        "support_count": result.support_count,
        "valid_candidate_count": result.valid_candidate_count,
        "total_weight": round(result.total_weight, 6),
        "winner_weight": round(result.winner_weight, 6),
        "consensus_ratio": round(result.consensus_ratio, 6),
        "supporting_frames": list(result.supporting_frames),
        "reference_text": result.reference_text,
        "exact_votes": [
            {
                "raw_text": vote.raw_text,
                "count": vote.count,
                "weight": round(vote.weight, 6),
            }
            for vote in result.exact_votes
        ],
    }
    if result.method == "sequence_alignment":
        payload["alignment"] = {
            "reference": list(result.alignment_reference_row),
            "rows": [
                {
                    "frame_index": frame_index,
                    "raw_text": raw_text,
                    "tokens": list(tokens),
                }
                for frame_index, raw_text, tokens in result.alignment_rows
            ],
            "fused_raw_text": result.raw_text,
        }
    return payload


def build_fused_json(
    v5_payload: Mapping[str, Any], report: OCRFusionReport
) -> dict[str, object]:
    """Keep the V5 candidate payload and add one V6 result per track."""

    output = dict(v5_payload)
    source_tracks = {
        int(track.get("track_id", 0)): dict(track)
        for track in v5_payload.get("tracks", ())
        if isinstance(track, Mapping)
    }
    tracks: list[dict[str, object]] = []
    for result in report.results:
        track = dict(source_tracks.get(result.track_id, {"track_id": result.track_id}))
        track.setdefault("candidates", [])
        track["fusion"] = _result_to_json(result)
        tracks.append(track)
    output["tracks"] = tracks
    output["v6_fusion"] = {
        "status": "ok",
        "config": {
            "candidate_weight": "clamp(ocr_confidence * quality_score, 0, 1)",
            "include_plate_confidence": report.config.include_plate_confidence,
            "exact_consensus_threshold": report.config.exact_consensus_threshold,
            "sequence_alignment": "Needleman-Wunsch global alignment",
            "gap_token": GAP,
            "fusion_confidence": "0.50*consensus_ratio + 0.25*mean_support_ocr_confidence + 0.25*mean_support_quality",
            "single_candidate_evidence_factor": "min(valid_candidate_count / 3.0, 1.0); confidence *= 0.5 + 0.5*factor",
            "vietnam_postprocessing": False,
            "uppercase": False,
        },
        "summary": {
            "tracks_total": report.tracks_total,
            "tracks_with_ocr": report.tracks_with_ocr,
            "tracks_with_exact_consensus": report.tracks_with_exact_consensus,
            "tracks_using_alignment": report.tracks_using_alignment,
            "single_candidate_tracks": report.single_candidate_tracks,
            "no_valid_ocr_tracks": report.no_valid_ocr_tracks,
            "mean_fusion_confidence": round(report.mean_fusion_confidence, 6),
            "mean_fusion_confidence_all_tracks": round(
                report.mean_fusion_confidence_all_tracks, 6
            ),
        },
        "performance": {
            "exact_vote_ms_per_track": round(report.exact_vote_ms_per_track, 6),
            "alignment_ms_per_alignment_track": round(
                report.alignment_ms_per_alignment_track, 6
            ),
            "total_fusion_ms_video": round(report.total_ms, 6),
            "total_fusion_ms_per_track": round(report.total_ms_per_track, 6),
        },
    }
    return output


def write_fused_json(
    output_path: str | Path, v5_payload: Mapping[str, Any], report: OCRFusionReport
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_fused_json(v5_payload, report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def format_fusion_debug(
    report: OCRFusionReport,
    candidates: Iterable[OCRFusionCandidate | Mapping[str, Any] | Any] | None = None,
) -> str:
    lines: list[str] = []
    candidates_by_track: dict[int, list[OCRFusionCandidate]] = defaultdict(list)
    if candidates is not None:
        for item in candidates:
            normalized = _candidate_from_object(item)
            candidates_by_track[normalized.track_id].append(normalized)
    for result in report.results:
        lines.append(f"Track {result.track_id}")
        lines.append("  Candidates:")
        for candidate in sorted(
            candidates_by_track.get(result.track_id, ()),
            key=lambda item: (item.rank, item.frame_index),
        ):
            lines.append(
                f"    frame {candidate.frame_index} rank={candidate.rank} "
                f"{candidate.raw_text!r} ocr={candidate.ocr_confidence:.6f} "
                f"q={candidate.quality_score:.6f} "
                f"weight={candidate_weight(candidate, report.config.include_plate_confidence):.6f}"
            )
        for vote in result.exact_votes:
            lines.append(
                f"    exact {vote.raw_text!r}: count={vote.count} weight={vote.weight:.6f}"
            )
        lines.append(
            f"  Selected: {result.raw_text!r} method={result.method} "
            f"confidence={result.confidence:.6f} support={result.support_count}/"
            f"{result.valid_candidate_count} reference={result.reference_text!r}"
        )
        if result.alignment_rows:
            lines.append("  Alignment:")
            lines.append(f"    reference: {' '.join(result.alignment_reference_row)}")
            for frame_index, raw_text, tokens in result.alignment_rows:
                lines.append(
                    f"    frame {frame_index}: {raw_text} -> {' '.join(tokens)}"
                )
    lines.append(
        f"Fusion timing: total={report.total_ms:.3f} ms, "
        f"exact/track={report.exact_vote_ms_per_track:.3f} ms, "
        f"alignment/track={report.alignment_ms_per_alignment_track:.3f} ms"
    )
    return "\n".join(lines)


__all__ = [
    "GAP",
    "ExactVote",
    "FusedOCRResult",
    "OCRFusionCandidate",
    "OCRFusionConfig",
    "OCRFusionReport",
    "align_pair",
    "build_fused_json",
    "candidate_weight",
    "format_fusion_debug",
    "fuse_candidates",
    "fuse_track",
    "levenshtein_distance",
    "normalized_levenshtein_distance",
    "write_fused_json",
]
