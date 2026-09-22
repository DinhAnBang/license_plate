"""Diagnostic OCR and fusion JSON, separate from inference and algorithms."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .microcharnet_ocr import MicroCharNetOCR
from .ocr_stage import OCRPlateCandidate
from .ocr_fusion import (
    GAP, FusedOCRResult, OCRFusionCandidate, OCRFusionReport,
    _candidate_from_object, candidate_weight,
)

def _candidate_to_json(candidate: OCRPlateCandidate) -> dict[str, object]:
    ocr_payload: dict[str, object] = {
        "raw_text": candidate.raw_text,
        "confidence": round(candidate.ocr_confidence, 6),
        "char_confidences": (
            [round(value, 6) for value in candidate.char_confidences]
            if candidate.char_confidences is not None
            else None
        ),
        "status": candidate.status,
    }
    if candidate.error:
        ocr_payload["error"] = candidate.error
    return {
        "rank": candidate.rank,
        "frame_index": candidate.frame_index,
        "plate_class_id": candidate.plate_class_id,
        "plate_class_name": candidate.plate_class_name,
        "plate_confidence": round(candidate.plate_confidence, 6),
        "quality_score": round(candidate.quality_score, 6),
        "bbox_xyxy": list(candidate.bbox),
        "ocr": ocr_payload,
    }


def build_ocr_json(
    candidates: Iterable[OCRPlateCandidate], ocr: MicroCharNetOCR
) -> dict[str, object]:
    """Build the V5 JSON payload; no image arrays and no winner selection."""

    candidates = tuple(candidates)
    grouped: dict[int, list[OCRPlateCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.track_id, []).append(candidate)
    tracks = [
        {
            "track_id": track_id,
            "candidates": [_candidate_to_json(item) for item in grouped[track_id]],
        }
        for track_id in sorted(grouped)
    ]
    timing = dict(ocr.timing_totals)
    timing.update(
        {
            "total_retained_candidates": len(candidates),
            "successful_or_empty_results": sum(
                item.status in {"ok", "empty"} for item in candidates
            ),
        }
    )
    return {
        "status": "ok",
        "ocr_model": ocr.model_info,
        "summary": {
            "retained_topk_candidates": len(candidates),
            "ocr_inference_calls": int(ocr.inference_count),
            "session_init_count": int(ocr.session_init_count),
            "failed_candidates": sum(item.status == "decode_failed" for item in candidates),
            "temporal_vote": False,
            "vietnam_postprocessing": False,
        },
        "performance": timing,
        "tracks": tracks,
    }


def write_ocr_json(
    output_path: str | Path,
    candidates: Iterable[OCRPlateCandidate],
    ocr: MicroCharNetOCR,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_ocr_json(candidates, ocr), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


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


__all__ = ["build_ocr_json", "write_ocr_json", "build_fused_json", "write_fused_json", "format_fusion_debug"]
