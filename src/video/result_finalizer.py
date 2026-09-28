"""Finalize every non-duplicate video track without discarding failures."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from ..config import PipelineConfig
from ..core.ocr_fusion import OCRFusionReport
from ..core.status import missing_evidence_status
from .plate_buffer import VideoPlateBuffer
from .postprocess_stage import postprocess_video_fused


def finalize_video_tracks(
    history: Mapping[int, dict[str, object]],
    class_votes: Mapping[int, Counter[str]],
    plate_observations: Mapping[int, int],
    buffer: VideoPlateBuffer,
    fusion: OCRFusionReport,
    config: PipelineConfig,
) -> list[dict[str, object]]:
    fused_by_track = {item.track_id: item for item in fusion.results}
    rows: list[dict[str, object]] = []
    for track_id in sorted(history):
        timeline = history[track_id]
        if timeline.get("removal_reason") == "duplicate":
            continue
        votes = class_votes.get(track_id, Counter())
        if not votes:
            raise ValueError(f"No class observation for track {track_id}")
        class_name = min(votes, key=lambda name: (-votes[name], name))
        fused = fused_by_track[track_id]
        raw_text = fused.raw_text if fused.valid_candidate_count else None
        post = postprocess_video_fused(
            raw_text, fused.confidence, class_name, config=config.vietnam,
        )
        normalized = post.postprocessed
        observation_count = int(plate_observations.get(track_id, 0))
        missing = missing_evidence_status(observation_count, post.status != "no_ocr")
        if missing is not None:
            status, reason = missing, None if missing == "no_plate" else post.reason
        elif post.status in {"low_format_confidence", "vehicle_plate_family_mismatch"}:
            status, reason = "low_confidence", post.reason
        elif not post.valid:
            status, reason = "unrecognized_format", post.reason
        elif fused.confidence < config.low_confidence_threshold:
            status, reason = "low_confidence", "weak_ocr_evidence"
        else:
            status, reason = "ok", None
        selected = buffer.selected(track_id)
        best = selected[0] if selected else None
        rows.append({
            "track_id": track_id,
            "vehicle": {
                "class_name": class_name,
                "class_counts": dict(sorted(votes.items())),
                "first_frame": timeline["first_frame"],
                "last_frame": timeline["last_frame"],
                "observations": timeline["hits"],
                "removal_reason": timeline.get("removal_reason"),
            },
            "plate": {
                "status": status,
                "status_reason": reason,
                "postprocess_status": post.status,
                "raw_text": raw_text,
                "normalized_text": normalized.normalized_text if normalized else None,
                "text": normalized.corrected_text if normalized else None,
                "formatted": post.formatted_text,
                "format_family": normalized.format_family if normalized else None,
                "format_valid": post.valid,
                "format_reason": post.reason,
                "confidence": round(fused.confidence, 6),
                "plate_observations": observation_count,
                "best_bbox_xyxy": list(best.candidate.plate_bbox) if best else None,
                "best_frame": best.candidate.frame_index if best else None,
            },
            "evidence": {
                "ocr_candidates": fused.valid_candidate_count,
                "fusion_method": fused.method,
                "support_count": fused.support_count,
                "topk_selected": len(selected),
            },
        })
    return rows


__all__ = ["finalize_video_tracks"]
