"""Production two-pass video ALPR orchestration."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2

from ..core.ocr_fusion import fuse_candidates
from .ocr_stage import recognize_video_topk
from .plate_buffer import VideoBufferedPlate, VideoPlateBuffer
from ..core.plate_quality import crop_plate_from_frame, score_plate_quality
from .plate_stage import detect_frame_plate_candidates
from .plate_ownership import resolve_frame_plate_ownership
from .renderer import write_annotated_video
from .result_finalizer import finalize_video_tracks
from .tracking_runtime import TrackedVideoObservation, iter_tracked_video_frames


def run_video(
    pipeline: Any,
    path: str | Path,
    *,
    output: str | Path | None = None,
    save_annotated: bool = True,
    save_topk_crops: bool = False,
    debug: bool | None = None,
) -> dict[str, Any]:
    """Run the existing stages once, then render with stored per-frame boxes."""
    source = Path(path)
    output_path = pipeline._output_path(source, output)
    artifact_stem = pipeline._artifact_stem(source, output_path, output)
    detailed = pipeline.debug if debug is None else debug
    buffer = VideoPlateBuffer(pipeline.config.video.topk)
    frames: list[dict[str, object]] = []
    class_votes: dict[int, Counter[str]] = defaultdict(Counter)
    plate_observations: Counter[int] = Counter()
    last: TrackedVideoObservation | None = None
    raw_vehicle_detections = 0
    selected_plates = 0
    raw_plate_candidates = 0

    for observation in iter_tracked_video_frames(
        str(source), detector=pipeline.vehicle_detector,
        validation_config=pipeline.config.video.validation,
        tracking_config=pipeline.config.video.tracking,
    ):
        last = observation
        raw_vehicle_detections += len(observation.validated)
        vehicles: list[dict[str, object]] = []
        for tracked in observation.tracked:
            class_votes[tracked.track_id][tracked.detection.class_name] += 1
            vehicles.append({
                "track_id": tracked.track_id,
                "class_name": tracked.detection.class_name,
                "confidence": round(float(tracked.detection.confidence), 6),
                "bbox_xyxy": list(tracked.detection.bbox),
            })
        decisions = resolve_frame_plate_ownership(
            detect_frame_plate_candidates(observation, pipeline.plate_detector),
            conflict_iou_threshold=pipeline.config.video.ownership.conflict_iou_threshold,
            overlap_over_smaller_threshold=pipeline.config.video.ownership.overlap_over_smaller_threshold,
        )
        raw_plate_candidates += len(decisions)
        plates: list[dict[str, object]] = []
        for decision in decisions:
            if decision.ownership_status != "selected":
                continue
            candidate = decision.candidate
            selected_plates += 1
            plate_observations[candidate.track_id] += 1
            plates.append({
                "track_id": candidate.track_id,
                "bbox_xyxy": list(candidate.plate_bbox),
                "candidate_index": candidate.candidate_index,
                "class_name": candidate.plate_class_name,
                "confidence": round(candidate.plate_confidence, 6),
            })
            crop = crop_plate_from_frame(observation.image, candidate.plate_bbox)
            if crop is None:
                continue
            quality = score_plate_quality(
                crop, candidate.plate_confidence, pipeline.config.video.quality,
            )
            buffer.add(VideoBufferedPlate(candidate, quality, crop))
        frames.append({
            "frame_index": observation.frame_index,
            "vehicles": vehicles,
            "plates": plates,
        })

    if last is None:
        raise ValueError(f"Video contains no decodable frames: {source}")
    history = last.track_history
    final_track_ids = tuple(sorted(
        track_id for track_id, item in history.items()
        if item.get("removal_reason") != "duplicate"
    ))
    if save_topk_crops:
        for track_id in final_track_ids:
            for rank, item in enumerate(buffer.selected(track_id), start=1):
                crop_path = (output_path.parent / "crops" / f"track_{track_id}"
                             / f"top{rank}_frame{item.candidate.frame_index:06d}.jpg")
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(crop_path), item.crop):
                    raise OSError(f"Could not write crop: {crop_path}")
    ocr_candidates = recognize_video_topk(buffer, pipeline.ocr_engine, final_track_ids)
    fusion = fuse_candidates(
        ocr_candidates, track_ids=final_track_ids, config=pipeline.config.fusion,
    )
    rows = finalize_video_tracks(
        history, class_votes, plate_observations, buffer, fusion, pipeline.config,
    )
    final_by_track = {int(item["track_id"]): item for item in rows}
    payload: dict[str, Any] = {
        "status": "ok",
        "input": {
            "path": str(source), "type": "video", "frames": len(frames),
            "fps": round(last.fps, 6), "width": last.width, "height": last.height,
        },
        "summary": {
            "frames_read": len(frames),
            "vehicles": len(rows),
            "vehicles_with_plate": sum(item["plate"]["plate_observations"] > 0 for item in rows),
            "vehicles_with_ocr": sum(bool(item["plate"]["raw_text"]) for item in rows),
            "successful_results": sum(item["plate"]["status"] == "ok" for item in rows),
            "duplicate_tracks_excluded": len(history) - len(rows),
            "raw_vehicle_detections": raw_vehicle_detections,
            "raw_plate_candidates": raw_plate_candidates,
            "ownership_selected_plates": selected_plates,
            "ocr_crops": len(ocr_candidates),
        },
        "vehicles": rows,
        "frames": frames,
    }
    if save_annotated:
        annotated_path = output_path.parent / f"{artifact_stem}_annotated.mp4"
        payload["summary"]["frames_rendered"] = write_annotated_video(
            source, annotated_path, frames, final_by_track,
            mode="debug" if detailed else "customer",
        )
        payload["annotated_path"] = str(annotated_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    payload["output_path"] = str(output_path)
    return payload


__all__ = ["run_video"]
