"""Video ALPR execution using shared detection and OCR stages."""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .ocr_fusion import FusedOCRResult, fuse_candidates
from .ocr_stage import OCRPlateCandidate, run_ocr_on_topk
from .pipeline_support import _debug_ocr, _draw_box, _ms, _write_json
from .plate_buffer import PlateBufferManager
from .plate_ownership_temporal import TemporalPlateOwnershipResolver
from .plate_quality import crop_plate_from_frame, score_plate_quality
from .plate_stage import buffered_plate_candidate, detect_tracked_plates
from .result_finalizer import FinalResultCollector, finalize_vehicle
from .result_serialization import serialize_vehicle_result
from .vehicle_stage import create_video_tracker
from .vn_plate_postprocessor import postprocess_vietnam_plate


def run_video(
    self,
    path: str | Path,
    *,
    output: str | Path | None = None,
    save_annotated: bool = False,
    save_topk_crops: bool = False,
    debug: bool | None = None,
) -> dict[str, Any]:
    source = Path(path)
    detailed = self.debug if debug is None else debug
    output_path = self._output_path(source, output)
    artifact_stem = self._artifact_stem(source, output_path, output)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not np.isfinite(fps) or fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video metadata: {source}")
    tracker = create_video_tracker(
        fps, self.config.vehicle.confidence, self.config.tracking, detailed,
    )
    resolver = TemporalPlateOwnershipResolver(self.config.ownership)
    manager = PlateBufferManager(self.config.buffer)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotated_path = output_path.parent / f"{artifact_stem}_annotated.mp4"
    writer = None
    if save_annotated:
        writer = cv2.VideoWriter(str(annotated_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            capture.release()
            raise OSError(f"Could not create annotated video: {annotated_path}")

    frame_count = 0
    vehicle_seconds = tracking_seconds = ownership_seconds = quality_seconds = buffer_seconds = 0.0
    quality_calls = buffer_calls = 0
    vehicle_inference_before = getattr(self.vehicle_detector, "total_inference_seconds", 0.0)
    plate_calls_before = self.plate_detector.detect_call_count
    plate_seconds_before = self.plate_detector.total_processing_seconds
    plate_inference_before = getattr(self.plate_detector, "total_inference_seconds", 0.0)
    ocr_calls_before = self.ocr_engine.inference_count
    ocr_ms_before = self.ocr_engine.timing_totals["total_ms_per_crop"] * ocr_calls_before
    overall_started = time.perf_counter()
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frame_index = frame_count
            frame_count += 1
            stage = time.perf_counter()
            detections = self.vehicle_detector.detect(frame)
            vehicle_seconds += time.perf_counter() - stage
            stage = time.perf_counter()
            tracks = tracker.update(detections, frame_index)
            tracking_seconds += time.perf_counter() - stage
            for track in tracks:
                manager.mark_vehicle_seen(track.track_id, frame_index)
            raw_plates = detect_tracked_plates(frame, tracks, self.plate_detector, frame_index)
            stage = time.perf_counter()
            resolution = resolver.resolve(frame_index, raw_plates)
            ownership_seconds += time.perf_counter() - stage
            for plate in resolution.candidates:
                crop = crop_plate_from_frame(frame, plate.plate_bbox)
                if crop is None:
                    stage = time.perf_counter()
                    manager.record_invalid_plate(plate.track_id, frame_index, plate.plate_class_name)
                    buffer_seconds += time.perf_counter() - stage
                    buffer_calls += 1
                    continue
                stage = time.perf_counter()
                quality = score_plate_quality(crop, plate.plate_confidence, self.config.quality)
                quality_seconds += time.perf_counter() - stage
                quality_calls += 1
                candidate = buffered_plate_candidate(plate, crop, quality)
                stage = time.perf_counter()
                manager.add_plate(candidate)
                buffer_seconds += time.perf_counter() - stage
                buffer_calls += 1
            resolver.cleanup(
                track.track_id for track in tracker.all_tracks if track.state.name != "REMOVED"
            )
            if writer is not None:
                annotated = frame.copy()
                for track in tracks:
                    _draw_box(annotated, track.bbox, f"ID {track.track_id} {track.class_name}", (0, 200, 0))
                for plate in resolution.candidates:
                    _draw_box(annotated, plate.plate_bbox, plate.plate_class_name, (0, 0, 255))
                writer.write(annotated)
            if frame_count == 1 or frame_count % 30 == 0:
                print(f"Processed {frame_count} video frames", flush=True)
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    if frame_count == 0:
        raise ValueError(f"Video contains no decodable frames: {source}")

    summaries = manager.finalize_all()
    if save_topk_crops:
        for summary in summaries:
            for rank, candidate in enumerate(manager.get_top_candidates(summary.track_id), 1):
                crop_path = output_path.parent / f"{artifact_stem}_topk" / f"track_{summary.track_id}" / f"rank_{rank}_frame_{candidate.frame_index}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(crop_path), candidate.crop):
                    raise OSError(f"Could not write crop: {crop_path}")
    ocr_results = run_ocr_on_topk(manager, self.ocr_engine, debug=detailed)
    grouped_ocr: dict[int, list[OCRPlateCandidate]] = defaultdict(list)
    for candidate in ocr_results:
        grouped_ocr[candidate.track_id].append(candidate)
    fusion = fuse_candidates(
        ocr_results,
        track_ids=(track.track_id for track in tracker.all_tracks),
        config=self.config.fusion,
    )
    fused_by_track: dict[int, FusedOCRResult] = {result.track_id: result for result in fusion.results}
    summary_by_track = {item.track_id: item for item in summaries}
    postprocess_seconds = 0.0
    processed_by_track = {}
    final_results = FinalResultCollector()
    for track in tracker.all_tracks:
        summary = summary_by_track[track.track_id]
        fused = fused_by_track[track.track_id]
        stage = time.perf_counter()
        processed = postprocess_vietnam_plate(fused.raw_text, fused.confidence, self.config.vietnam)
        postprocess_seconds += time.perf_counter() - stage
        processed_by_track[track.track_id] = processed
        best = manager.get_top_candidates(track.track_id)
        final_results.add(finalize_vehicle(
            identity_key="track_id", identity=track.track_id,
            vehicle_class_id=track.class_id, vehicle_class_name=track.class_name,
            first_frame=track.first_frame, last_frame=track.last_frame,
            vehicle_observation_count=summary.vehicle_observation_count,
            plate_observation_count=summary.plate_observation_count,
            plate_layout=summary.dominant_plate_class,
            best_plate_bbox=best[0].bbox if best else None,
            best_plate_frame=best[0].frame_index if best else None,
            postprocessed=processed, fusion=fused,
            ocr_candidate_count=len(grouped_ocr.get(track.track_id, ())),
            low_confidence_threshold=self.config.low_confidence_threshold,
        ))
    rows = [serialize_vehicle_result(result) for result in final_results.results]
    if detailed:
        for row in rows:
            track_id = row["track_id"]
            fused = fused_by_track[track_id]
            processed = processed_by_track[track_id]
            row["debug"] = {
                "ocr_candidates": [_debug_ocr(item) for item in grouped_ocr.get(track_id, ())],
                "quality": [
                    {"rank": rank, "frame_index": candidate.frame_index,
                     "score": round(candidate.quality.total_score, 6)}
                    for rank, candidate in enumerate(manager.get_top_candidates(track_id), 1)
                ],
                "fusion": {
                    "reference_text": fused.reference_text,
                    "consensus_ratio": round(fused.consensus_ratio, 6),
                    "supporting_frames": list(fused.supporting_frames),
                },
                "corrections": [
                    {"index": c.index, "from": c.from_char, "to": c.to_char, "reason": c.reason}
                    for c in processed.corrections
                ],
                "unknown_characters": list(processed.unknown_characters),
            }
    processed_track_count = len(rows)
    # Keep only results that meet the configured minimum OCR/fusion confidence.
    # Format validation remains available in the JSON but does not filter rows.
    rows = [
        row for row in rows
        if row["plate"]["confidence"] >= self.config.low_confidence_threshold
    ]
    plate_calls = self.plate_detector.detect_call_count - plate_calls_before
    ocr_calls = self.ocr_engine.inference_count - ocr_calls_before
    ocr_ms = self.ocr_engine.timing_totals["total_ms_per_crop"] * self.ocr_engine.inference_count - ocr_ms_before
    elapsed = time.perf_counter() - overall_started
    payload: dict[str, Any] = {
        "status": "ok",
        "input": {"path": str(source), "type": "video", "frames": frame_count,
                  "fps": round(fps, 6), "width": width, "height": height},
        "summary": {
            "vehicles": len(rows),
            "vehicles_with_plate": sum(row["plate"]["plate_observations"] > 0 for row in rows),
            "vehicles_with_ocr": sum(bool(row["plate"]["raw_text"]) for row in rows),
            "successful_results": sum(row["plate"]["status"] == "ok" for row in rows),
        },
        "vehicles": rows,
        "performance": {
            "vehicle_detection_ms_per_frame": _ms(vehicle_seconds, frame_count),
            "vehicle_inference_ms_per_frame": _ms(getattr(self.vehicle_detector, "total_inference_seconds", 0.0) - vehicle_inference_before, frame_count),
            "tracking_ms_per_frame": _ms(tracking_seconds, frame_count),
            "plate_detection_ms_per_roi": _ms(self.plate_detector.total_processing_seconds - plate_seconds_before, plate_calls),
            "plate_inference_ms_per_roi": _ms(getattr(self.plate_detector, "total_inference_seconds", 0.0) - plate_inference_before, plate_calls),
            "ownership_ms_per_frame": _ms(ownership_seconds, frame_count),
            "v4_quality_ms_per_plate": _ms(quality_seconds, quality_calls),
            "v4_buffer_ms_per_plate": _ms(buffer_seconds, buffer_calls),
            "ocr_ms_per_crop": round(ocr_ms / ocr_calls, 6) if ocr_calls else 0.0,
            "ocr_inference_calls": ocr_calls,
            "fusion_ms_per_track": round(fusion.total_ms_per_track, 6),
            "postprocess_ms_per_track": _ms(postprocess_seconds, processed_track_count),
            "overall_fps": round(frame_count / elapsed, 6),
            "elapsed_seconds": round(elapsed, 6),
            "session_init_count": self.session_init_count,
        },
    }
    if writer is not None:
        payload["annotated_path"] = str(annotated_path)
    _write_json(output_path, payload)
    payload["output_path"] = str(output_path)
    return payload
