"""Production image/video ALPR pipeline using the three ONNX models."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

from .config import PipelineConfig
from .microcharnet_ocr import MicroCharNetOCR, OCRPlateCandidate, run_ocr_on_image_candidates, run_ocr_on_topk
from .ocr_fusion import FusedOCRResult, fuse_candidates
from .plate_buffer import BufferedPlateCandidate, PlateBufferManager
from .plate_detector import (
    PlateDetector, TrackedPlateCandidate, crop_vehicle_roi,
    detect_tracked_plates, local_bbox_to_global, select_best_plate,
)
from .plate_ownership_temporal import TemporalPlateOwnershipResolver
from .plate_quality import crop_plate_from_frame, score_plate_quality
from .result_finalizer import FinalResultCollector, finalize_vehicle
from .tracking import ByteTracker
from .vehicle_detector import VehicleDetector
from .vn_plate_postprocessor import postprocess_vietnam_plate


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})


def _ms(seconds: float, count: int = 1) -> float:
    return round(seconds * 1000.0 / count, 6) if count else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _draw_box(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (max(0, x1), max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _debug_ocr(candidate: OCRPlateCandidate) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "frame_index": candidate.frame_index,
        "raw_text": candidate.raw_text,
        "ocr_confidence": round(candidate.ocr_confidence, 6),
        "quality_score": round(candidate.quality_score, 6),
        "char_confidences": list(candidate.char_confidences) if candidate.char_confidences else None,
        "status": candidate.status,
        "error": candidate.error,
    }


class ALPRPipeline:
    """Load detector/OCR sessions once, then process independent inputs."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        device: str = "auto",
        debug: bool = False,
        vehicle_detector: VehicleDetector | None = None,
        plate_detector: PlateDetector | None = None,
        ocr_engine: MicroCharNetOCR | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.debug = debug
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available")
        providers = (["CPUExecutionProvider"] if device == "cpu" else
                     ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else None)
        for model in (self.config.vehicle.model, self.config.plate.model, self.config.ocr.model):
            if Path(model).suffix.lower() != ".onnx":
                raise ValueError(f"Production model must be ONNX: {model}")

        def load_detector(factory: Any, *args: Any) -> Any:
            try:
                return factory(*args, debug=debug, providers=providers)
            except Exception:
                if device != "auto" or "CUDAExecutionProvider" not in available:
                    raise
                return factory(*args, debug=debug, providers=["CPUExecutionProvider"])

        self.vehicle_detector = vehicle_detector or load_detector(
            VehicleDetector, self.config.vehicle.model,
            self.config.vehicle.confidence, self.config.vehicle.iou,
        )
        self.plate_detector = plate_detector or load_detector(
            PlateDetector, self.config.plate.model,
            self.config.plate.confidence, self.config.plate.iou,
        )
        self.ocr_engine = ocr_engine or MicroCharNetOCR(
            self.config.ocr.model, conf_threshold=self.config.ocr.confidence,
            iou_threshold=self.config.ocr.iou, providers=providers,
        )
        if device == "cuda" and not (vehicle_detector or plate_detector or ocr_engine):
            selected = (
                self.vehicle_detector.session.get_providers()[0],
                self.plate_detector.session.get_providers()[0],
                self.ocr_engine.provider,
            )
            if any(provider != "CUDAExecutionProvider" for provider in selected):
                raise RuntimeError(f"CUDA execution was requested but providers are {selected}")

    @property
    def models(self) -> dict[str, str]:
        return {
            "vehicle": str(self.config.vehicle.model),
            "plate": str(self.config.plate.model),
            "ocr": str(self.config.ocr.model),
        }

    @property
    def session_init_count(self) -> dict[str, int]:
        return {
            "vehicle": getattr(self.vehicle_detector, "session_init_count", 1),
            "plate": getattr(self.plate_detector, "session_init_count", 1),
            "ocr": self.ocr_engine.session_init_count,
        }

    def _output_path(self, source: Path, output: str | Path | None) -> Path:
        if output is not None:
            return Path(output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("output") / f"{source.stem}_{timestamp}.json"

    @staticmethod
    def _artifact_stem(source: Path, output_path: Path, output: str | Path | None) -> str:
        """Use one shared prefix for default JSON and media artifacts."""

        return output_path.stem if output is None else source.stem

    def process_image(
        self,
        path: str | Path,
        *,
        output: str | Path | None = None,
        save_annotated: bool = False,
        save_topk_crops: bool = False,
        debug: bool | None = None,
    ) -> dict[str, Any]:
        source = Path(path)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Could not decode image: {source}")
        detailed = self.debug if debug is None else debug
        output_path = self._output_path(source, output)
        artifact_stem = self._artifact_stem(source, output_path, output)
        frame_height, frame_width = image.shape[:2]
        overall_started = time.perf_counter()
        vehicle_inference_before = getattr(self.vehicle_detector, "total_inference_seconds", 0.0)
        plate_calls_before = self.plate_detector.detect_call_count
        plate_seconds_before = self.plate_detector.total_processing_seconds
        plate_inference_before = getattr(self.plate_detector, "total_inference_seconds", 0.0)
        ocr_calls_before = self.ocr_engine.inference_count
        ocr_ms_before = self.ocr_engine.timing_totals["total_ms_per_crop"] * ocr_calls_before
        stage = time.perf_counter()
        vehicles = [
            vehicle for vehicle in self.vehicle_detector.detect(image)
            if vehicle.confidence >= self.config.vehicle.image_confidence
        ]
        vehicle_seconds = time.perf_counter() - stage
        annotated = image.copy() if save_annotated else None
        raw: list[TrackedPlateCandidate] = []
        for vehicle_index, vehicle in enumerate(vehicles):
            if annotated is not None:
                _draw_box(annotated, vehicle.bbox, f"{vehicle_index} {vehicle.class_name}", (0, 200, 0))
            cropped = crop_vehicle_roi(image, vehicle.bbox)
            if cropped is None:
                continue
            roi, vehicle_bbox = cropped
            best = select_best_plate(self.plate_detector.detect(roi))
            if best is None:
                continue
            global_bbox = local_bbox_to_global(best.bbox, vehicle_bbox, frame_width, frame_height)
            if global_bbox[2] <= global_bbox[0] or global_bbox[3] <= global_bbox[1]:
                continue
            raw.append(TrackedPlateCandidate(
                frame_index=0, track_id=vehicle_index,
                vehicle_class_id=vehicle.class_id, vehicle_class_name=vehicle.class_name,
                vehicle_confidence=vehicle.confidence, vehicle_bbox=vehicle_bbox,
                plate_class_id=best.class_id, plate_class_name=best.class_name,
                plate_confidence=best.confidence, plate_bbox=global_bbox,
            ))
        owner = TemporalPlateOwnershipResolver(self.config.ownership)
        stage = time.perf_counter()
        resolution = owner.resolve(0, raw)
        ownership_seconds = time.perf_counter() - stage
        selected = {candidate.track_id: candidate for candidate in resolution.candidates}
        image_candidates: list[BufferedPlateCandidate] = []
        quality_seconds = 0.0
        for plate in resolution.candidates:
            crop = crop_plate_from_frame(image, plate.plate_bbox)
            if crop is None:
                continue
            stage = time.perf_counter()
            quality = score_plate_quality(crop, plate.plate_confidence, self.config.quality)
            quality_seconds += time.perf_counter() - stage
            image_candidates.append(BufferedPlateCandidate(
                frame_index=0, track_id=plate.track_id, plate_class_id=plate.plate_class_id,
                plate_class_name=plate.plate_class_name, plate_confidence=plate.plate_confidence,
                bbox=plate.plate_bbox, quality=quality, crop=crop,
            ))
            if annotated is not None:
                _draw_box(annotated, plate.plate_bbox, plate.plate_class_name, (0, 0, 255))
            if save_topk_crops:
                crop_path = output_path.parent / f"{artifact_stem}_crops" / f"vehicle_{plate.track_id}.jpg"
                crop_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(crop_path), crop):
                    raise OSError(f"Could not write crop: {crop_path}")
        ocr_results = run_ocr_on_image_candidates(image_candidates, self.ocr_engine, debug=detailed)
        ocr_by_index = {candidate.track_id: candidate for candidate in ocr_results}
        final_rows: list[dict[str, Any]] = []
        postprocess_seconds = 0.0
        for index, vehicle in enumerate(vehicles):
            plate = selected.get(index)
            ocr = ocr_by_index.get(index)
            raw_text = ocr.raw_text if ocr else ""
            confidence = ocr.ocr_confidence if ocr else 0.0
            stage = time.perf_counter()
            normalized = postprocess_vietnam_plate(raw_text, confidence, self.config.vietnam)
            postprocess_seconds += time.perf_counter() - stage
            final = finalize_vehicle(
                identity_key="vehicle_index", identity=index,
                vehicle_class_id=vehicle.class_id, vehicle_class_name=vehicle.class_name,
                first_frame=None, last_frame=None, vehicle_observation_count=1,
                plate_observation_count=int(plate is not None),
                plate_layout=plate.plate_class_name if plate else None,
                best_plate_bbox=plate.plate_bbox if plate else None,
                best_plate_frame=None, postprocessed=normalized, fusion=None,
                ocr_candidate_count=int(ocr is not None),
                low_confidence_threshold=self.config.low_confidence_threshold,
            )
            row = final.to_json()
            row["vehicle"]["bbox_xyxy"] = list(vehicle.bbox)
            row["vehicle"]["confidence"] = round(vehicle.confidence, 6)
            row["evidence"]["fusion_method"] = "single_image" if ocr else None
            if detailed:
                row["debug"] = {
                    "ocr": _debug_ocr(ocr) if ocr else None,
                    "corrections": [
                        {"index": c.index, "from": c.from_char, "to": c.to_char, "reason": c.reason}
                        for c in normalized.corrections
                    ],
                    "unknown_characters": list(normalized.unknown_characters),
                }
            final_rows.append(row)
        # Keep only results that meet the configured minimum OCR/fusion confidence.
        # Format validation remains available in the JSON but does not filter rows.
        final_rows = [
            row for row in final_rows
            if row["plate"]["confidence"] >= self.config.low_confidence_threshold
        ]
        plate_calls = self.plate_detector.detect_call_count - plate_calls_before
        ocr_calls = self.ocr_engine.inference_count - ocr_calls_before
        ocr_ms = self.ocr_engine.timing_totals["total_ms_per_crop"] * self.ocr_engine.inference_count - ocr_ms_before
        payload: dict[str, Any] = {
            "status": "ok",
            "input": {"path": str(source), "type": "image", "width": frame_width, "height": frame_height},
            "summary": {
                "vehicles": len(final_rows),
                "vehicles_with_plate": sum(row["plate"]["plate_observations"] > 0 for row in final_rows),
                "vehicles_with_ocr": sum(bool(row["plate"]["raw_text"]) for row in final_rows),
                "successful_results": sum(row["plate"]["status"] == "ok" for row in final_rows),
            },
            "vehicles": final_rows,
            "performance": {
                "vehicle_detection_ms": _ms(vehicle_seconds),
                "vehicle_inference_ms": _ms(getattr(self.vehicle_detector, "total_inference_seconds", 0.0) - vehicle_inference_before),
                "plate_detection_ms_per_roi": _ms(self.plate_detector.total_processing_seconds - plate_seconds_before, plate_calls),
                "plate_inference_ms_per_roi": _ms(getattr(self.plate_detector, "total_inference_seconds", 0.0) - plate_inference_before, plate_calls),
                "ownership_ms": _ms(ownership_seconds),
                "v4_quality_ms": _ms(quality_seconds),
                "ocr_ms_per_crop": round(ocr_ms / ocr_calls, 6) if ocr_calls else 0.0,
                "postprocess_ms_per_vehicle": _ms(postprocess_seconds, len(vehicles)),
                "total_ms": _ms(time.perf_counter() - overall_started),
                "session_init_count": self.session_init_count,
            },
        }
        if annotated is not None:
            annotated_path = output_path.parent / f"{artifact_stem}_annotated.jpg"
            if not cv2.imwrite(str(annotated_path), annotated):
                raise OSError(f"Could not write annotated image: {annotated_path}")
            payload["annotated_path"] = str(annotated_path)
        _write_json(output_path, payload)
        payload["output_path"] = str(output_path)
        return payload

    def process_video(
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
        tracker_config = self.config.tracking
        tracker = ByteTracker(
            fps=fps, track_low_threshold=self.config.vehicle.confidence,
            track_high_threshold=tracker_config.high_threshold,
            new_track_threshold=tracker_config.new_track_threshold,
            match_cost_threshold=tracker_config.match_cost_threshold,
            second_match_cost_threshold=tracker_config.second_match_cost_threshold,
            unconfirmed_match_cost_threshold=tracker_config.unconfirmed_match_cost_threshold,
            track_buffer_seconds=tracker_config.track_buffer_seconds,
            min_confirmed_hits=tracker_config.min_confirmed_hits,
            duplicate_iou_threshold=tracker_config.duplicate_iou_threshold,
            cross_class_duplicate_iou_threshold=tracker_config.cross_class_duplicate_iou_threshold,
            cross_class_duplicate_area_ratio_threshold=tracker_config.cross_class_duplicate_area_ratio_threshold,
            debug=detailed,
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
                    candidate = BufferedPlateCandidate(
                        frame_index=frame_index, track_id=plate.track_id,
                        plate_class_id=plate.plate_class_id, plate_class_name=plate.plate_class_name,
                        plate_confidence=plate.plate_confidence, bbox=plate.plate_bbox,
                        quality=quality, crop=crop,
                    )
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
        rows = [result.to_json() for result in final_results.results]
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


__all__ = ["ALPRPipeline", "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS"]
