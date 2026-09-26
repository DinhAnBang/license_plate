"""Still-image ALPR execution using shared detection and OCR stages."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2

from ..ocr_stage import run_ocr_on_image_candidates
from ..pipeline_support import _debug_ocr, _draw_box, _ms, _plate_display_label, _write_json
from ..plate_buffer import BufferedPlateCandidate
from ..plate_ownership_temporal import TemporalPlateOwnershipResolver
from ..plate_quality import crop_plate_from_frame, score_plate_quality
from ..plate_stage import buffered_plate_candidate, detect_vehicle_plates
from ..plate_types import TrackedPlateCandidate
from ..result_finalizer import finalize_vehicle
from ..result_serialization import serialize_vehicle_result
from ..vn_plate_postprocessor import (
    postprocess_vietnam_plate,
    preferred_family_for_vehicle_class,
)
from ..vehicle_stage import detect_image_vehicles


def run_image(
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
    vehicles = detect_image_vehicles(
        image, self.vehicle_detector, self.config.vehicle.image_confidence,
    )
    vehicle_seconds = time.perf_counter() - stage
    annotated = image.copy() if save_annotated else None
    raw: list[TrackedPlateCandidate] = []
    for vehicle_index, vehicle in enumerate(vehicles):
        if annotated is not None:
            _draw_box(annotated, vehicle.bbox, f"{vehicle_index} {vehicle.class_name}", (0, 200, 0))
        raw.extend(
            detect_vehicle_plates(
                image, vehicle, vehicle_index, self.plate_detector, 0,
            )
        )
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
        image_candidates.append(buffered_plate_candidate(plate, crop, quality))
        if save_topk_crops:
            crop_path = output_path.parent / "crops" / f"vehicle_{plate.track_id}.jpg"
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
        normalized = postprocess_vietnam_plate(
            raw_text,
            confidence,
            self.config.vietnam,
            char_confidences=ocr.char_confidences if ocr else None,
            preferred_family=preferred_family_for_vehicle_class(vehicle.class_name),
        )
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
        row = serialize_vehicle_result(final)
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
    if annotated is not None:
        plate_labels = {
            int(row["vehicle_index"]): _plate_display_label(row["plate"])
            for row in final_rows
        }
        for plate in resolution.candidates:
            _draw_box(
                annotated,
                plate.plate_bbox,
                plate_labels.get(plate.track_id, "unreadable"),
                (0, 0, 255),
            )
    plate_calls = self.plate_detector.detect_call_count - plate_calls_before
    ocr_calls = self.ocr_engine.inference_count - ocr_calls_before
    ocr_ms = self.ocr_engine.timing_totals["total_ms_per_crop"] * self.ocr_engine.inference_count - ocr_ms_before
    payload: dict[str, Any] = {
        "status": "ok",
        "input": {"path": str(source), "type": "image", "width": frame_width, "height": frame_height},
        "summary": {
            "detected_vehicles": len(vehicles),
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        annotated_path = output_path.parent / f"{artifact_stem}_annotated.jpg"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise OSError(f"Could not write annotated image: {annotated_path}")
        payload["annotated_path"] = str(annotated_path)
    _write_json(output_path, payload)
    payload["output_path"] = str(output_path)
    return payload
