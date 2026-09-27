"""JSON representation of finalized ALPR domain results."""

from __future__ import annotations

from typing import Any

from .result_finalizer import FinalVehicleResult


def serialize_vehicle_result(result: FinalVehicleResult) -> dict[str, Any]:
    vehicle: dict[str, Any] = {
        "class_id": result.vehicle_class_id,
        "class_name": result.vehicle_class_name,
        "first_frame": result.first_frame,
        "last_frame": result.last_frame,
        "observations": result.vehicle_observation_count,
    }
    plate: dict[str, Any] = {
        "status": result.status,
        "status_reason": result.status_reason,
        "postprocess_status": result.postprocess_status,
        "raw_text": result.plate_text_raw,
        "normalized_text": result.plate_text_normalized,
        "text": result.plate_text,
        "formatted": result.plate_formatted,
        "format_family": result.plate_format_family,
        "format_valid": result.format_valid,
        "format_reason": result.format_reason,
        "confidence": round(result.ocr_confidence, 6),
        "plate_observations": result.plate_observation_count,
        "layout": result.plate_layout,
        "best_bbox_xyxy": list(result.best_plate_bbox) if result.best_plate_bbox else None,
        "best_frame": result.best_plate_frame,
    }
    evidence: dict[str, Any] = {
        "ocr_candidates": result.ocr_candidate_count,
        "fusion_method": result.fusion_method,
        "support_count": result.support_count,
    }
    return {result.identity_key: result.identity, "vehicle": vehicle, "plate": plate, "evidence": evidence}
