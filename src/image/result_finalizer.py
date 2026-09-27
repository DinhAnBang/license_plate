"""Final public result for one detected image vehicle."""

from __future__ import annotations

from dataclasses import dataclass

from ..ocr_fusion import FusedOCRResult
from .vn_plate_postprocessor import VietnamPlateResult


@dataclass(frozen=True, slots=True)
class FinalVehicleResult:
    identity_key: str
    identity: int
    vehicle_class_id: int
    vehicle_class_name: str
    first_frame: int | None
    last_frame: int | None
    vehicle_observation_count: int
    plate_observation_count: int
    plate_text_raw: str
    plate_text_normalized: str
    plate_text: str
    plate_formatted: str | None
    plate_format_family: str | None
    ocr_confidence: float
    plate_layout: str | None
    best_plate_bbox: tuple[int, int, int, int] | None
    best_plate_frame: int | None
    status: str
    status_reason: str | None
    postprocess_status: str
    fusion_method: str | None
    support_count: int
    ocr_candidate_count: int
    format_valid: bool = False
    format_reason: str | None = None

def finalize_vehicle(
    *,
    identity_key: str,
    identity: int,
    vehicle_class_id: int,
    vehicle_class_name: str,
    first_frame: int | None,
    last_frame: int | None,
    vehicle_observation_count: int,
    plate_observation_count: int,
    plate_layout: str | None,
    best_plate_bbox: tuple[int, int, int, int] | None,
    best_plate_frame: int | None,
    postprocessed: VietnamPlateResult,
    fusion: FusedOCRResult | None,
    ocr_candidate_count: int,
    low_confidence_threshold: float = 0.50,
) -> FinalVehicleResult:
    if identity_key != "vehicle_index":
        raise ValueError("image identity_key must be vehicle_index")
    confidence = postprocessed.fusion_confidence
    family_mismatch = (
        vehicle_class_name in {"car", "bus", "truck"}
        and postprocessed.format_family == "motorbike_common"
    )
    format_valid = postprocessed.format_valid and not family_mismatch
    format_reason = postprocessed.format_reason
    if family_mismatch:
        format_reason = "vehicle_plate_family_mismatch"
    if plate_observation_count <= 0:
        status = "no_plate"
        status_reason = None
    elif not postprocessed.raw_text:
        status = "no_ocr"
        status_reason = None
    elif family_mismatch:
        status = "low_confidence"
        status_reason = "vehicle_plate_family_mismatch"
    elif confidence < low_confidence_threshold or postprocessed.status == "low_format_confidence":
        status = "low_confidence"
        status_reason = "weak_ocr_evidence" if confidence < low_confidence_threshold else "too_many_position_corrections"
    elif postprocessed.status == "unrecognized_format":
        status = "unrecognized_format"
        status_reason = "no_known_format_family"
    else:
        status = "ok"
        status_reason = None

    return FinalVehicleResult(
        identity_key=identity_key,
        identity=int(identity),
        vehicle_class_id=int(vehicle_class_id),
        vehicle_class_name=vehicle_class_name,
        first_frame=first_frame,
        last_frame=last_frame,
        vehicle_observation_count=vehicle_observation_count,
        plate_observation_count=plate_observation_count,
        plate_text_raw=postprocessed.raw_text,
        plate_text_normalized=postprocessed.normalized_text,
        plate_text=postprocessed.corrected_text,
        plate_formatted=None if family_mismatch else postprocessed.formatted_text,
        plate_format_family=postprocessed.format_family,
        ocr_confidence=confidence,
        plate_layout=plate_layout,
        best_plate_bbox=best_plate_bbox,
        best_plate_frame=best_plate_frame,
        status=status,
        status_reason=status_reason,
        postprocess_status=postprocessed.status,
        fusion_method=fusion.method if fusion else None,
        support_count=fusion.support_count if fusion else int(bool(postprocessed.raw_text)),
        ocr_candidate_count=ocr_candidate_count,
        format_valid=format_valid,
        format_reason=format_reason,
    )


__all__ = ["FinalVehicleResult", "finalize_vehicle"]
