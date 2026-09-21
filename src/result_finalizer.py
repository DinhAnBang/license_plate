"""One final public result for every confirmed video track or image vehicle."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .ocr_fusion import FusedOCRResult
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

    def to_json(self, debug: bool = False) -> dict[str, Any]:
        vehicle: dict[str, Any] = {
            "class_id": self.vehicle_class_id,
            "class_name": self.vehicle_class_name,
            "first_frame": self.first_frame,
            "last_frame": self.last_frame,
            "observations": self.vehicle_observation_count,
        }
        plate: dict[str, Any] = {
            "status": self.status,
            "status_reason": self.status_reason,
            "postprocess_status": self.postprocess_status,
            "raw_text": self.plate_text_raw,
            "normalized_text": self.plate_text_normalized,
            "text": self.plate_text,
            "formatted": self.plate_formatted,
            "format_family": self.plate_format_family,
            "confidence": round(self.ocr_confidence, 6),
            "plate_observations": self.plate_observation_count,
            "layout": self.plate_layout,
            "best_bbox_xyxy": list(self.best_plate_bbox) if self.best_plate_bbox else None,
            "best_frame": self.best_plate_frame,
        }
        evidence: dict[str, Any] = {
            "ocr_candidates": self.ocr_candidate_count,
            "fusion_method": self.fusion_method,
            "support_count": self.support_count,
        }
        return {self.identity_key: self.identity, "vehicle": vehicle, "plate": plate, "evidence": evidence}


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
    if identity_key not in {"track_id", "vehicle_index"}:
        raise ValueError("identity_key must be track_id or vehicle_index")
    confidence = postprocessed.fusion_confidence
    family_mismatch = (
        vehicle_class_name in {"car", "bus", "truck"}
        and postprocessed.format_family == "motorbike_common"
    )
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
    )


class FinalResultCollector:
    """Store one final result per confirmed video track, including EOF flush."""

    def __init__(self) -> None:
        self._results: dict[int, FinalVehicleResult] = {}

    def add(self, result: FinalVehicleResult) -> None:
        if result.identity_key != "track_id":
            raise ValueError("video finalization requires track_id")
        if result.identity in self._results:
            raise ValueError(f"track {result.identity} already finalized")
        self._results[result.identity] = result

    @property
    def results(self) -> tuple[FinalVehicleResult, ...]:
        return tuple(self._results[key] for key in sorted(self._results))


__all__ = ["FinalVehicleResult", "FinalResultCollector", "finalize_vehicle"]
