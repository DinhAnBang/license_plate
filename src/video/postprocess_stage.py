"""Conservative video adapter for the existing Vietnamese plate postprocessor."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.plate_postprocess import (
    VietnamPlateResult, VietnamPostprocessConfig,
    postprocess_vietnam_plate, preferred_family_for_vehicle_class,
)


@dataclass(frozen=True, slots=True)
class VideoPostprocessResult:
    status: str
    reason: str | None
    valid: bool
    formatted_text: str | None
    postprocessed: VietnamPlateResult | None


def postprocess_video_fused(
    raw_text: str | None,
    fusion_confidence: float,
    vehicle_class_name: str,
    *,
    config: VietnamPostprocessConfig | None = None,
) -> VideoPostprocessResult:
    """Never infer characters or call postprocessing for a track without OCR.

    Video fusion has no reliably aligned per-character confidence. Passing
    ``None`` explicitly disables the shared postprocessor's optional
    low-confidence deletion fallback.
    """
    if raw_text is None or not raw_text.strip():
        return VideoPostprocessResult("no_ocr", "no_ocr_candidates", False, None, None)
    preferred_family = preferred_family_for_vehicle_class(vehicle_class_name)
    result = postprocess_vietnam_plate(
        raw_text,
        fusion_confidence,
        config=config,
        char_confidences=None,
        preferred_family=preferred_family,
    )
    family_mismatch = (
        preferred_family is not None
        and result.format_family is not None
        and result.format_family != preferred_family
    )
    if family_mismatch:
        return VideoPostprocessResult(
            "vehicle_plate_family_mismatch", "vehicle_plate_family_mismatch",
            False, None, result,
        )
    if not result.format_valid:
        return VideoPostprocessResult(
            result.status, result.format_reason, False, None, result,
        )
    return VideoPostprocessResult(
        result.status, None, True, result.formatted_text, result,
    )


__all__ = ["VideoPostprocessResult", "postprocess_video_fused"]
