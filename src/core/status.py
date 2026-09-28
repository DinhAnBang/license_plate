"""Common missing-evidence status, used before mode-specific status rules."""

from __future__ import annotations


def missing_evidence_status(
    plate_observation_count: int, has_ocr_text: bool,
) -> str | None:
    if plate_observation_count <= 0:
        return "no_plate"
    if not has_ocr_text:
        return "no_ocr"
    return None
