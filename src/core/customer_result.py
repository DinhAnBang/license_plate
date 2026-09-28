"""Shared release-output eligibility and text selection."""

from __future__ import annotations

from typing import Any


def license_text(plate: dict[str, Any]) -> str | None:
    for key in ("formatted", "text", "normalized_text", "raw_text"):
        value = plate.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def is_customer_result(
    item: dict[str, Any], *, min_confidence: float | None = None,
    require_formatted: bool = False,
) -> bool:
    plate = item.get("plate", {})
    confidence = float(plate.get("confidence", 0.0))
    if plate.get("status") != "ok" or plate.get("format_valid") is not True:
        return False
    if require_formatted:
        if not plate.get("formatted"):
            return False
    elif not license_text(plate):
        return False
    return min_confidence is None or confidence >= min_confidence
