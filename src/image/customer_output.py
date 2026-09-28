"""Small customer-facing output derived from the full development result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..core.customer_result import is_customer_result, license_text as _license_text


def build_customer_payload(
    result: dict[str, Any], *, min_confidence: float | None = None,
) -> dict[str, Any]:
    """Reduce the internal result to the release/EXE JSON contract.

    Development output remains unchanged.  The public confidence is the final
    plate/OCR fusion confidence produced by the complete pipeline.
    """

    if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")

    vehicles: list[dict[str, Any]] = []
    for item in result.get("vehicles", []):
        vehicle = item.get("vehicle", {})
        plate = item.get("plate", {})
        confidence = float(plate.get("confidence", 0.0))
        if not is_customer_result(item, min_confidence=min_confidence):
            continue
        identity = item.get("vehicle_index")
        if identity is None:
            continue
        vehicles.append(
            {
                "vehicle_index": identity,
                "vehicle_type": vehicle.get("class_name"),
                "license": _license_text(plate),
                "confidence": confidence,
                "status": plate.get("status"),
            }
        )

    return {
        "status": result.get("status", "ok"),
        "vehicles": vehicles,
    }


def write_customer_json(path: str | Path, payload: dict[str, Any]) -> None:
    """Write UTF-8 customer JSON with stable human-readable formatting."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "build_customer_payload",
    "is_customer_result",
    "write_customer_json",
]
