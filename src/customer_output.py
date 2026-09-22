"""Small customer-facing output derived from the full development result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _license_text(plate: dict[str, Any]) -> str | None:
    """Return the most useful public plate representation, if any."""

    for key in ("formatted", "text", "normalized_text", "raw_text"):
        value = plate.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_customer_payload(result: dict[str, Any]) -> dict[str, Any]:
    """Reduce the internal result to the release/EXE JSON contract.

    Development output remains unchanged.  The public confidence is the final
    plate/OCR fusion confidence produced by the complete pipeline.
    """

    vehicles: list[dict[str, Any]] = []
    for item in result.get("vehicles", []):
        vehicle = item.get("vehicle", {})
        plate = item.get("plate", {})
        identity = item.get("track_id", item.get("vehicle_index"))
        vehicles.append(
            {
                "track_id": identity,
                "vehicle_type": vehicle.get("class_name"),
                "license": _license_text(plate),
                "confidence": plate.get("confidence", 0.0),
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


__all__ = ["build_customer_payload", "write_customer_json"]
