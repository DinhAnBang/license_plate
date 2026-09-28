"""Release-only video JSON; complete per-track evidence stays in the core result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_video_customer_payload(
    result: dict[str, Any], *, min_confidence: float | None = None,
) -> dict[str, Any]:
    if min_confidence is not None and not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be in [0, 1]")
    vehicles = []
    for item in result.get("vehicles", []):
        plate = item["plate"]
        confidence = float(plate["confidence"])
        if (plate["status"] != "ok" or not plate["format_valid"]
                or not plate["formatted"]
                or (min_confidence is not None and confidence < min_confidence)):
            continue
        vehicles.append({
            "track_id": item["track_id"],
            "vehicle_type": item["vehicle"]["class_name"],
            "license": plate["formatted"],
            "confidence": confidence,
            "status": "ok",
        })
    return {"status": result.get("status", "ok"), "vehicles": vehicles}


def write_video_customer_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


__all__ = ["build_video_customer_payload", "write_video_customer_json"]
