"""Debug evidence, timing and JSON helpers for the image pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .ocr_stage import OCRPlateCandidate


def _ms(seconds: float, count: int = 1) -> float:
    return round(seconds * 1000.0 / count, 6) if count else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _debug_ocr(candidate: OCRPlateCandidate) -> dict[str, Any]:
    return {
        "rank": candidate.rank,
        "frame_index": candidate.frame_index,
        "raw_text": candidate.raw_text,
        "ocr_confidence": round(candidate.ocr_confidence, 6),
        "quality_score": round(candidate.quality_score, 6),
        "char_confidences": list(candidate.char_confidences) if candidate.char_confidences else None,
        "status": candidate.status,
        "error": candidate.error,
    }
