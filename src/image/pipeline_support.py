"""Media annotation, debug evidence and JSON primitives for pipeline runners."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .ocr_stage import OCRPlateCandidate


def _ms(seconds: float, count: int = 1) -> float:
    return round(seconds * 1000.0 / count, 6) if count else 0.0


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def _draw_box(frame: np.ndarray, bbox: tuple[int, int, int, int], label: str, color: tuple[int, int, int]) -> None:
    x1, y1, x2, y2 = (int(value) for value in bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (max(0, x1), max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def _plate_display_label(plate: dict[str, Any]) -> str:
    """Return recognized plate text for annotations, with a clear empty fallback."""

    for key in ("formatted", "text", "normalized_text", "raw_text"):
        value = plate.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unreadable"


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
