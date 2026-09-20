"""Deterministic full-string voting over OCR results from one track."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .config import OCR_VOTE_CONFIDENCE_WEIGHT, OCR_VOTE_QUALITY_WEIGHT


class OCRVoter:
    _VALIDATION_RANK = {"INVALID": 0, "UNCERTAIN": 1, "VALID": 2}

    @staticmethod
    def weight(candidate: Mapping[str, Any]) -> float:
        return OCR_VOTE_CONFIDENCE_WEIGHT * float(candidate["ocr_conf"]) + OCR_VOTE_QUALITY_WEIGHT * float(candidate["quality"])

    @classmethod
    def validation_rank(cls, candidate: Mapping[str, Any]) -> int:
        # Missing metadata is the legacy voter behavior and remains neutral.
        return cls._VALIDATION_RANK.get(str(candidate.get("validation_status", "")), 1)

    @classmethod
    def vote(cls, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        groups: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            text = str(candidate["text"])
            if text:
                groups.setdefault(text, []).append(index)

        if not groups:
            return {"raw_text": "", "text": "", "ocr_conf": 0.0, "winner_index": None, "vote_weight": 0.0}

        def group_key(indexes: list[int]) -> tuple[float, float, float, float, float, float, int]:
            weights = [cls.weight(candidates[index]) for index in indexes]
            return (
                float(max(cls.validation_rank(candidates[index]) for index in indexes)),
                round(sum(float(candidates[index].get("validation_score", 0.0)) for index in indexes) / len(indexes), 9),
                round(sum(weights), 9),
                round(max(weights), 9),
                round(sum(float(candidates[index]["ocr_conf"]) for index in indexes) / len(indexes), 9),
                round(max(float(candidates[index]["quality"]) for index in indexes), 9),
                -min(indexes),
            )

        winning_text, winning_indexes = max(groups.items(), key=lambda item: group_key(item[1]))
        winner_index = max(winning_indexes, key=lambda index: (cls.weight(candidates[index]), -index))
        winner = candidates[winner_index]
        return {
            "raw_text": str(winner["raw_text"]),
            "text": winning_text,
            "ocr_conf": float(winner["ocr_conf"]),
            "winner_index": winner_index,
            "vote_weight": cls.weight(winner),
            "validation_status": str(winner.get("validation_status", "")),
            "validation_score": float(winner.get("validation_score", 0.0)),
        }
