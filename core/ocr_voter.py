"""Deterministic full-string voting over OCR results from one track."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class OCRVoter:
    @staticmethod
    def weight(candidate: Mapping[str, Any]) -> float:
        return 0.70 * float(candidate["ocr_conf"]) + 0.30 * float(candidate["quality"])

    @classmethod
    def vote(cls, candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        groups: dict[str, list[int]] = {}
        for index, candidate in enumerate(candidates):
            text = str(candidate["text"])
            if text:
                groups.setdefault(text, []).append(index)

        if not groups:
            return {"raw_text": "", "text": "", "ocr_conf": 0.0, "winner_index": None, "vote_weight": 0.0}

        def group_key(indexes: list[int]) -> tuple[float, float, float, float, int]:
            weights = [cls.weight(candidates[index]) for index in indexes]
            return (
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
        }
