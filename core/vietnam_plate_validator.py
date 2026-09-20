"""Vietnamese plate structure validation for the T5 post-OCR rule layer.

The validator is deliberately independent of detection and tracking.  It
formats OCR output, evaluates several conservative plate shapes, and applies
only position-aware OCR corrections.  It never performs a global O/0, I/1,
or B/8 replacement.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

from .vietnam_plate_data import province_code_status


VALID = "VALID"
UNCERTAIN = "UNCERTAIN"
INVALID = "INVALID"


@dataclass(frozen=True)
class PlatePattern:
    """A small declarative shape for a common domestic plate family."""

    name: str
    plate_type: str
    positions: tuple[str, ...]

    @property
    def length(self) -> int:
        return len(self.positions)


@dataclass(frozen=True)
class PlateValidationResult:
    original_text: str
    cleaned_text: str
    normalized_text: str
    status: str
    plate_type: str
    pattern_name: str | None
    confidence_or_score: float
    corrections: tuple[str, ...]
    reasons: tuple[str, ...]

    @property
    def score(self) -> float:
        """Short alias useful to callers that do not need the long field name."""

        return self.confidence_or_score


@dataclass(frozen=True)
class _Candidate:
    text: str
    pattern: PlatePattern
    corrections: tuple[str, ...]


class VietnamPlateValidator:
    """Validate OCR strings against conservative Vietnamese plate patterns."""

    # Keep this list broad enough for legacy motorcycle formats while making
    # each accepted shape explicit and testable.
    DEFAULT_PATTERNS: tuple[PlatePattern, ...] = (
        PlatePattern("car_4_digit_serial", "car", ("digit", "digit", "letter", "digit", "digit", "digit", "digit")),
        PlatePattern("car_5_digit_serial", "car", ("digit", "digit", "letter", "digit", "digit", "digit", "digit", "digit")),
        PlatePattern("motorcycle_one_letter_series", "motorcycle", ("digit", "digit", "letter", "digit", "digit", "digit", "digit", "digit", "digit")),
        PlatePattern("motorcycle_two_letter_series", "motorcycle", ("digit", "digit", "letter", "letter", "digit", "digit", "digit", "digit", "digit")),
        PlatePattern("motorcycle_two_letter_4_digit_legacy", "motorcycle", ("digit", "digit", "letter", "letter", "digit", "digit", "digit", "digit")),
    )

    _DIGIT_CORRECTIONS = {
        "O": "0",
        "I": "1",
        "L": "1",
        "Z": "2",
        "S": "5",
        "B": "8",
        "G": "6",
    }
    _LETTER_CORRECTIONS = {
        "0": "O",
        "1": "I",
        "2": "Z",
        "5": "S",
        "6": "G",
        "8": "B",
    }

    def __init__(
        self,
        *,
        correction_enabled: bool = True,
        max_corrections: int = 2,
        patterns: Iterable[PlatePattern] | None = None,
    ) -> None:
        if max_corrections < 0:
            raise ValueError("max_corrections must be non-negative")
        self.correction_enabled = bool(correction_enabled)
        self.max_corrections = int(max_corrections)
        self.patterns = tuple(patterns or self.DEFAULT_PATTERNS)
        if not self.patterns:
            raise ValueError("at least one plate pattern is required")

    @staticmethod
    def clean(text: object) -> tuple[str, str]:
        original = "" if text is None else str(text)
        upper = original.upper()
        cleaned = "".join(
            char for char in upper if "0" <= char <= "9" or "A" <= char <= "Z"
        )
        return original, cleaned

    def validate(self, text: object) -> PlateValidationResult:
        original, cleaned = self.clean(text)
        if not cleaned:
            return self._invalid(
                original,
                cleaned,
                "OCR text is empty",
                plate_type="unknown",
            )

        candidates: list[_Candidate] = []
        for pattern in self.patterns:
            if len(cleaned) != pattern.length:
                continue
            candidates.extend(self._same_length_candidates(cleaned, pattern))

        direct = self._select_direct(candidates)
        if direct is not None:
            prefix_status = province_code_status(direct.text[:2])
            if prefix_status == INVALID:
                return self._invalid(
                    original,
                    cleaned,
                    "province prefix 00 is not a valid domestic plate prefix",
                    plate_type=direct.pattern.plate_type,
                    pattern=direct.pattern,
                    normalized=direct.text,
                    corrections=direct.corrections,
                )
            status = VALID if prefix_status == VALID else UNCERTAIN
            reasons = [
                f"matches {direct.pattern.name}",
                "known province prefix" if prefix_status == VALID else "plausible but unlisted province prefix",
            ]
            if direct.corrections:
                reasons.append("position-aware OCR correction applied")
            score = 1.0 - min(0.45, 0.12 * len(direct.corrections))
            if prefix_status == UNCERTAIN:
                score -= 0.15
            return PlateValidationResult(
                original_text=original,
                cleaned_text=cleaned,
                normalized_text=direct.text,
                status=status,
                plate_type=direct.pattern.plate_type,
                pattern_name=direct.pattern.name,
                confidence_or_score=round(max(0.0, score), 6),
                corrections=direct.corrections,
                reasons=tuple(reasons),
            )

        near = self._near_shape(cleaned)
        if near is not None:
            return PlateValidationResult(
                original_text=original,
                cleaned_text=cleaned,
                normalized_text=cleaned,
                status=UNCERTAIN,
                plate_type=near.plate_type,
                pattern_name=near.name,
                confidence_or_score=0.45,
                corrections=(),
                reasons=(
                    f"close to {near.name}",
                    "one character may be missing or extra",
                ),
            )

        return self._invalid(
            original,
            cleaned,
            "does not match a supported Vietnam plate structure",
        )

    def _same_length_candidates(
        self, cleaned: str, pattern: PlatePattern
    ) -> list[_Candidate]:
        choices: list[tuple[str, ...]] = []
        for char, expected in zip(cleaned, pattern.positions):
            if self._matches(char, expected):
                choices.append((char,))
                continue
            if not self.correction_enabled or self.max_corrections == 0:
                return []
            replacement = (
                self._DIGIT_CORRECTIONS.get(char)
                if expected == "digit"
                else self._LETTER_CORRECTIONS.get(char)
            )
            if replacement is None:
                return []
            # The observed character is not a valid member of this position's
            # alphabet; only the explicitly justified replacement is a
            # candidate.  This prevents O from silently surviving a numeric
            # position and prevents global conversion behavior.
            choices.append((replacement,))

        results: list[_Candidate] = []
        for values in product(*choices):
            corrections = tuple(
                f"{source} -> {target}"
                for source, target in zip(cleaned, values)
                if source != target
            )
            if len(corrections) > self.max_corrections:
                continue
            results.append(_Candidate("".join(values), pattern, corrections))
        return results

    def _select_direct(self, candidates: list[_Candidate]) -> _Candidate | None:
        if not candidates:
            return None

        def key(item: _Candidate) -> tuple[int, int, int, str, str]:
            prefix = province_code_status(item.text[:2])
            status_rank = {VALID: 2, UNCERTAIN: 1, INVALID: 0}[prefix]
            return (
                status_rank,
                -len(item.corrections),
                -self.patterns.index(item.pattern),
                item.text,
                item.pattern.name,
            )

        return max(candidates, key=key)

    def _near_shape(self, cleaned: str) -> PlatePattern | None:
        for pattern in self.patterns:
            if abs(len(cleaned) - pattern.length) != 1:
                continue
            if len(cleaned) < pattern.length:
                # One OCR character is missing. Try every insertion point,
                # using a wildcard for the unknown character.
                for position in range(pattern.length):
                    observed = list(cleaned)
                    observed.insert(position, "?")
                    if self._shape_compatible(observed, pattern):
                        return pattern
            else:
                # One spurious OCR character is present.
                for position in range(len(cleaned)):
                    observed = list(cleaned)
                    del observed[position]
                    if self._shape_compatible(observed, pattern):
                        return pattern
        return None

    def _shape_compatible(self, values: list[str], pattern: PlatePattern) -> bool:
        for char, expected in zip(values, pattern.positions):
            if char == "?":
                continue
            if self._matches(char, expected):
                continue
            if not self.correction_enabled:
                return False
            replacement = (
                self._DIGIT_CORRECTIONS.get(char)
                if expected == "digit"
                else self._LETTER_CORRECTIONS.get(char)
            )
            if replacement is None:
                return False
        return True

    @staticmethod
    def _matches(char: str, expected: str) -> bool:
        return (expected == "digit" and "0" <= char <= "9") or (
            expected == "letter" and "A" <= char <= "Z"
        )

    @staticmethod
    def _invalid(
        original: str,
        cleaned: str,
        reason: str,
        *,
        plate_type: str = "unknown",
        pattern: PlatePattern | None = None,
        normalized: str | None = None,
        corrections: tuple[str, ...] = (),
    ) -> PlateValidationResult:
        return PlateValidationResult(
            original_text=original,
            cleaned_text=cleaned,
            normalized_text=cleaned if normalized is None else normalized,
            status=INVALID,
            plate_type=plate_type,
            pattern_name=pattern.name if pattern is not None else None,
            confidence_or_score=0.0,
            corrections=corrections,
            reasons=(reason,),
        )


__all__ = [
    "INVALID",
    "UNCERTAIN",
    "VALID",
    "PlatePattern",
    "PlateValidationResult",
    "VietnamPlateValidator",
]
