"""Conservative Vietnamese plate display suggestions for fused raw OCR."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


_DIGIT_TO_LETTERS: dict[str, tuple[str, ...]] = {
    "0": ("O",), "1": ("I", "L"), "2": ("Z",),
    "5": ("S",), "6": ("G",), "8": ("B",),
}
_LETTER_TO_DIGIT = {
    letter: digit for digit, letters in _DIGIT_TO_LETTERS.items() for letter in letters
}
_TEMPLATES: tuple[tuple[str, str], ...] = (
    ("car_common", "DDLDDDD"),
    ("car_common", "DDLDDDDD"),
    ("motorbike_common", "DDLADDDD"),
    ("motorbike_common", "DDLADDDDD"),
)


@dataclass(frozen=True, slots=True)
class VietnamPostprocessConfig:
    separators: str = " -."
    max_position_corrections: int = 2

    def __post_init__(self) -> None:
        if self.max_position_corrections < 0:
            raise ValueError("max_position_corrections must be nonnegative")
        if any(char.isalnum() for char in self.separators):
            raise ValueError("separators cannot contain letters or digits")


@dataclass(frozen=True, slots=True)
class PlateCorrection:
    index: int
    from_char: str
    to_char: str
    reason: str


@dataclass(frozen=True, slots=True)
class VietnamPlateResult:
    raw_text: str
    normalized_text: str
    corrected_text: str
    formatted_text: str | None
    format_family: str | None
    corrections: tuple[PlateCorrection, ...]
    correction_count: int
    status: str
    fusion_confidence: float
    unknown_characters: tuple[tuple[int, str], ...] = ()


def _expect(char: str, expectation: str) -> tuple[str, str | None] | None:
    if expectation == "A":
        return (char, None) if char in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" else None
    if expectation == "D":
        if char in "0123456789":
            return char, None
        digit = _LETTER_TO_DIGIT.get(char)
        return (digit, "digit_expected") if digit is not None else None
    if char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return char, None
    letters = _DIGIT_TO_LETTERS.get(char)
    # 1 could be I or L. Do not make an arbitrary choice at a letter position.
    return (letters[0], "letter_expected") if letters is not None and len(letters) == 1 else None


def _format(text: str, family: str) -> str:
    prefix_length = 3 if family == "car_common" else 4
    suffix = text[prefix_length:]
    split = len(suffix) - 2
    return f"{text[:prefix_length]}-{suffix[:split]}.{suffix[split:]}"


def postprocess_vietnam_plate(
    raw_text: str,
    fusion_confidence: float = 0.0,
    config: VietnamPostprocessConfig | None = None,
) -> VietnamPlateResult:
    """Suggest a known family using only position compatible OCR confusions.

    Unknown formats and excessive corrections keep the normalized source text.
    The original OCR fusion confidence is carried through without promotion.
    """

    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be str")
    if not isfinite(fusion_confidence) or not 0.0 <= fusion_confidence <= 1.0:
        raise ValueError("fusion_confidence must be finite and in [0, 1]")
    config = config or VietnamPostprocessConfig()
    stripped = raw_text.strip()
    normalized = "".join(
        chr(ord(char) - 32) if "a" <= char <= "z" else char
        for char in stripped
        if char not in config.separators
    )
    unknown = tuple(
        (index, char)
        for index, char in enumerate(normalized)
        if char not in "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )

    def result(status: str, corrected: str, family: str | None = None,
               corrections: tuple[PlateCorrection, ...] = ()) -> VietnamPlateResult:
        return VietnamPlateResult(
            raw_text=raw_text,
            normalized_text=normalized,
            corrected_text=corrected,
            formatted_text=_format(corrected, family) if family else None,
            format_family=family,
            corrections=corrections,
            correction_count=len(corrections),
            status=status,
            fusion_confidence=fusion_confidence,
            unknown_characters=unknown,
        )

    if not normalized:
        return result("no_text", "")

    options: list[tuple[int, int, str, str, tuple[PlateCorrection, ...]]] = []
    for family_priority, (family, pattern) in enumerate(_TEMPLATES):
        if len(normalized) != len(pattern):
            continue
        corrected: list[str] = []
        changes: list[PlateCorrection] = []
        for index, (char, expectation) in enumerate(zip(normalized, pattern)):
            option = _expect(char, expectation)
            if option is None:
                break
            replacement, reason = option
            corrected.append(replacement)
            if reason is not None:
                changes.append(PlateCorrection(index, char, replacement, reason))
        else:
            options.append((len(changes), family_priority, family, "".join(corrected), tuple(changes)))

    if not options:
        return result("unrecognized_format", normalized)
    cost, _, family, corrected, changes = min(options, key=lambda item: (item[0], item[1]))
    if cost > config.max_position_corrections:
        return result("low_format_confidence", normalized)
    return result("corrected" if cost else "ok", corrected, family, changes)


__all__ = ["PlateCorrection", "VietnamPlateResult", "VietnamPostprocessConfig", "postprocess_vietnam_plate"]
