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

# Current domestic locality codes from Appendix 02 of Circular 51/2025/TT-BCA.
# Existing plates keep their issued code, so the list intentionally contains
# all codes published in the appendix rather than only one code per locality.
_LOCALITY_CODES = frozenset({
    "11", "12", "14", "15", "16", "17", "18", "19", "20", "21",
    "22", "23", "24", "25", "26", "27", "28", "29", "30", "31",
    "32", "33", "34", "35", "36", "37", "38", "39", "40", "41",
    "43", "47", "48", "49", "50", "51", "52", "53", "54", "55",
    "56", "57", "58", "59", "60", "61", "62", "63", "64", "65",
    "66", "67", "68", "69", "70", "71", "72", "73", "74", "75",
    "76", "77", "78", "79", "80", "81", "82", "83", "84", "85",
    "86", "88", "89", "90", "92", "93", "94", "95", "97", "98",
    "99",
})
_SERIAL_LETTERS = frozenset("ABCDEFGHKLMNPSTUVXYZ")
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
    format_valid: bool = False
    format_reason: str | None = None


def _validate_common_format(text: str, family: str) -> tuple[bool, str | None]:
    """Validate a normalized domestic common plate without guessing characters."""

    if len(text) < 2 or not text[:2].isdigit():
        return False, "invalid_locality_prefix"
    if text[:2] not in _LOCALITY_CODES:
        return False, "unknown_locality_code"

    if family == "car_common":
        if len(text) not in {7, 8}:
            return False, "invalid_length"
        serial = text[2:3]
        registration_number = text[3:]
    elif family == "motorbike_common":
        if len(text) not in {8, 9}:
            return False, "invalid_length"
        serial = text[2:4]
        registration_number = text[4:]
    else:
        return False, "unsupported_family"

    if not serial or serial[0] not in _SERIAL_LETTERS:
        return False, "invalid_serial"
    if family == "motorbike_common" and len(serial) == 2:
        # A digit in the second serial position is retained for older plates
        # such as 81B1-989.45; current civilian series also use two letters.
        if serial[1] not in _SERIAL_LETTERS and serial[1] not in "0123456789":
            return False, "invalid_serial"
    if not registration_number.isdigit():
        return False, "invalid_registration_number"
    return True, None


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
               corrections: tuple[PlateCorrection, ...] = (),
               format_valid: bool = False,
               format_reason: str | None = None) -> VietnamPlateResult:
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
            format_valid=format_valid,
            format_reason=format_reason,
        )

    if not normalized:
        return result("no_text", "", format_reason="empty_text")

    options: list[tuple[int, int, str, str, tuple[PlateCorrection, ...]]] = []
    rejected_reasons: list[str] = []
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
            candidate = "".join(corrected)
            valid, reason = _validate_common_format(candidate, family)
            if valid:
                options.append((len(changes), family_priority, family, candidate, tuple(changes)))
            elif reason is not None:
                rejected_reasons.append(reason)

    if not options:
        return result(
            "unrecognized_format",
            normalized,
            format_reason=(rejected_reasons[0] if rejected_reasons else "invalid_length"),
        )
    cost, _, family, corrected, changes = min(options, key=lambda item: (item[0], item[1]))
    if cost > config.max_position_corrections:
        return result(
            "low_format_confidence",
            normalized,
            format_reason="too_many_position_corrections",
        )
    return result(
        "corrected" if cost else "ok",
        corrected,
        family,
        changes,
        format_valid=True,
    )


__all__ = ["PlateCorrection", "VietnamPlateResult", "VietnamPostprocessConfig", "postprocess_vietnam_plate"]
