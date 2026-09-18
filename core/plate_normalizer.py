"""Safe formatting of OCR plate text without character correction."""

from __future__ import annotations


class PlateNormalizer:
    @staticmethod
    def normalize(text: str) -> str:
        if not isinstance(text, str):
            raise TypeError("plate text must be a string")
        return "".join(char for char in text.upper() if "0" <= char <= "9" or "A" <= char <= "Z")
