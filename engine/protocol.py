"""Validation and response helpers for the JSON Lines engine protocol."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


@dataclass(frozen=True)
class ProtocolError(ValueError):
    code: str
    message: str

    def __str__(self) -> str:
        return self.message


REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
WINDOWS_RESERVED_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def validate_request_id(value: Any) -> str:
    if (
        not isinstance(value, str)
        or REQUEST_ID_PATTERN.fullmatch(value) is None
        or value.upper() in WINDOWS_RESERVED_NAMES
    ):
        raise ProtocolError(
            "INVALID_REQUEST_ID",
            "Request id must contain 1-80 ASCII letters, digits, hyphens, or underscores and not be a Windows reserved name.",
        )
    return value


def validate_request(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProtocolError("INVALID_REQUEST", "Request must be a JSON object.")

    validate_request_id(value.get("id"))

    action = value.get("action")
    if not isinstance(action, str) or not action:
        raise ProtocolError("INVALID_REQUEST", "Request action is required.")
    if action not in {"process", "ping", "shutdown"}:
        raise ProtocolError("UNSUPPORTED_ACTION", f"Unsupported action: {action}")

    if action == "process":
        media_type = value.get("type")
        if not isinstance(media_type, str) or not media_type:
            raise ProtocolError("INVALID_REQUEST", "Process request type is required.")
        if media_type not in {"image", "video"}:
            raise ProtocolError("UNSUPPORTED_TYPE", f"Unsupported type: {media_type}")
        path = value.get("path")
        if not isinstance(path, str) or not path.strip():
            raise ProtocolError("INVALID_REQUEST", "Process request path must be a non-empty string.")

    return value


def error_response(request_id: Any, code: str, message: str) -> dict[str, Any]:
    return {
        "id": request_id if isinstance(request_id, (str, int)) and not isinstance(request_id, bool) else None,
        "status": "error",
        "error": {"code": code, "message": message[:300]},
    }
