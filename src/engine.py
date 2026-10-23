"""Persistent JSON-lines interface for the C# application."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO
from uuid import uuid4

from .alpr_pipeline import ALPRPipeline, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from .core.customer_result import is_customer_result, license_text
from .paths import application_directory


def _emit(stream: TextIO, payload: dict[str, Any]) -> None:
    stream.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    stream.flush()


def _response(result: dict[str, Any], request_id: str, media_type: str, elapsed_ms: float,
              min_confidence: float) -> dict[str, Any]:
    plates = []
    for item in result.get("vehicles", []):
        if not is_customer_result(item, min_confidence=min_confidence,
                                  require_formatted=media_type == "video"):
            continue
        plate = item["plate"]
        plates.append({
            "plate_text": license_text(plate),
            "detection_confidence": float(plate["confidence"]),
            "ocr_confidence": float(plate["confidence"]),
            "crop_path": None,
        })
    return {
        "request_id": request_id,
        "status": "success",
        "input_type": media_type,
        "output_image": result.get("annotated_path") if media_type == "image" else None,
        "output_video": result.get("annotated_path") if media_type == "video" else None,
        "output_json": result.get("output_path"),
        "processing": {"total_ms": round(elapsed_ms, 3)},
        "count": len(plates),
        "plates": plates,
    }


def _handle(request: Any, pipeline: ALPRPipeline) -> tuple[dict[str, Any], bool]:
    request_id = request.get("id") if isinstance(request, dict) else None
    request_id = request_id if isinstance(request_id, str) else ""
    try:
        if not isinstance(request, dict):
            raise ValueError("Request must be a JSON object")
        action = request.get("action")
        if action == "shutdown":
            return {"request_id": request_id, "status": "success", "action": "shutdown"}, True
        if action == "ping":
            return {"request_id": request_id, "status": "success", "action": "ping"}, False
        if action != "process":
            raise ValueError(f"Unsupported action: {action}")
        raw_path = request.get("path")
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("path must be a nonempty string")
        source = Path(raw_path).resolve()
        if not source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source}")
        media_type = request.get("type")
        extensions = IMAGE_EXTENSIONS if media_type == "image" else VIDEO_EXTENSIONS if media_type == "video" else None
        if extensions is None or source.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported type or extension: {media_type}, {source.suffix}")
        # Write each request's files beside the executable, as with an
        # initial --input PATH. The suffix prevents same-second collisions.
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = application_directory() / f"{source.stem}_{timestamp}_{uuid4().hex[:8]}"
        output = output_dir / f"{source.stem}.json"
        started = time.perf_counter()
        if media_type == "image":
            result = pipeline.process_image(source, output=output, save_annotated=True)
        else:
            result = pipeline.process_video(source, output=output, save_annotated=True)
        return _response(result, request_id, media_type,
                         (time.perf_counter() - started) * 1000,
                         pipeline.config.low_confidence_threshold), False
    except Exception as exc:
        return {
            "request_id": request_id,
            "status": "error",
            "error": {"code": type(exc).__name__, "message": str(exc)},
        }, False


def run_input_stream(device: str = "auto", *, pipeline: ALPRPipeline | None = None,
                     debug: bool = False, input_stream: TextIO | None = None,
                     output_stream: TextIO | None = None) -> int:
    """Keep model sessions alive and process requests until shutdown or stdin EOF."""
    incoming = input_stream if input_stream is not None else sys.stdin
    outgoing = output_stream if output_stream is not None else sys.stdout
    _emit(outgoing, {"event": "starting"})
    try:
        if pipeline is None:
            with contextlib.redirect_stdout(sys.stderr):
                pipeline = ALPRPipeline(device=device, debug=debug)
    except Exception as exc:
        print(f"ALPR startup error: {exc}", file=sys.stderr)
        return 1
    _emit(outgoing, {"event": "ready"})
    for line in incoming:
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _emit(outgoing, {"status": "error", "error": {
                "code": "JSONDecodeError", "message": str(exc),
            }})
            continue
        with contextlib.redirect_stdout(sys.stderr):
            response, should_stop = _handle(request, pipeline)
        _emit(outgoing, response)
        if should_stop:
            break
    return 0


__all__ = ["run_input_stream"]
