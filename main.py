"""JSON Lines stdin/stdout entry point for the persistent AI engine."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

from engine import AIPlateEngine
from engine.protocol import error_response


def emit(value: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description="Persistent license plate AI engine over JSON Lines.")
    parser.add_argument("--detector-model", type=Path, default=Path("models/best.onnx"))
    parser.add_argument("--ocr-model", type=Path, default=Path("models/OCR/microcharnet.onnx"))
    parser.add_argument("--no-json-files", action="store_true", help="Return results in memory without saving JSON files.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    engine = AIPlateEngine(
        detector_model=args.detector_model,
        ocr_model=args.ocr_model,
        write_json=not args.no_json_files,
    )
    emit({"event": "starting"})
    try:
        with redirect_stdout(sys.stderr):
            engine.startup()
    except Exception as exc:
        logging.exception("Engine startup failed")
        emit({"event": "error", "error": {"code": "STARTUP_ERROR", "message": str(exc)[:300]}})
        return 1

    emit({"event": "ready"})
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
            except json.JSONDecodeError as exc:
                response = error_response(None, "INVALID_REQUEST", f"Invalid JSON: {exc.msg}")
            else:
                with redirect_stdout(sys.stderr):
                    response = engine.handle_request(request)
            emit(response)
            if engine.state == engine.SHUTTING_DOWN:
                break
    except KeyboardInterrupt:
        pass
    finally:
        engine.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
