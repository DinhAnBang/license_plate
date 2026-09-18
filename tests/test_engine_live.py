"""End-to-end test of the real persistent engine and JSON Lines transport."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

from engine import AIPlateEngine


ROOT = Path(__file__).resolve().parents[1]


def _check_outputs(response: dict, media_type: str) -> None:
    result = response["result"]
    assert result["status"] == "ok" and result["type"] == media_type
    assert result["count"] == len(result["plates"])
    source_stem = Path(result["source"]).stem
    saved_json = ROOT / "output" / "json" / f"{source_stem}.json"
    assert json.loads(saved_json.read_text(encoding="utf-8")) == result

    for plate in result["plates"]:
        assert (ROOT / plate["crop"]).is_file()
        assert {"raw_text", "text", "ocr_conf"}.issubset(plate)
    if media_type == "image":
        assert (ROOT / "output" / "images" / f"{source_stem}_result.jpg").is_file()
    else:
        assert (ROOT / "output" / "videos" / f"{source_stem}_tracked.mp4").is_file()
        assert all("candidates" not in item and "vote_weight" not in item for item in result["plates"])


def main() -> int:
    detector_model = ROOT / "models" / "best.onnx"
    ocr_model = ROOT / "models" / "OCR" / "microcharnet.onnx"
    if not detector_model.is_file() or not ocr_model.is_file():
        print("RUNTIME TEST BLOCKED: required ONNX model is missing")
        return 2

    image_path = ROOT / "input" / "images1.jpg"
    video_paths = sorted((ROOT / "input").glob("*.mp4"))
    assert image_path.is_file()
    assert video_paths

    requests = [
        {"id": "001", "action": "ping"},
        {"id": "002", "action": "process", "type": "image", "path": str(image_path)},
        {"id": "003", "action": "process", "type": "image", "path": str(image_path)},
        {"id": "004", "action": "process", "type": "image", "path": str(image_path)},
        {"id": "005", "action": "unknown"},
        {"id": "006", "action": "process", "type": "image", "path": str(ROOT / "input" / "missing.jpg")},
        {"id": "007", "action": "process", "type": "video", "path": str(video_paths[0])},
        {"id": "008", "action": "ping"},
        {"id": "009", "action": "shutdown"},
    ]

    started = time.perf_counter()
    process = subprocess.Popen(
        [sys.executable, str(ROOT / "main.py")],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert process.stdin is not None and process.stdout is not None and process.stderr is not None
    try:
        starting = json.loads(process.stdout.readline())
        ready = json.loads(process.stdout.readline())
        startup_ms = (time.perf_counter() - started) * 1000.0
        assert starting == {"event": "starting"}
        assert ready == {"event": "ready"}

        responses: list[dict] = []
        times_ms: dict[str, float] = {}
        for request in requests:
            request_started = time.perf_counter()
            process.stdin.write(json.dumps(request) + "\n")
            process.stdin.flush()
            response = json.loads(process.stdout.readline())
            times_ms[request["id"]] = (time.perf_counter() - request_started) * 1000.0
            responses.append(response)

        process.stdin.close()
        assert process.wait(timeout=10) == 0
        assert process.stdout.read() == ""
        stderr_text = process.stderr.read()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        process.stdout.close()
        process.stderr.close()

    assert [item["id"] for item in responses] == [item["id"] for item in requests]
    assert responses[4]["error"]["code"] == "UNSUPPORTED_ACTION"
    assert responses[5]["error"]["code"] == "INPUT_NOT_FOUND"
    assert responses[6]["status"] == responses[7]["status"] == "ok"
    assert responses[8]["state"] == "shutting_down"
    for response in responses[1:4]:
        _check_outputs(response, "image")
    _check_outputs(responses[6], "video")
    assert "Processing:" in stderr_text  # human-readable progress stayed off stdout

    engine = AIPlateEngine(project_root=ROOT, write_json=False)
    with redirect_stdout(sys.stderr):
        engine.startup()
    detector_session_id = id(engine.detector.session)
    ocr_session_id = id(engine.ocr.session)
    assert engine.detector_warmed and engine.ocr_warmed
    for index in range(3):
        response = engine.handle_request({
            "id": f"direct-{index}",
            "action": "process",
            "type": "image",
            "path": str(image_path),
        })
        assert response["status"] == "ok"
    with redirect_stdout(sys.stderr):
        video_response = engine.handle_request({
            "id": "direct-video",
            "action": "process",
            "type": "video",
            "path": str(video_paths[0]),
        })
    assert video_response["status"] == "ok"
    assert id(engine.detector.session) == detector_session_id
    assert id(engine.ocr.session) == ocr_session_id
    assert engine.image_processor.detector is engine.video_processor.detector is engine.detector
    assert engine.image_processor.ocr is engine.video_processor.ocr is engine.ocr
    engine.shutdown()

    image_times = [times_ms[key] for key in ("002", "003", "004")]
    print(f"Startup: {startup_ms:.2f} ms")
    for index, elapsed in enumerate(image_times, start=1):
        print(f"Image request {index}: {elapsed:.2f} ms")
    print(f"Average after READY: {sum(image_times) / len(image_times):.2f} ms")
    print(f"Video request: {times_ms['007']:.2f} ms")
    print(f"FIFO: {[item['id'] for item in responses]}")
    print(f"Image count: {responses[1]['result']['count']}")
    print(f"Video count: {responses[6]['result']['count']}")
    print("Detector sessions: 1; OCR sessions: 1; both unchanged across requests")
    print("Live JSON Lines, outputs, and model lifetime: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
