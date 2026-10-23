"""The long-lived protocol must handle more than one input per process."""

from __future__ import annotations

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src import engine


def test_engine_reuses_pipeline_for_image_video_and_shutdown(tmp_path, monkeypatch):
    image = tmp_path / "one.jpg"
    video = tmp_path / "two.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    instances = []

    class FakePipeline:
        def __init__(self, *, device, debug=False):
            self.config = SimpleNamespace(low_confidence_threshold=0.5)
            self.calls = []
            instances.append(self)

        def process_image(self, path: Path, **kwargs):
            self.calls.append(("image", path, kwargs))
            return _result("image", kwargs["output"])

        def process_video(self, path: Path, **kwargs):
            self.calls.append(("video", path, kwargs))
            return _result("video", kwargs["output"])

    monkeypatch.setattr(engine, "ALPRPipeline", FakePipeline)
    monkeypatch.setattr(engine, "application_directory", lambda: tmp_path)
    requests = (
        {"id": "a", "action": "process", "type": "image", "path": str(image)},
        {"id": "b", "action": "process", "type": "video", "path": str(video)},
        {"id": "c", "action": "process", "type": "image", "path": str(tmp_path / "missing.jpg")},
        {"id": "stop", "action": "shutdown"},
    )
    incoming = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    outgoing = io.StringIO()

    assert engine.run_input_stream(input_stream=incoming, output_stream=outgoing) == 0
    lines = [json.loads(line) for line in outgoing.getvalue().splitlines()]
    assert [item.get("event") for item in lines[:2]] == ["starting", "ready"]
    assert [item["request_id"] for item in lines[2:]] == ["a", "b", "c", "stop"]
    assert [item["status"] for item in lines[2:]] == ["success", "success", "error", "success"]
    assert lines[2]["plates"][0]["plate_text"] == "51A-123.45"
    assert lines[3]["input_type"] == "video"
    assert lines[3]["output_video"].endswith(".mp4")
    assert len(instances) == 1
    assert [call[0] for call in instances[0].calls] == ["image", "video"]
    assert Path(lines[2]["output_json"]).parent.parent == tmp_path
    assert Path(lines[3]["output_video"]).parent.parent == tmp_path


def test_input_path_keeps_loaded_pipeline_for_later_requests(tmp_path, monkeypatch):
    import main

    image = tmp_path / "first.jpg"
    video = tmp_path / "second.mp4"
    image.write_bytes(b"image")
    video.write_bytes(b"video")
    instances = []

    class FakePipeline:
        def __init__(self, **_kwargs):
            self.config = SimpleNamespace(low_confidence_threshold=0.5)
            self.calls = []
            instances.append(self)

        def process_image(self, path: Path, **kwargs):
            self.calls.append(("image", path))
            result = _result("image", kwargs["output"])
            result["input"] = {"type": "image"}
            result["summary"] = {
                "vehicles": 1, "vehicles_with_plate": 1,
                "vehicles_with_ocr": 1, "successful_results": 1,
            }
            return result

        def process_video(self, path: Path, **kwargs):
            self.calls.append(("video", path))
            return _result("video", kwargs["output"])

    requests = (
        {"id": "later", "action": "process", "type": "video", "path": str(video)},
        {"id": "stop", "action": "shutdown"},
    )
    incoming = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    outgoing = io.StringIO()
    monkeypatch.setattr(main, "ALPRPipeline", FakePipeline)
    monkeypatch.setattr(engine, "application_directory", lambda: tmp_path)
    monkeypatch.setattr(main.sys, "stdin", incoming)
    monkeypatch.setattr(main.sys, "stdout", outgoing)

    assert main.main(["--input", str(image), "--output", str(tmp_path / "first.json")]) == 0
    assert len(instances) == 1
    assert instances[0].calls == [("image", image), ("video", video)]
    protocol = [json.loads(line) for line in outgoing.getvalue().splitlines() if line.startswith("{")]
    assert [item.get("event") for item in protocol[:2]] == ["starting", "ready"]
    assert protocol[2]["request_id"] == "later"
    assert protocol[2]["status"] == "success"
    assert protocol[3]["action"] == "shutdown"


def test_input_without_path_enters_request_loop(monkeypatch):
    import main

    seen = []
    monkeypatch.setattr(engine, "run_input_stream", lambda **kwargs: seen.append(kwargs) or 0)
    assert main.main(["--input"]) == 0
    assert seen == [{"device": "auto", "debug": False}]
    with pytest.raises(SystemExit):
        main.parse_args(["--serve"])


def _result(media_type: str, output: Path) -> dict:
    return {
        "vehicles": [{"plate": {
            "status": "ok", "format_valid": True,
            "formatted": "51A-123.45", "confidence": 0.9,
        }}],
        "annotated_path": str(output.with_suffix(".jpg" if media_type == "image" else ".mp4")),
        "output_path": str(output),
    }
