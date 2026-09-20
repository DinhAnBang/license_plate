"""Persistent engine lifecycle and JSON Lines transport contract tests."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import main as engine_main
from engine import AIPlateEngine


class FakeSession:
    def __init__(self) -> None:
        self.runs = 0

    def run(self, names: list[str], feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.runs += 1
        assert len(names) == 1 and len(feed) == 1
        return [np.zeros((1, 1), dtype=np.float32)]


class FakeDetector:
    input_shape = [1, 3, 640, 640]
    input_type = "tensor(float)"
    input_name = "images"
    output_info = [{"name": "output0", "shape": [1, 5, 8400], "type": "tensor(float)"}]

    def __init__(self) -> None:
        self.session = FakeSession()
        self.conf_threshold = 0.5
        self.detect_calls = 0
        self.fail_once = False

    def detect(self, image: np.ndarray, *, conf_threshold: float | None = None) -> list[dict]:
        self.detect_calls += 1
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic detector failure")
        return [{"conf": 0.9, "box": [20, 20, 100, 60]}]

    def detect_with_stats(
        self, image: np.ndarray, *, conf_threshold: float | None = None
    ) -> tuple[list[dict], dict[str, int]]:
        detections = self.detect(image, conf_threshold=conf_threshold)
        return detections, {
            "raw_detector_candidates": 1,
            "detections_after_threshold": len(detections),
            "detections_after_nms": len(detections),
        }


class FakeOCR:
    input_height = 128
    input_width = 128
    input_name = "images"
    output_name = "output0"
    session_creation_count = 1

    def __init__(self) -> None:
        self.session = FakeSession()
        self.recognize_calls = 0

    def recognize(self, crop: np.ndarray) -> dict:
        self.recognize_calls += 1
        return {"text": "72a-162.31", "confidence": 0.8}


def make_inputs(root: Path) -> None:
    input_dir = root / "input"
    input_dir.mkdir()
    image = np.full((100, 140, 3), 128, dtype=np.uint8)
    assert cv2.imwrite(str(input_dir / "a.jpg"), image)
    assert cv2.imwrite(str(input_dir / "b.jpg"), image)
    video = cv2.VideoWriter(
        str(input_dir / "v.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (140, 100)
    )
    assert video.isOpened()
    for _ in range(4):
        video.write(image)
    video.release()


def test_shared_lifetime(root: Path) -> None:
    detector = FakeDetector()
    ocr = FakeOCR()
    engine = AIPlateEngine(project_root=root, detector=detector, ocr=ocr, write_json=False)
    assert engine.handle_request({"id": "pre", "action": "ping"})["error"]["code"] == "ENGINE_NOT_READY"
    engine.startup()
    assert engine.state == engine.READY
    assert engine.detector_warmed and engine.ocr_warmed
    assert detector.session.runs == ocr.session.runs == 1
    assert engine.image_processor.detector is engine.video_processor.detector is detector
    assert engine.image_processor.ocr is engine.video_processor.ocr is ocr

    detector.fail_once = True
    failed = engine.handle_request({"id": "fail", "action": "process", "type": "image", "path": "input/a.jpg"})
    assert failed["error"]["code"] == "PROCESSING_ERROR"
    assert engine.state == engine.READY
    good = engine.handle_request({"id": "good", "action": "process", "type": "image", "path": "input/a.jpg"})
    assert good["plates"][0]["plate_text"] == "72A16231"
    assert not (root / "output/requests/good/result.json").exists()
    assert not (root / "output/requests/fail").exists()
    engine.shutdown()
    assert engine.state == engine.STOPPED


def test_json_lines(root: Path) -> None:
    detector = FakeDetector()
    ocr = FakeOCR()
    engine = AIPlateEngine(project_root=root, detector=detector, ocr=ocr)
    requests = [
        {"id": "001", "action": "ping"},
        {"id": "002", "action": "process", "type": "image", "path": "input/a.jpg"},
        {"id": "003", "action": "process", "type": "image", "path": "input/b.jpg"},
        {"id": "004", "action": "unknown"},
        {"id": "004a", "action": "process", "type": "image"},
        {"id": "004b", "action": "process", "type": "audio", "path": "input/a.jpg"},
        {"id": "../escape", "action": "process", "type": "image", "path": "input/a.jpg"},
        {"id": "005", "action": "process", "type": "image", "path": "input/a.jpg"},
        {"id": "006", "action": "process", "type": "image", "path": "input/missing.jpg"},
        {"id": "007", "action": "process", "type": "image", "path": "input/b.jpg"},
        {"id": "008", "action": "process", "type": "video", "path": "input/v.mp4"},
        {"id": "009", "action": "ping"},
        {"id": "010", "action": "shutdown"},
    ]
    lines = [json.dumps(item) for item in requests]
    lines.insert(7, "{broken")
    stdin = io.StringIO("\n".join(lines) + "\n")
    stdout = io.StringIO()
    stderr = io.StringIO()

    with (
        patch.object(sys, "stdin", stdin),
        patch.object(sys, "stdout", stdout),
        patch.object(sys, "stderr", stderr),
        patch.object(sys, "argv", ["main.py"]),
        patch.object(engine_main, "AIPlateEngine", return_value=engine),
    ):
        assert engine_main.main() == 0

    lines = stdout.getvalue().splitlines()
    objects = [json.loads(line) for line in lines]
    assert objects[:2] == [{"event": "starting"}, {"event": "ready"}]
    responses = objects[2:]
    expected_ids = [item["id"] for item in requests]
    expected_ids.insert(7, None)
    def response_id(item: dict) -> str | None:
        return item.get("request_id", item.get("id"))

    assert [response_id(item) for item in responses] == expected_ids
    by_id = {response_id(item): item for item in responses}
    assert by_id["004"]["error"]["code"] == "UNSUPPORTED_ACTION"
    assert by_id["004a"]["error"]["code"] == "INVALID_REQUEST"
    assert by_id["004b"]["error"]["code"] == "UNSUPPORTED_TYPE"
    assert by_id["../escape"]["error"]["code"] == "INVALID_REQUEST_ID"
    assert by_id[None]["error"]["code"] == "INVALID_REQUEST"
    assert by_id["006"]["error"]["code"] == "INPUT_NOT_FOUND"
    assert by_id["005"]["status"] == "success"
    assert by_id["007"]["status"] == "success"
    assert by_id["009"]["status"] == "ok"
    assert by_id["010"] == {"id": "010", "status": "ok", "state": "shutting_down"}
    assert engine.state == engine.STOPPED
    assert detector.session.runs == ocr.session.runs == 1
    assert ocr.recognize_calls == 4 + 3  # four images plus video Top-3

    image_result = by_id["002"]
    video_result = by_id["008"]
    assert image_result["input_type"] == "image" and image_result["count"] == 1
    assert video_result["input_type"] == "video" and video_result["count"] == 1
    assert image_result["request_id"] == "002" and video_result["request_id"] == "008"
    assert set(image_result) == {
        "status", "request_id", "input_type", "output_image", "processing", "count", "plates"
    }
    assert set(video_result) == {
        "status", "request_id", "input_type", "output_video", "processing", "count", "plates"
    }
    assert "raw_text" not in image_result["plates"][0]
    assert "track_id" not in video_result["plates"][0]
    assert (root / "output/requests/002/annotated.jpg").is_file()
    assert Path(image_result["output_image"]).is_file()
    assert Path(image_result["plates"][0]["crop_path"]).is_file()
    assert (root / "output/requests/008/annotated.mp4").is_file()
    assert Path(video_result["output_video"]).is_file()
    assert Path(video_result["plates"][0]["crop_path"]).is_file()
    for request_id, result in (("002", image_result), ("008", video_result)):
        saved = json.loads((root / f"output/requests/{request_id}/result.json").read_text(encoding="utf-8"))
        assert saved == result
    assert not (root / "output/escape").exists()


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="plate_engine_test_") as temp_dir:
        root = Path(temp_dir)
        make_inputs(root)
        test_shared_lifetime(root)
        test_json_lines(root)
    print("Engine lifecycle, FIFO, JSON Lines, sessions, and outputs: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
