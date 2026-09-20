"""Request-scoped output collision, reuse, validation, and cleanup tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from engine import AIPlateEngine
from tests.test_engine import FakeDetector, FakeOCR, make_inputs


class VariableDetector(FakeDetector):
    count = 1

    def detect(self, image: np.ndarray) -> list[dict]:
        self.detect_calls += 1
        return [
            {"conf": 0.9, "box": [10 + index * 35, 20, 40 + index * 35, 60]}
            for index in range(self.count)
        ]


def process(engine: AIPlateEngine, request_id: str, media_type: str, path: Path) -> dict:
    response = engine.handle_request({
        "id": request_id, "action": "process", "type": media_type, "path": str(path)
    })
    assert response["status"] == "success", response
    assert response["request_id"] == request_id
    return response


def check_saved(root: Path, request_id: str, result: dict) -> Path:
    request_dir = root / "output" / "requests" / request_id
    assert json.loads((request_dir / "result.json").read_text(encoding="utf-8")) == result
    assert all(Path(plate["crop_path"]).is_file() for plate in result["plates"])
    assert all(Path(plate["crop_path"]).is_absolute() for plate in result["plates"])
    return request_dir


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="plate_request_output_") as temp_dir:
        root = Path(temp_dir)
        make_inputs(root)
        camera_in = root / "camera_in"
        camera_out = root / "camera_out"
        camera_in.mkdir()
        camera_out.mkdir()
        source_a = camera_in / "frame001.jpg"
        source_b = camera_out / "frame001.jpg"
        assert cv2.imwrite(str(source_a), np.full((100, 140, 3), 90, dtype=np.uint8))
        assert cv2.imwrite(str(source_b), np.full((100, 140, 3), 220, dtype=np.uint8))
        video = root / "input" / "v.mp4"

        detector = VariableDetector()
        ocr = FakeOCR()
        engine = AIPlateEngine(project_root=root, detector=detector, ocr=ocr)
        engine.startup()
        assert detector.session.runs == ocr.session.runs == 1

        for bad_id in ("../test", "abc/def", "abc\\def", "C:", ".", "..", "bad*id", "bad?id", 'bad"id', "bad<id", "bad>id", "bad|id", "CON", "lpt1", "a" * 81, "", "é", 123):
            response = engine.handle_request({
                "id": bad_id, "action": "process", "type": "image", "path": str(source_a)
            })
            assert response["error"]["code"] == "INVALID_REQUEST_ID"
            assert not (root / "output" / "requests").exists()
        assert engine.state == engine.READY

        detector.count = 3
        first = process(engine, "req-A", "image", source_a)
        assert first["count"] == 3
        a_dir = check_saved(root, "req-A", first)
        a_annotated = (a_dir / "annotated.jpg").read_bytes()
        a_crop = (a_dir / "crops" / "plate_001.jpg").read_bytes()

        detector.count = 1
        second = process(engine, "req-B", "image", source_b)
        b_dir = check_saved(root, "req-B", second)
        assert a_dir != b_dir and (a_dir / "annotated.jpg").read_bytes() == a_annotated
        assert (a_dir / "crops" / "plate_001.jpg").read_bytes() == a_crop
        assert (b_dir / "annotated.jpg").read_bytes() != a_annotated

        repeated = process(engine, "req-A", "image", source_a)
        assert repeated["count"] == 1
        assert [path.name for path in (a_dir / "crops").iterdir()] == ["plate_001.jpg"]
        assert not (a_dir / "crops" / "plate_002.jpg").exists()
        assert (b_dir / "annotated.jpg").is_file()
        check_saved(root, "req-A", repeated)

        image_before = process(engine, "req-001", "image", source_a)
        video_result = process(engine, "req-002", "video", video)
        image_after = process(engine, "req-003", "image", source_b)
        for request_id, result, annotated in (
            ("req-001", image_before, "annotated.jpg"),
            ("req-002", video_result, "annotated.mp4"),
            ("req-003", image_after, "annotated.jpg"),
        ):
            request_dir = check_saved(root, request_id, result)
            assert (request_dir / annotated).is_file()
        assert video_result["count"] == 1
        assert video_result["plates"][0]["crop_path"].endswith("track_0001.jpg")

        detector.count = 0
        empty_image = process(engine, "empty-image", "image", source_a)
        empty_video = process(engine, "empty-video", "video", video)
        for request_id, result, annotated in (
            ("empty-image", empty_image, "annotated.jpg"),
            ("empty-video", empty_video, "annotated.mp4"),
        ):
            request_dir = check_saved(root, request_id, result)
            assert result["count"] == 0 and result["plates"] == []
            assert (request_dir / annotated).is_file()
            assert list((request_dir / "crops").iterdir()) == []

        # Reusing an id for another media type must remove its old artifacts.
        process(engine, "req-switch", "image", source_a)
        process(engine, "req-switch", "video", video)
        assert not (root / "output/requests/req-switch/annotated.jpg").exists()
        assert (root / "output/requests/req-switch/annotated.mp4").is_file()

        detector.count = 1
        original_imwrite = cv2.imwrite

        def fail_annotated(path: str, image: np.ndarray) -> bool:
            return False if str(path).endswith("annotated.jpg") else original_imwrite(path, image)

        with patch("core.image_processor.cv2.imwrite", side_effect=fail_annotated):
            failed = engine.handle_request({
                "id": "fail-partial", "action": "process", "type": "image", "path": str(source_a)
            })
        assert failed["error"]["code"] == "PROCESSING_ERROR"
        assert not (root / "output/requests/fail-partial").exists()
        assert process(engine, "after-failure", "image", source_b)["count"] == 1

        assert detector.session.runs == ocr.session.runs == 1
        assert engine.image_processor.detector is engine.video_processor.detector is detector
        assert engine.image_processor.ocr is engine.video_processor.ocr is ocr
        engine.shutdown()

    print("Request namespace collision, reuse, mixed media, invalid ids, no detection, and cleanup: OK")
    print("Detector sessions: 1; OCR sessions: 1; no model reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
