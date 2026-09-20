"""Persistent source sessions remain shared when a BYTE helper is exercised."""

from __future__ import annotations

import tempfile
from pathlib import Path

from core.byte_tracker import ByteTracker
from core.quality import PlateQualityEvaluator
from core.video_processor import VideoProcessor
from engine import AIPlateEngine
from tests.test_engine import FakeDetector, FakeOCR, make_inputs


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="plate_byte_engine_") as temp_dir:
        root = Path(temp_dir)
        make_inputs(root)
        detector = FakeDetector()
        ocr = FakeOCR()
        engine = AIPlateEngine(
            project_root=root,
            detector=detector,
            ocr=ocr,
            write_json=False,
        )
        engine.startup()
        assert engine.tracker_mode == "sort"
        assert engine.handle_request(
            {"id": "image-before", "action": "process", "type": "image", "path": "input/a.jpg"}
        )["status"] == "success"

        detector_session = id(detector.session)
        ocr_session = id(ocr.session)
        byte_processor = VideoProcessor(
            detector,
            project_root=root,
            output_dir=root / "byte-output",
            tracker_mode="byte",
            quality_evaluator=PlateQualityEvaluator(),
            ocr=ocr,
        )
        first = byte_processor.process(root / "input/v.mp4")
        assert isinstance(byte_processor.tracker, ByteTracker)
        assert first["tracks"][0]["track_id"] == 1
        assert engine.handle_request(
            {"id": "sort-video", "action": "process", "type": "video", "path": "input/v.mp4"}
        )["status"] == "success"
        assert engine.handle_request(
            {"id": "image-after", "action": "process", "type": "image", "path": "input/b.jpg"}
        )["status"] == "success"
        second = byte_processor.process(root / "input/v.mp4")
        assert second["tracks"][0]["track_id"] == 1
        assert byte_processor.tracker.created_tracks == 1

        assert id(detector.session) == detector_session
        assert id(ocr.session) == ocr_session
        assert engine.image_processor.detector is engine.video_processor.detector is detector
        assert engine.image_processor.ocr is engine.video_processor.ocr is ocr
        engine.shutdown()
    print("SORT/BYTE helper persistent detector and OCR session lifetime: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
