"""Focused Phase 8-10 contract tests for voting and OCR failure isolation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np

from core.image_processor import ImageProcessor
from core.ocr_voter import OCRVoter
from core.plate_normalizer import PlateNormalizer
from core.quality import PlateQualityEvaluator
from core.result_writer import VideoResultWriter
from core.tracker import PlateTracker
from core.video_processor import VideoProcessor


class FakeDetector:
    def __init__(self, detections: bool = True) -> None:
        self.enabled = detections
        self.calls = 0

    def detect(self, image: np.ndarray) -> list[dict]:
        self.calls += 1
        return [{"conf": 0.9, "box": [20, 20, 100, 60]}] if self.enabled else []


class FailingOCR:
    session_creation_count = 1

    def __init__(self, detector: FakeDetector, expected_frames: int = 0) -> None:
        self.detector = detector
        self.expected_frames = expected_frames
        self.calls = 0

    def recognize(self, crop: np.ndarray) -> dict:
        if self.expected_frames:
            assert self.detector.calls == self.expected_frames, "Video OCR ran before tracking finished"
        self.calls += 1
        raise RuntimeError("synthetic OCR failure")


def test_voter() -> None:
    assert PlateNormalizer.normalize(" 72a-162.31 ") == "72A16231"
    assert PlateNormalizer.normalize("72O-162.31") == "72O16231"
    assert PlateNormalizer.normalize("\t59x1_12345\n") == "59X112345"

    candidates = [
        {"raw_text": "72a-16231", "text": "72A16231", "ocr_conf": 0.8, "quality": 0.8},
        {"raw_text": "72A16231", "text": "72A16231", "ocr_conf": 0.7, "quality": 0.8},
        {"raw_text": "72A16281", "text": "72A16281", "ocr_conf": 0.95, "quality": 0.9},
    ]
    vote = OCRVoter.vote(candidates)
    assert vote["text"] == "72A16231"
    assert vote["winner_index"] == 0
    assert vote["raw_text"] == "72a-16231" and vote["ocr_conf"] == 0.8
    assert OCRVoter.vote([{**candidates[0], "text": ""}])["winner_index"] is None

    # Equal group totals: prefer higher individual weight, then OCR confidence,
    # then quality, and finally input order.
    equal = [
        {"raw_text": "A", "text": "A", "ocr_conf": 0.5, "quality": 0.5},
        {"raw_text": "B", "text": "B", "ocr_conf": 0.5, "quality": 0.5},
    ]
    assert OCRVoter.vote(equal)["winner_index"] == 0

    group_total_tie = [
        {"raw_text": "A", "text": "A", "ocr_conf": 0.5, "quality": 0.5},
        {"raw_text": "A", "text": "A", "ocr_conf": 0.5, "quality": 0.5},
        {"raw_text": "B", "text": "B", "ocr_conf": 1.0, "quality": 1.0},
    ]
    assert OCRVoter.vote(group_total_tie)["winner_index"] == 2

    confidence_tie = [
        {"raw_text": "A", "text": "A", "ocr_conf": 0.5, "quality": 0.5},
        {"raw_text": "B", "text": "B", "ocr_conf": 0.6, "quality": 0.2666666667},
    ]
    assert OCRVoter.vote(confidence_tie)["winner_index"] == 1

    quality_tie = [
        {"raw_text": "A", "text": "A", "ocr_conf": 0.6, "quality": 0.6},
        {"raw_text": "A", "text": "A", "ocr_conf": 0.4, "quality": 0.4},
        {"raw_text": "B", "text": "B", "ocr_conf": 0.5, "quality": 0.8333333333},
        {"raw_text": "B", "text": "B", "ocr_conf": 0.5, "quality": 0.1666666667},
    ]
    assert OCRVoter.vote(quality_tie)["winner_index"] == 2


def test_processors() -> None:
    with tempfile.TemporaryDirectory(prefix="plate_ocr_pipeline_") as temp_dir:
        root = Path(temp_dir)
        input_dir = root / "input"
        input_dir.mkdir()
        image_path = input_dir / "test.jpg"
        image = np.full((100, 140, 3), 128, dtype=np.uint8)
        assert cv2.imwrite(str(image_path), image)

        image_detector = FakeDetector()
        image_ocr = FailingOCR(image_detector)
        image_processor = ImageProcessor(image_detector, project_root=root, ocr=image_ocr)
        image_result = image_processor.process(image_path)
        assert image_result["count"] == len(image_result["plates"]) == 1
        assert {key: image_result["plates"][0][key] for key in ("raw_text", "text", "ocr_conf")} == {
            "raw_text": "", "text": "", "ocr_conf": 0.0
        }
        assert image_ocr.calls == 1
        image_json = json.loads((root / "output/json/test.json").read_text(encoding="utf-8"))
        assert image_json == image_result

        empty_image = ImageProcessor(FakeDetector(False), project_root=root, ocr=image_ocr)
        assert empty_image.process(image_path)["plates"] == []

        video_path = input_dir / "test.mp4"
        writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (140, 100))
        assert writer.isOpened()
        for _ in range(4):
            writer.write(image)
        writer.release()

        video_detector = FakeDetector()
        video_ocr = FailingOCR(video_detector, expected_frames=4)
        processor = VideoProcessor(
            video_detector,
            project_root=root,
            tracker=PlateTracker(),
            quality_evaluator=PlateQualityEvaluator(),
            result_writer=VideoResultWriter(project_root=root),
            ocr=video_ocr,
            top_k=3,
        )
        video_result = processor.process(video_path)
        saved = json.loads((root / video_result["json"]).read_text(encoding="utf-8"))
        assert saved["count"] == len(saved["plates"]) == 1
        assert saved["plates"][0]["raw_text"] == saved["plates"][0]["text"] == ""
        assert saved["plates"][0]["ocr_conf"] == 0.0
        assert saved["plates"][0]["best_frame"] == 1
        assert video_ocr.calls == video_result["ocr_calls"] == 3
        assert video_result["ocr_report"][0]["candidates_retained"] == 3
        assert video_detector.calls == 4

        empty_detector = FakeDetector(False)
        empty_video = VideoProcessor(
            empty_detector,
            project_root=root,
            tracker=PlateTracker(),
            quality_evaluator=PlateQualityEvaluator(),
            result_writer=VideoResultWriter(project_root=root),
            ocr=FailingOCR(empty_detector),
        )
        assert empty_video.process(video_path)["best_crops"] == []
        empty_json = json.loads((root / "output/json/test.json").read_text(encoding="utf-8"))
        assert empty_json["count"] == 0 and empty_json["plates"] == []


if __name__ == "__main__":
    test_voter()
    test_processors()
    print("Phase 8-10 OCR integration contracts: OK")
