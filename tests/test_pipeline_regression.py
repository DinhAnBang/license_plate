"""End-to-end image/video contract with deterministic in-memory model doubles."""

import cv2
import numpy as np

import src.image_pipeline as image_pipeline
import src.video_pipeline as video_pipeline
from src.alpr_pipeline import ALPRPipeline
from src.microcharnet_ocr import OCRResult
from src.plate_detector import PlateDetection
from src.vehicle_detector import VehicleDetection


class VehicleModel:
    session_init_count = 1
    total_inference_seconds = 0.0

    def detect(self, frame):
        return [VehicleDetection(2, "car", 0.95, (10, 10, 110, 90))]


class PlateModel:
    session_init_count = 1
    total_inference_seconds = 0.0
    total_processing_seconds = 0.0
    detect_call_count = 0

    def detect(self, roi):
        self.detect_call_count += 1
        return [PlateDetection(1, "dai", 0.95, (30, 50, 70, 70))]


class OCRModel:
    session_init_count = 1
    inference_count = 0

    @property
    def timing_totals(self):
        return {"total_ms_per_crop": 0.0}

    def recognize(self, crop):
        self.inference_count += 1
        return OCRResult("51A12345", 0.95, tuple([0.95] * 8))


def make_pipeline():
    return ALPRPipeline(
        vehicle_detector=VehicleModel(), plate_detector=PlateModel(),
        ocr_engine=OCRModel(), device="cpu",
    )


def test_image_pipeline_json_contract(tmp_path, monkeypatch):
    source = tmp_path / "image.jpg"
    cv2.imwrite(str(source), np.full((100, 120, 3), 180, dtype=np.uint8))
    drawn_labels = []
    draw_box = image_pipeline._draw_box

    def capture_draw(frame, bbox, label, color):
        drawn_labels.append((label, color))
        draw_box(frame, bbox, label, color)

    monkeypatch.setattr(image_pipeline, "_draw_box", capture_draw)
    result = make_pipeline().process_image(source, output=tmp_path / "result.json", save_annotated=True)
    assert result["summary"] == {
        "vehicles": 1, "vehicles_with_plate": 1,
        "vehicles_with_ocr": 1, "successful_results": 1,
    }
    assert result["vehicles"][0]["vehicle_index"] == 0
    assert result["vehicles"][0]["plate"]["formatted"] == "51A-123.45"
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "image_annotated.jpg").is_file()
    assert ("51A-123.45", (0, 0, 255)) in drawn_labels
    assert not any(label == "dai" for label, _ in drawn_labels)


def test_video_pipeline_json_contract(tmp_path, monkeypatch):
    source = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (120, 100))
    assert writer.isOpened()
    for _ in range(5):
        writer.write(np.full((100, 120, 3), 180, dtype=np.uint8))
    writer.release()

    drawn_labels = []
    draw_box = video_pipeline._draw_box

    def capture_draw(frame, bbox, label, color):
        drawn_labels.append((label, color))
        draw_box(frame, bbox, label, color)

    monkeypatch.setattr(video_pipeline, "_draw_box", capture_draw)
    model = make_pipeline()
    result = model.process_video(source, output=tmp_path / "result.json", save_annotated=True)
    assert result["input"]["frames"] == 5
    assert len(result["vehicles"]) == 1
    assert result["vehicles"][0]["track_id"] == 1
    assert result["vehicles"][0]["plate"]["formatted"] == "51A-123.45"
    assert result["performance"]["ocr_inference_calls"] <= 5
    assert model.plate_detector.detect_call_count == 4
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "video_annotated.mp4").is_file()
    assert ("51A-123.45", (0, 0, 255)) in drawn_labels
    assert not any(label == "dai" for label, _ in drawn_labels)
