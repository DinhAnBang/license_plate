"""End-to-end image/video contract with deterministic in-memory model doubles."""

import cv2
import numpy as np

import src.image.pipeline as image_pipeline
import src.video.pipeline as video_pipeline
from src.alpr_pipeline import ALPRPipeline
from src.microcharnet_ocr import OCRResult
from src.plate_detector import PlateDetection
from src.tracking import ByteTracker
from src.vehicle_detector import VehicleDetection


class VehicleModel:
    session_init_count = 1
    total_inference_seconds = 0.0

    def detect(self, frame):
        return [VehicleDetection(2, "car", 0.95, (10, 10, 110, 90))]


class DuplicateVehicleModel(VehicleModel):
    def __init__(self):
        self._frame_index = 0

    def detect(self, frame):
        if self._frame_index == 0:
            detections = [
                VehicleDetection(2, "car", 0.95, (0, 0, 100, 100)),
                VehicleDetection(2, "car", 0.95, (120, 0, 220, 100)),
            ]
        else:
            detections = [
                VehicleDetection(2, "car", 0.95, (50, 0, 150, 100)),
                VehicleDetection(2, "car", 0.95, (55, 0, 155, 100)),
            ]
        self._frame_index += 1
        return detections


class PlateModel:
    session_init_count = 1
    total_inference_seconds = 0.0
    total_processing_seconds = 0.0
    detect_call_count = 0

    def detect(self, roi):
        self.detect_call_count += 1
        return [PlateDetection(1, "dai", 0.95, (30, 50, 70, 70))]


class NoPlateModel(PlateModel):
    def detect(self, roi):
        self.detect_call_count += 1
        return []


class OCRModel:
    session_init_count = 1
    inference_count = 0

    @property
    def timing_totals(self):
        return {"total_ms_per_crop": 0.0}

    def recognize(self, crop):
        self.inference_count += 1
        return OCRResult("51A12345", 0.95, tuple([0.95] * 8))


class LowConfidenceOCRModel(OCRModel):
    def recognize(self, crop):
        self.inference_count += 1
        return OCRResult("51A12345", 0.10, tuple([0.10] * 8))


def make_pipeline(*, vehicle_detector=None, plate_detector=None, ocr_engine=None):
    return ALPRPipeline(
        vehicle_detector=vehicle_detector or VehicleModel(),
        plate_detector=plate_detector or PlateModel(),
        ocr_engine=ocr_engine or OCRModel(), device="cpu",
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
        "detected_vehicles": 1, "vehicles": 1, "vehicles_with_plate": 1,
        "vehicles_with_ocr": 1, "successful_results": 1,
    }
    assert result["vehicles"][0]["vehicle_index"] == 0
    assert result["vehicles"][0]["plate"]["formatted"] == "51A-123.45"
    assert (tmp_path / "result.json").is_file()
    assert (tmp_path / "image_annotated.jpg").is_file()
    assert ("51A-123.45", (0, 0, 255)) in drawn_labels
    assert not any(label == "dai" for label, _ in drawn_labels)


def test_image_pipeline_preserves_no_plate_vehicle(tmp_path):
    source = tmp_path / "image.jpg"
    cv2.imwrite(str(source), np.full((100, 120, 3), 180, dtype=np.uint8))

    result = make_pipeline(plate_detector=NoPlateModel()).process_image(
        source, output=tmp_path / "result.json",
    )

    assert result["summary"] == {
        "detected_vehicles": 1,
        "vehicles": 1,
        "vehicles_with_plate": 0,
        "vehicles_with_ocr": 0,
        "successful_results": 0,
    }
    assert result["vehicles"][0]["plate"]["status"] == "no_plate"


def test_image_pipeline_preserves_low_confidence_vehicle(tmp_path):
    source = tmp_path / "image.jpg"
    cv2.imwrite(str(source), np.full((100, 120, 3), 180, dtype=np.uint8))

    result = make_pipeline(ocr_engine=LowConfidenceOCRModel()).process_image(
        source, output=tmp_path / "result.json",
    )

    assert result["summary"] == {
        "detected_vehicles": 1,
        "vehicles": 1,
        "vehicles_with_plate": 1,
        "vehicles_with_ocr": 1,
        "successful_results": 0,
    }
    assert result["vehicles"][0]["plate"]["status"] == "low_confidence"


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


def test_video_pipeline_preserves_no_plate_track(tmp_path):
    source = tmp_path / "video.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (120, 100),
    )
    assert writer.isOpened()
    for _ in range(3):
        writer.write(np.full((100, 120, 3), 180, dtype=np.uint8))
    writer.release()

    result = make_pipeline(plate_detector=NoPlateModel()).process_video(
        source, output=tmp_path / "result.json",
    )

    assert result["summary"] == {
        "detected_vehicles": 1,
        "vehicles": 1,
        "vehicles_with_plate": 0,
        "vehicles_with_ocr": 0,
        "successful_results": 0,
    }
    assert result["vehicles"][0]["plate"]["status"] == "no_plate"


def test_video_pipeline_emits_one_result_for_suppressed_duplicate_track(
    tmp_path, monkeypatch,
):
    source = tmp_path / "duplicate.mp4"
    writer = cv2.VideoWriter(
        str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (240, 100),
    )
    assert writer.isOpened()
    for _ in range(2):
        writer.write(np.full((100, 240, 3), 180, dtype=np.uint8))
    writer.release()

    tracker = ByteTracker(
        fps=30.0,
        min_confirmed_hits=1,
        cross_class_dedup_enabled=False,
        active_duplicate_suppression_enabled=True,
        active_duplicate_iou_threshold=0.40,
        active_duplicate_min_frames=1,
    )
    monkeypatch.setattr(
        video_pipeline,
        "create_video_tracker",
        lambda *_args, **_kwargs: tracker,
    )

    result = make_pipeline(
        vehicle_detector=DuplicateVehicleModel(),
    ).process_video(source, output=tmp_path / "result.json")

    assert len(tracker.all_tracks) == 2
    assert len(tracker.result_tracks) == 1
    assert result["summary"]["detected_vehicles"] == 1
    assert [row["track_id"] for row in result["vehicles"]] == [1]
