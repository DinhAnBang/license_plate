"""End-to-end image contract with deterministic in-memory model doubles."""

from dataclasses import replace

import cv2
import numpy as np

import src.image.pipeline as image_pipeline
from src.alpr_pipeline import ALPRPipeline
from src.config import PipelineConfig
from src.core.ocr import OCRResult
from src.core.plate_detector import PlateDetection
from src.core.vehicle_detector import VehicleDetection


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


def test_image_quality_and_ownership_config_reach_production(tmp_path, monkeypatch):
    source = tmp_path / "image.jpg"
    cv2.imwrite(str(source), np.full((100, 120, 3), 180, dtype=np.uint8))
    base = PipelineConfig()
    image = replace(
        base.image,
        quality=replace(base.image.quality, target_plate_height=96),
        ownership=replace(base.image.ownership, history_size=21),
    )
    config = replace(base, image=image)
    seen = {}
    original_score = image_pipeline.score_plate_quality
    original_resolver = image_pipeline.TemporalPlateOwnershipResolver

    def capture_score(crop, confidence, quality_config):
        seen["quality"] = quality_config
        return original_score(crop, confidence, quality_config)

    def capture_resolver(ownership_config):
        seen["ownership"] = ownership_config
        return original_resolver(ownership_config)

    monkeypatch.setattr(image_pipeline, "score_plate_quality", capture_score)
    monkeypatch.setattr(image_pipeline, "TemporalPlateOwnershipResolver", capture_resolver)
    pipeline = ALPRPipeline(
        config=config, vehicle_detector=VehicleModel(), plate_detector=PlateModel(),
        ocr_engine=OCRModel(), device="cpu",
    )
    pipeline.process_image(source, output=tmp_path / "result.json")
    assert seen == {"quality": image.quality, "ownership": image.ownership}
