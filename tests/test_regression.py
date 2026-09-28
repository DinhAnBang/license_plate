"""Behavior snapshots for the production algorithms and public result shape."""

import numpy as np
import pytest

from src.core.ocr import (
    MicroCharNetOCR, OCRCharacter, OCRResult, _RawCharacter, _Transform,
    _class_agnostic_nms, _group_and_sort_characters,
)
from src.image.customer_output import build_customer_payload
from src.core.ocr_fusion import OCRFusionCandidate, fuse_track
from src.core.plate_detector import PlateDetection
from src.image.plate_types import ImagePlateCandidate
from src.core.plate_geometry import bbox_iou, intersection_over_plate_area, local_bbox_to_global
from src.core.plate_quality import PlateQualityConfig, VideoPlateQualityConfig, score_plate_quality
from src.image.plate_ownership import TemporalPlateOwnershipResolver
from src.image.plate_stage import detect_vehicle_plates
from src.image.result_finalizer import finalize_vehicle
from src.image.result_serialization import serialize_vehicle_result
from src.core.plate_postprocess import postprocess_vietnam_plate
from src.core.vehicle_detector import VehicleDetection


def plate(vehicle_index=1, frame=0, box=(20, 60, 50, 75), confidence=0.9):
    return ImagePlateCandidate(
        frame_index=frame, vehicle_index=vehicle_index, vehicle_class_id=2,
        vehicle_class_name="car", vehicle_confidence=0.9,
        vehicle_bbox=(0, 0, 100, 100), plate_class_id=1,
        plate_class_name="dai", plate_confidence=confidence, plate_bbox=box,
    )


def test_geometry_and_ownership_snapshot():
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert intersection_over_plate_area((0, 0, 10, 10), (0, 0, 5, 10)) == 0.5
    assert [item.vehicle_index for item in TemporalPlateOwnershipResolver().resolve(0, [
        plate(1, confidence=0.8), plate(2, confidence=0.9),
    ]).candidates] == [2]
    assert TemporalPlateOwnershipResolver().resolve(0, [
        plate(1, box=(10, 60, 30, 75), confidence=0.8),
        plate(1, box=(60, 60, 80, 75), confidence=0.9),
    ]).candidates[0].plate_confidence == pytest.approx(0.9)


def test_shared_geometry_preserves_image_and_video_clipping_policies():
    local = (-5, 5, 100, 40)
    parent = (10, 10, 40, 30)
    assert local_bbox_to_global(local, parent, 100, 100) == (5, 15, 100, 50)
    assert local_bbox_to_global(local, parent, 100, 100, clip_to_vehicle=True) == (10, 15, 40, 30)


def test_shared_quality_preserves_image_and_video_focus_measurements():
    import cv2

    rng = np.random.default_rng(42)
    crop = rng.integers(0, 256, (15, 30, 3), dtype=np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    image = score_plate_quality(crop, 0.8, PlateQualityConfig())
    video = score_plate_quality(crop, 0.8, VideoPlateQualityConfig())
    resized = cv2.resize(gray, (128, 64), interpolation=cv2.INTER_CUBIC)
    assert image.sharpness_raw == pytest.approx(float(cv2.Laplacian(resized, cv2.CV_64F).var()))
    assert video.sharpness_raw == pytest.approx(float(cv2.Laplacian(gray, cv2.CV_64F).var()))
    assert image.sharpness_raw != video.sharpness_raw


def test_temporal_ownership_keeps_history_and_one_owner():
    resolver = TemporalPlateOwnershipResolver()
    first = resolver.resolve(0, [plate(1)])
    assert [item.vehicle_index for item in first.candidates] == [1]
    assert first.stats.history_updates == 1
    second = resolver.resolve(1, [plate(1, frame=1), plate(2, frame=1, confidence=0.1)])
    assert len(second.candidates) == 1
    assert second.stats.conflict_groups == 1
    assert second.stats.removed_conflict_candidates == 1


def test_temporal_ownership_rejects_single_candidate_that_jumps_from_history():
    resolver = TemporalPlateOwnershipResolver()
    for frame_index in range(5):
        accepted = resolver.resolve(frame_index, [plate(1, frame=frame_index)])
        assert len(accepted.candidates) == 1

    history_before = resolver.history[1]
    rejected = resolver.resolve(
        5, [plate(1, frame=5, box=(65, 10, 95, 25), confidence=0.99)],
    )

    assert rejected.candidates == ()
    assert rejected.stats.temporal_outliers_rejected == 1
    assert rejected.stats.history_updates == 0
    assert rejected.diagnostics[0].rejected_by_history is True
    assert resolver.history[1] == history_before


def test_temporal_ownership_accepts_small_motion_after_history_is_reliable():
    resolver = TemporalPlateOwnershipResolver()
    for frame_index in range(5):
        resolver.resolve(frame_index, [plate(1, frame=frame_index)])

    moved = resolver.resolve(5, [plate(1, frame=5, box=(23, 59, 53, 74))])

    assert len(moved.candidates) == 1
    assert moved.stats.temporal_outliers_rejected == 0


def test_temporal_ownership_assigns_nearby_vehicles_their_own_plates():
    """Same-track alternatives must not bridge two physical plate groups."""

    resolver = TemporalPlateOwnershipResolver()
    vehicle_boxes = {
        1: (0, 0, 200, 100),
        2: (40, 0, 240, 100),
    }
    own_plate_boxes = {
        1: (85, 60, 115, 75),
        2: (125, 60, 155, 75),
    }

    def candidate(vehicle_index, frame_index, plate_bbox, confidence=0.8):
        return ImagePlateCandidate(
            frame_index=frame_index,
            vehicle_index=vehicle_index,
            vehicle_class_id=2,
            vehicle_class_name="car",
            vehicle_confidence=0.9,
            vehicle_bbox=vehicle_boxes[vehicle_index],
            plate_class_id=1,
            plate_class_name="dai",
            plate_confidence=confidence,
            plate_bbox=plate_bbox,
        )

    # Establish reliable ownership history before the vehicles overlap.
    for frame_index in range(5):
        resolution = resolver.resolve(
            frame_index,
            [
                candidate(1, frame_index, own_plate_boxes[1]),
                candidate(2, frame_index, own_plate_boxes[2]),
            ],
        )
        assert len(resolution.candidates) == 2

    # Both vehicle ROIs now contain both physical plates. The wrong candidate
    # deliberately has higher detector confidence, so confidence alone loses.
    resolution = resolver.resolve(
        5,
        [
            candidate(1, 5, own_plate_boxes[1], confidence=0.75),
            candidate(1, 5, own_plate_boxes[2], confidence=0.99),
            candidate(2, 5, own_plate_boxes[1], confidence=0.99),
            candidate(2, 5, own_plate_boxes[2], confidence=0.75),
        ],
    )

    selected = {
        item.vehicle_index: item.plate_bbox for item in resolution.candidates
    }
    assert selected == own_plate_boxes
    assert resolution.stats.final_results == 2
    assert resolution.stats.removed_conflict_candidates == 2


def test_plate_ownership_selects_after_all_vehicle_candidates_are_adapted():
    class MultiplePlateDetector:
        def detect(self, _roi):
            return [
                PlateDetection(1, "dai", 0.60, (10, 50, 30, 60)),
                PlateDetection(0, "vuong", 0.95, (60, 50, 80, 60)),
            ]

    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    vehicle = VehicleDetection(2, "car", 0.9, (10, 10, 110, 90))
    candidates = detect_vehicle_plates(
        frame, vehicle, vehicle_index=7, plate_detector=MultiplePlateDetector(),
        frame_index=0,
    )

    assert len(candidates) == 2
    resolution = TemporalPlateOwnershipResolver().resolve(0, candidates)
    assert len(resolution.candidates) == 1
    assert resolution.candidates[0].vehicle_index == 7
    assert resolution.candidates[0].plate_confidence == pytest.approx(0.95)
    assert resolution.stats.conflict_groups == 1
    assert resolution.stats.removed_conflict_candidates == 1


def test_ocr_character_nms_snapshot():
    chars = [
        _RawCharacter("A", 0, 0.9, (0, 0, 10, 10)),
        _RawCharacter("B", 1, 0.8, (0, 0, 10, 10)),
    ]
    assert [item.char for item in _class_agnostic_nms(chars, 0.7)] == ["A"]
    assert OCRResult("51A12345", 0.8).text == "51A12345"


def test_ocr_end2end_decode_without_onnx_session():
    model = object.__new__(MicroCharNetOCR)
    model.conf_threshold = 0.25
    model.iou_threshold = 0.70
    model.num_classes = 2
    model.class_names = ("A", "B")
    transform = _Transform(1.0, 1.0, 0, 0, 100, 50)
    rows = np.asarray([[[20, 0, 30, 10, 0.8, 1], [0, 0, 10, 10, 0.9, 0]]], dtype=np.float32)
    result, characters = model._decode_end2end(rows, transform)
    assert result.text == "AB"
    assert result.confidence == pytest.approx(0.85)
    assert [character.char for character in characters] == ["A", "B"]


def test_ocr_end2end_suppresses_overlapping_class_hypotheses():
    model = object.__new__(MicroCharNetOCR)
    model.conf_threshold = 0.25
    model.iou_threshold = 0.70
    model.num_classes = 3
    model.class_names = ("C", "0", "V")
    transform = _Transform(1.0, 1.0, 0, 0, 100, 30)
    rows = np.asarray(
        [[
            [10, 5, 20, 25, 0.90, 0],
            [10, 5, 20, 25, 0.70, 1],
            [10, 6, 20, 25, 0.60, 0],
            [22, 5, 32, 25, 0.80, 2],
        ]],
        dtype=np.float32,
    )

    result, characters = model._decode_end2end(rows, transform)

    assert result.text == "CV"
    assert [character.char for character in characters] == ["C", "V"]
    assert result.char_confidences == pytest.approx((0.90, 0.80))


def test_ocr_groups_tilted_two_line_plate_before_sorting_columns():
    characters = [
        OCRCharacter("A", 0, 0.35, (0, 19, 8, 45)),
        OCRCharacter("6", 1, 0.79, (8, 8, 18, 24)),
        OCRCharacter("6", 1, 0.82, (12, 25, 22, 42)),
        OCRCharacter("6", 1, 0.81, (17, 6, 26, 23)),
        OCRCharacter("6", 1, 0.83, (23, 23, 33, 40)),
        OCRCharacter("H", 2, 0.53, (31, 3, 41, 21)),
        OCRCharacter("6", 1, 0.81, (33, 22, 43, 39)),
        OCRCharacter("9", 3, 0.78, (40, 3, 49, 19)),
        OCRCharacter("6", 1, 0.72, (43, 20, 53, 37)),
    ]

    lines = _group_and_sort_characters(characters)

    assert ["".join(item.char for item in line) for line in lines] == [
        "66H9", "A6666",
    ]


def test_fusion_postprocess_and_result_schema_snapshot():
    candidates = [
        OCRFusionCandidate(1, 1, 1, "51A12345", 0.9, 0.8),
        OCRFusionCandidate(1, 4, 2, "51A12345", 0.8, 0.7),
    ]
    fused = fuse_track(1, candidates)
    assert fused.raw_text == "51A12345"
    assert fused.method == "weighted_exact_vote"
    normalized = postprocess_vietnam_plate(fused.raw_text, fused.confidence)
    assert normalized.formatted_text == "51A-123.45"
    two_line = postprocess_vietnam_plate(
        "66h9a6666",
        0.71,
        preferred_family="motorbike_common",
        char_confidences=(0.79, 0.82, 0.53, 0.78, 0.35, 0.79, 0.83, 0.81, 0.72),
    )
    assert two_line.corrected_text == "66H96666"
    assert two_line.formatted_text == "66H9-66.66"
    assert [change.reason for change in two_line.corrections] == [
        "low_confidence_extra_character",
    ]
    result = finalize_vehicle(
        identity_key="vehicle_index", identity=1, vehicle_class_id=2,
        vehicle_class_name="car", first_frame=0, last_frame=4,
        vehicle_observation_count=5, plate_observation_count=2,
        plate_layout="dai", best_plate_bbox=(20, 60, 50, 75),
        best_plate_frame=1, postprocessed=normalized, fusion=fused,
        ocr_candidate_count=2,
    )
    payload = serialize_vehicle_result(result)
    assert set(payload) == {"vehicle_index", "vehicle", "plate", "evidence"}
    assert payload["plate"]["status"] == "ok"
    assert payload["plate"]["formatted"] == "51A-123.45"
    assert payload["evidence"]["support_count"] == 2
    customer = build_customer_payload({"status": "ok", "vehicles": [payload]})
    assert customer == {"status": "ok", "vehicles": [{
        "vehicle_index": 1, "vehicle_type": "car", "license": "51A-123.45",
        "confidence": payload["plate"]["confidence"], "status": "ok",
    }]}
    assert build_customer_payload(
        {
            "status": "ok",
            "vehicles": [
                payload,
                {
                    "track_id": 2,
                    "vehicle": {"class_name": "car"},
                    "plate": {
                        "status": "no_plate",
                        "confidence": 0.0,
                    },
                },
            ],
        },
        min_confidence=0.5,
    ) == customer
    assert build_customer_payload(
        {
            "status": "ok",
            "vehicles": [
                payload,
                {
                    "vehicle_index": 3,
                    "vehicle": {"class_name": "car"},
                    "plate": {
                        "status": "low_confidence",
                        "format_valid": True,
                        "formatted": "51A-123.45",
                        "confidence": 0.95,
                    },
                },
                {
                    "vehicle_index": 4,
                    "vehicle": {"class_name": "car"},
                    "plate": {
                        "status": "ok",
                        "format_valid": False,
                        "formatted": "51A-123.45",
                        "confidence": 0.95,
                    },
                },
                {
                    "vehicle_index": 5,
                    "vehicle": {"class_name": "car"},
                    "plate": {
                        "status": "no_plate",
                        "format_valid": False,
                        "confidence": 0.0,
                    },
                },
            ],
        },
        min_confidence=0.5,
    ) == customer
    assert finalize_vehicle(
        identity_key="vehicle_index", identity=2, vehicle_class_id=2,
        vehicle_class_name="car", first_frame=0, last_frame=1,
        vehicle_observation_count=2, plate_observation_count=0,
        plate_layout=None, best_plate_bbox=None, best_plate_frame=None,
        postprocessed=postprocess_vietnam_plate(""), fusion=None,
        ocr_candidate_count=0,
    ).status == "no_plate"
