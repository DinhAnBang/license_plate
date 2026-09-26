"""Behavior snapshots for the production algorithms and public result shape."""

import numpy as np
import pytest

from src.microcharnet_ocr import (
    MicroCharNetOCR, OCRCharacter, OCRResult, _RawCharacter, _Transform,
    _class_agnostic_nms, _group_and_sort_characters,
)
from src.customer_output import build_customer_payload
from src.ocr_fusion import OCRFusionCandidate, fuse_track
from src.ocr_serialization import build_fused_json, build_ocr_json
from src.ocr_stage import OCRPlateCandidate
from src.plate_buffer import BufferedPlateCandidate, PlateBufferConfig, PlateBufferManager
from src.plate_detector import PlateDetection
from src.plate_types import TrackedPlateCandidate
from src.geometry import bbox_iou, intersection_over_plate_area
from tools.diagnostics.legacy_plate_ownership import resolve_plate_ownership
from src.plate_ownership_temporal import TemporalPlateOwnershipResolver
from src.plate_quality import PlateQualityMetrics, crop_plate_from_frame, score_plate_quality
from src.plate_stage import detect_vehicle_plates
from src.result_finalizer import finalize_vehicle
from src.result_serialization import serialize_vehicle_result
from src.vn_plate_postprocessor import postprocess_vietnam_plate
from src.tracking import ByteTracker, TrackRemovalReason
from src.vehicle_detector import VehicleDetection


def plate(track_id=1, frame=0, box=(20, 60, 50, 75), confidence=0.9):
    return TrackedPlateCandidate(
        frame_index=frame, track_id=track_id, vehicle_class_id=2,
        vehicle_class_name="car", vehicle_confidence=0.9,
        vehicle_bbox=(0, 0, 100, 100), plate_class_id=1,
        plate_class_name="dai", plate_confidence=confidence, plate_bbox=box,
    )


def test_geometry_and_ownership_snapshot():
    assert bbox_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)
    assert intersection_over_plate_area((0, 0, 10, 10), (0, 0, 5, 10)) == 0.5
    assert [item.track_id for item in resolve_plate_ownership([
        plate(1, confidence=0.8), plate(2, confidence=0.9),
    ])] == [2]
    assert resolve_plate_ownership([
        plate(1, box=(10, 60, 30, 75), confidence=0.8),
        plate(1, box=(60, 60, 80, 75), confidence=0.9),
    ])[0].plate_confidence == pytest.approx(0.9)


def test_temporal_ownership_keeps_history_and_one_owner():
    resolver = TemporalPlateOwnershipResolver()
    first = resolver.resolve(0, [plate(1)])
    assert [item.track_id for item in first.candidates] == [1]
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

    def candidate(track_id, frame_index, plate_bbox, confidence=0.8):
        return TrackedPlateCandidate(
            frame_index=frame_index,
            track_id=track_id,
            vehicle_class_id=2,
            vehicle_class_name="car",
            vehicle_confidence=0.9,
            vehicle_bbox=vehicle_boxes[track_id],
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
        item.track_id: item.plate_bbox for item in resolution.candidates
    }
    assert selected == own_plate_boxes
    assert resolution.stats.final_results == 2
    assert resolution.stats.removed_conflict_candidates == 2


def test_quality_and_topk_snapshot():
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    frame[60:75, 20:50] = 200
    crop = crop_plate_from_frame(frame, (20, 60, 50, 75))
    assert crop is not None and crop.shape == (15, 30, 3)
    assert crop_plate_from_frame(frame, (-1, 60, 50, 75)) is None
    quality = score_plate_quality(crop, 0.9)
    assert 0 <= quality.total_score <= 1
    manager = PlateBufferManager(PlateBufferConfig(top_k=2, min_frame_gap=2))
    for frame_index in (0, 1, 3):
        manager.mark_vehicle_seen(1, frame_index)
        manager.add_plate(BufferedPlateCandidate(
            frame_index, 1, 1, "dai", 0.9, (20, 60, 50, 75), quality, crop,
        ))
    assert [candidate.frame_index for candidate in manager.get_top_candidates(1)] == [0, 3]
    assert manager.get_track_summary(1).plate_observation_count == 3


def test_topk_frontier_recovers_compatible_evidence_after_better_middle_crop():
    crop = np.zeros((10, 20, 3), dtype=np.uint8)

    def candidate(frame_index, score):
        quality = PlateQualityMetrics(
            plate_confidence=0.9,
            width=20,
            height=10,
            sharpness_raw=score * 100.0,
            sharpness_score=score,
            size_score=score,
            exposure_score=score,
            total_score=score,
        )
        return BufferedPlateCandidate(
            frame_index, 1, 1, "dai", 0.9, (0, 0, 20, 10), quality, crop,
        )

    manager = PlateBufferManager(PlateBufferConfig(top_k=4, min_frame_gap=2))
    manager.add_plate(candidate(0, 0.80))
    manager.add_plate(candidate(1, 0.90))
    manager.add_plate(candidate(2, 1.00))

    assert [item.frame_index for item in manager.get_top_candidates(1)] == [2, 0]


def test_topk_rejects_out_of_order_input_without_mutating_evidence():
    crop = np.zeros((10, 20, 3), dtype=np.uint8)
    quality = PlateQualityMetrics(0.9, 20, 10, 100.0, 0.9, 0.9, 0.9, 0.9)

    def candidate(frame_index):
        return BufferedPlateCandidate(
            frame_index, 1, 1, "dai", 0.9, (0, 0, 20, 10), quality, crop,
        )

    manager = PlateBufferManager(PlateBufferConfig(top_k=4, min_frame_gap=2))
    manager.add_plate(candidate(0))
    manager.add_plate(candidate(2))

    with pytest.raises(ValueError, match="strictly increasing frame order"):
        manager.add_plate(candidate(1))

    assert [item.frame_index for item in manager.get_top_candidates(1)] == [0, 2]
    assert manager.get_track_summary(1).plate_observation_count == 2


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
        frame, vehicle, identity=7, plate_detector=MultiplePlateDetector(),
        frame_index=0,
    )

    assert len(candidates) == 2
    resolution = TemporalPlateOwnershipResolver().resolve(0, candidates)
    assert len(resolution.candidates) == 1
    assert resolution.candidates[0].track_id == 7
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


def test_tracker_cross_class_duplicate_cleanup():
    tracker = ByteTracker(fps=30.0)
    detections = [
        VehicleDetection(2, "car", 0.9, (0, 0, 100, 100)),
        VehicleDetection(7, "truck", 0.8, (0, 0, 100, 100)),
    ]
    assert tracker.update(detections, 0) == []
    tracks = tracker.update(detections, 1)
    assert len(tracks) == 1
    assert tracks[0].class_name == "car"
    assert tracker.diagnostics["cross_class_detections_removed"] == 2


def test_duplicate_track_is_not_emitted_as_a_second_vehicle():
    tracker = ByteTracker(
        fps=30.0,
        min_confirmed_hits=1,
        cross_class_dedup_enabled=False,
        active_duplicate_suppression_enabled=True,
        active_duplicate_iou_threshold=0.40,
        active_duplicate_min_frames=1,
    )
    first_frame = [
        VehicleDetection(2, "car", 0.95, (0, 0, 100, 100)),
        VehicleDetection(2, "car", 0.95, (120, 0, 220, 100)),
    ]
    overlapping_frame = [
        VehicleDetection(2, "car", 0.95, (50, 0, 150, 100)),
        VehicleDetection(2, "car", 0.95, (55, 0, 155, 100)),
    ]

    tracker.update(first_frame, 0)
    tracker.update(overlapping_frame, 1)

    assert len(tracker.all_tracks) == 2  # Diagnostic history remains visible.
    assert [track.track_id for track in tracker.result_tracks] == [1]
    duplicate = next(track for track in tracker.all_tracks if track.track_id == 2)
    assert duplicate.removal_reason is TrackRemovalReason.DUPLICATE
    assert duplicate.duplicate_of_track_id == 1

    # A real track removed by timeout must still be emitted at end of video.
    timeout_tracker = ByteTracker(
        fps=1.0,
        min_confirmed_hits=1,
        track_buffer_seconds=0.0,
    )
    timeout_tracker.update(
        [VehicleDetection(2, "car", 0.95, (0, 0, 100, 100))], 0,
    )
    timeout_tracker.update([], 1)
    assert len(timeout_tracker.result_tracks) == 1
    assert (
        timeout_tracker.result_tracks[0].removal_reason
        is TrackRemovalReason.TIMEOUT
    )


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
        identity_key="track_id", identity=1, vehicle_class_id=2,
        vehicle_class_name="car", first_frame=0, last_frame=4,
        vehicle_observation_count=5, plate_observation_count=2,
        plate_layout="dai", best_plate_bbox=(20, 60, 50, 75),
        best_plate_frame=1, postprocessed=normalized, fusion=fused,
        ocr_candidate_count=2,
    )
    payload = serialize_vehicle_result(result)
    assert set(payload) == {"track_id", "vehicle", "plate", "evidence"}
    assert payload["plate"]["status"] == "ok"
    assert payload["plate"]["formatted"] == "51A-123.45"
    assert payload["evidence"]["support_count"] == 2
    customer = build_customer_payload({"status": "ok", "vehicles": [payload]})
    assert customer == {"status": "ok", "vehicles": [{
        "track_id": 1, "vehicle_type": "car", "license": "51A-123.45",
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
        identity_key="track_id", identity=2, vehicle_class_id=2,
        vehicle_class_name="car", first_frame=0, last_frame=1,
        vehicle_observation_count=2, plate_observation_count=0,
        plate_layout=None, best_plate_bbox=None, best_plate_frame=None,
        postprocessed=postprocess_vietnam_plate(""), fusion=None,
        ocr_candidate_count=0,
    ).status == "no_plate"


def test_diagnostic_ocr_and_fusion_json_contract():
    class OCRInfo:
        model_info = {"model": "fake"}
        timing_totals = {"total_ms_per_crop": 0.0}
        inference_count = 1
        session_init_count = 1

    candidate = OCRPlateCandidate(
        track_id=1, frame_index=3, rank=1, plate_class_id=1,
        plate_class_name="dai", plate_confidence=0.9, quality_score=0.8,
        bbox=(1, 2, 11, 12), raw_text="51A12345", ocr_confidence=0.9,
        char_confidences=None,
    )
    ocr_json = build_ocr_json([candidate], OCRInfo())
    assert set(ocr_json) == {"status", "ocr_model", "summary", "performance", "tracks"}
    assert ocr_json["tracks"][0]["candidates"][0]["ocr"]["raw_text"] == "51A12345"
    from src.ocr_fusion import fuse_candidates
    fused_json = build_fused_json(ocr_json, fuse_candidates([candidate]))
    assert fused_json["tracks"][0]["fusion"]["raw_text"] == "51A12345"
    assert "v6_fusion" in fused_json
