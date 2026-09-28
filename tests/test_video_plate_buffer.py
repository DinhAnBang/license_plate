"""Tests for Module 6 video crop quality and Top-K buffering."""

import json
from itertools import combinations

import cv2
import numpy as np
import pytest

from src.core.plate_detector import PlateDetection
from src.core.vehicle_detector import VehicleDetection
from src.video.plate_buffer import VideoBufferedPlate, VideoPlateBuffer, VideoPlateBufferConfig
from src.core.plate_quality import VideoPlateQualityConfig, VideoPlateQualityMetrics, crop_plate_from_frame, score_plate_quality
from src.video.plate_stage import VideoPlateCandidate
from src.video.vehicle_validation import VehicleValidationConfig


def _candidate(frame_index: int, candidate_index: int = 0) -> VideoPlateCandidate:
    return VideoPlateCandidate(
        frame_index=frame_index,
        candidate_index=candidate_index,
        track_id=1,
        vehicle_class_id=2,
        vehicle_class_name="car",
        vehicle_confidence=0.90,
        vehicle_bbox=(0, 0, 100, 100),
        plate_class_id=1,
        plate_class_name="dai",
        plate_confidence=0.90,
        plate_bbox=(20, 70, 80, 90),
    )


def _buffered(frame_index: int, score: float) -> VideoBufferedPlate:
    metrics = VideoPlateQualityMetrics(
        plate_confidence=score,
        width=60,
        height=30,
        sharpness_raw=score * 100,
        sharpness_score=score,
        size_score=score,
        exposure_score=score,
        total_score=score,
    )
    return VideoBufferedPlate(
        candidate=_candidate(frame_index),
        quality=metrics,
        crop=np.zeros((30, 60, 3), dtype=np.uint8),
    )


def test_buffer_rebuild_keeps_good_evidence_after_late_bad_candidate():
    buffer = VideoPlateBuffer(VideoPlateBufferConfig(top_k=2, min_frame_gap=1))
    buffer.add(_buffered(0, 0.90))
    buffer.add(_buffered(1, 0.10))
    buffer.add(_buffered(2, 0.80))

    selected_frames = [item.candidate.frame_index for item in buffer.selected(1)]
    assert selected_frames == [0, 2]
    assert len(buffer.all_candidates(1)) == 3


def test_buffer_enforces_temporal_gap_and_does_not_force_four():
    buffer = VideoPlateBuffer(VideoPlateBufferConfig(top_k=4, min_frame_gap=5))
    for frame, score in [(0, 0.8), (1, 0.85), (2, 0.9), (6, 0.8)]:
        buffer.add(_buffered(frame, score))
    assert [item.candidate.frame_index for item in buffer.selected(1)] == [1, 6]
    assert buffer.eligible_count(1) == 4


def test_buffer_out_of_order_rejected_without_mutating_state():
    buffer = VideoPlateBuffer()
    buffer.add(_buffered(10, 0.9))
    before = buffer.selected(1)
    with pytest.raises(ValueError, match="out-of-order"):
        buffer.add(_buffered(9, 0.95))
    assert buffer.selected(1) == before
    assert len(buffer.all_candidates(1)) == 1


@pytest.mark.parametrize("scores", [[0.7, 0.8, 0.9, 0.95, 0.85], [0.95, 0.9, 0.8, 0.7, 0.85]])
def test_buffer_keeps_four_best_when_frames_are_separated(scores):
    buffer = VideoPlateBuffer(VideoPlateBufferConfig(top_k=4, min_frame_gap=5))
    for index, score in enumerate(scores):
        buffer.add(_buffered(index * 5, score))
    assert [item.quality.total_score for item in buffer.selected(1)] == sorted(scores, reverse=True)[:4]


@pytest.mark.parametrize("gap", [1, 3, 5])
def test_buffer_matches_brute_force_best_combination(gap):
    scores = [0.91, 0.87, 0.93, 0.72, 0.86, 0.90]
    buffer = VideoPlateBuffer(VideoPlateBufferConfig(top_k=3, min_frame_gap=gap))
    for frame, score in enumerate(scores):
        buffer.add(_buffered(frame, score))
    possibilities = [
        subset for size in range(4) for subset in combinations(range(len(scores)), size)
        if all(right - left >= gap for left, right in zip(subset, subset[1:]))
    ]
    expected = max(sum(scores[index] for index in subset) for subset in possibilities)
    actual = sum(item.quality.total_score for item in buffer.selected(1))
    assert actual == pytest.approx(expected)


def test_buffer_does_not_fill_topk_with_tiny_crop():
    buffer = VideoPlateBuffer(VideoPlateBufferConfig(top_k=4))
    buffer.add(_buffered(0, 0.90))
    tiny = _buffered(10, 0.80)
    tiny_metrics = VideoPlateQualityMetrics(
        plate_confidence=0.80,
        width=32,
        height=19,
        sharpness_raw=100.0,
        sharpness_score=0.70,
        size_score=0.30,
        exposure_score=0.90,
        total_score=0.80,
    )
    buffer.add(VideoBufferedPlate(tiny.candidate, tiny_metrics, tiny.crop[:, :32]))

    assert buffer.eligible_count(1) == 1
    assert len(buffer.selected(1)) == 1
    assert buffer.rejection_reason(buffer.all_candidates(1)[1]) == "crop_too_narrow"


def test_quality_score_uses_original_focus_information():
    sharp = np.zeros((80, 120, 3), dtype=np.uint8)
    sharp[:, ::4] = 255
    blurred = cv2.GaussianBlur(sharp, (15, 15), 0)

    sharp_metrics = score_plate_quality(sharp, 0.8, VideoPlateQualityConfig())
    blurred_metrics = score_plate_quality(blurred, 0.8, VideoPlateQualityConfig())

    assert sharp_metrics.sharpness_raw > blurred_metrics.sharpness_raw
    assert sharp_metrics.sharpness_score > blurred_metrics.sharpness_score


def test_crop_is_exact_bbox_and_invalid_coordinates_return_none():
    frame = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
    crop = crop_plate_from_frame(frame, (2, 1, 6, 5))
    assert np.array_equal(crop, frame[1:5, 2:6])
    assert crop.shape == (4, 4, 3)
    crop[0, 0] = 0
    assert not np.array_equal(crop, frame[1:5, 2:6])
    assert crop_plate_from_frame(frame, (-1, 1, 6, 5)) is None
    assert crop_plate_from_frame(frame, (2, 1, 9, 5)) is None
