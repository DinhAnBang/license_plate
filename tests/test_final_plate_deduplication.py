"""T5 final duplicate removal tests with deterministic synthetic tracklets."""

from __future__ import annotations

from core.final_plate_deduplication import FinalPlateDeduplicator
from core.tracklet_stitcher import PlateEvent, Tracklet, TrackletCandidate


def make_event(
    event_id: int,
    track_id: int,
    first: int,
    last: int,
    text: str,
    history: list[tuple[int, tuple[int, int, int, int]]],
    *,
    candidate_texts: tuple[str, ...] | None = None,
    status: str = "VALID",
) -> PlateEvent:
    texts = candidate_texts or (text,)
    candidates = tuple(
        TrackletCandidate(
            track_id=track_id,
            frame_index=last - index,
            detection_confidence=0.8,
            quality=0.8 - index * 0.01,
            sharpness=0.8,
            sharpness_raw=100.0,
            brightness=0.8,
            brightness_raw=128.0,
            size=0.5,
            crop_width=100,
            crop_height=40,
            crop_area=4_000,
            aspect_ratio=2.5,
            box=history[-1][1],
            crop=f"crop-{track_id}-{index}",
            raw_text=raw,
            plate_text=normalized,
            ocr_confidence=0.9,
            validation_status=status,
            validation_score=1.0 if status == "VALID" else 0.4,
        )
        for index, (raw, normalized) in enumerate((value, value) for value in texts)
    )
    tracklet = Tracklet(
        track_id=track_id,
        first_detected_frame=first,
        last_detected_frame=last,
        first_box=history[0][1],
        last_box=history[-1][1],
        hits=len(history),
        plate_text=text,
        ocr_confidence=0.9,
        candidates=candidates,
        selected_candidate=candidates[0],
        observation_history=tuple(history),
    )
    return PlateEvent(
        event_id=event_id,
        tracklets=[tracklet],
        canonical_plate_text=text,
        canonical_ocr_confidence=0.9,
        best_candidate=candidates[0],
        canonical_validation_status=status,
        canonical_validation_score=1.0 if status == "VALID" else 0.4,
    )


def boxes(start: int, stop: int, box: tuple[int, int, int, int] = (10, 10, 110, 50)):
    return [(frame, box) for frame in range(start, stop + 1)]


def test_sequential_same_text_merges() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 3, "59S120468", boxes(1, 3)),
            make_event(2, 8, 5, 7, "59S120468", boxes(5, 7)),
        ],
        30.0,
    )
    assert result.metrics["sequential_duplicate_merges"] == 1
    assert result.events[0].member_track_ids == [3, 8]


def test_o_zero_normalized_text_merges() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 3, "59S120468", boxes(1, 3)),
            make_event(2, 8, 5, 7, "59S120468", boxes(5, 7), candidate_texts=("59S12O468", "59S120468")),
        ],
        30.0,
    )
    assert len(result.events) == 1


def test_overlap_with_multi_frame_same_bbox_merges() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 5, "59S120468", boxes(1, 5)),
            make_event(2, 8, 3, 7, "59S120468", boxes(3, 7)),
        ],
        30.0,
    )
    assert result.metrics["overlapping_duplicate_merges"] == 1
    assert result.events[0].member_track_ids == [3, 8]


def test_overlap_same_text_but_far_bbox_does_not_merge() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 5, "59S120468", boxes(1, 5)),
            make_event(2, 8, 3, 7, "59S120468", boxes(3, 7, (500, 10, 600, 50))),
        ],
        30.0,
    )
    assert len(result.events) == 2
    assert result.decisions[-1]["reason"] == "REJECT_OVERLAP_PHYSICAL_IDENTITY"


def test_two_different_close_plates_do_not_merge() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 5, "59S120468", boxes(1, 5)),
            make_event(2, 8, 3, 7, "59S120469", boxes(3, 7, (12, 10, 112, 50))),
        ],
        30.0,
    )
    assert len(result.events) == 2


def test_same_text_too_far_in_time_does_not_merge() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 3, "59S120468", boxes(1, 3)),
            make_event(2, 8, 100, 102, "59S120468", boxes(100, 102)),
        ],
        30.0,
    )
    assert len(result.events) == 2
    assert result.decisions[-1]["reason"] == "REJECT_TIME_GAP"


def test_invalid_event_cannot_pull_another_event_into_merge() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 3, "CONPHONG", boxes(1, 3), status="INVALID"),
            make_event(2, 8, 5, 7, "59S120468", boxes(5, 7)),
        ],
        30.0,
    )
    assert len(result.events) == 2
    assert result.decisions[-1]["reason"] == "REJECT_INVALID_EVENT"


def test_top_k_consensus_can_merge_different_final_strings() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 3, "59S120468", boxes(1, 3), candidate_texts=("59S120468", "59S12O468")),
            make_event(2, 8, 5, 7, "59S12046B", boxes(5, 7), candidate_texts=("59S12O468", "59S12046B")),
        ],
        30.0,
    )
    assert len(result.events) == 1


def test_edit_distance_two_without_consensus_does_not_merge() -> None:
    result = FinalPlateDeduplicator().deduplicate(
        [
            make_event(1, 3, 1, 3, "59S120468", boxes(1, 3)),
            make_event(2, 8, 5, 7, "59S120499", boxes(5, 7)),
        ],
        30.0,
    )
    assert len(result.events) == 2


def test_input_order_does_not_change_final_result() -> None:
    first = [
        make_event(1, 3, 1, 3, "59S120468", boxes(1, 3)),
        make_event(2, 8, 5, 7, "59S120468", boxes(5, 7)),
    ]
    second = [
        make_event(2, 8, 5, 7, "59S120468", boxes(5, 7)),
        make_event(1, 3, 1, 3, "59S120468", boxes(1, 3)),
    ]
    left = FinalPlateDeduplicator().deduplicate(first, 30.0)
    right = FinalPlateDeduplicator().deduplicate(second, 30.0)
    assert [(event.member_track_ids, event.canonical_plate_text) for event in left.events] == [
        (event.member_track_ids, event.canonical_plate_text) for event in right.events
    ]


if __name__ == "__main__":
    for test in (
        test_sequential_same_text_merges,
        test_o_zero_normalized_text_merges,
        test_overlap_with_multi_frame_same_bbox_merges,
        test_overlap_same_text_but_far_bbox_does_not_merge,
        test_two_different_close_plates_do_not_merge,
        test_same_text_too_far_in_time_does_not_merge,
        test_invalid_event_cannot_pull_another_event_into_merge,
        test_top_k_consensus_can_merge_different_final_strings,
        test_edit_distance_two_without_consensus_does_not_merge,
        test_input_order_does_not_change_final_result,
    ):
        test()
    print("T5 final deduplication tests: OK")
