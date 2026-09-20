"""T4 conservative tracklet stitching unit tests."""

from __future__ import annotations

from core.tracklet_stitcher import (
    Tracklet,
    TrackletCandidate,
    TrackletStitchConfig,
    TrackletStitcher,
    levenshtein_distance,
)


def candidate(
    track_id: int,
    frame: int,
    text: str,
    *,
    ocr_conf: float = 0.9,
    quality: float = 0.7,
    det_conf: float = 0.8,
    box: tuple[int, int, int, int] = (10, 10, 110, 50),
) -> TrackletCandidate:
    return TrackletCandidate(
        track_id=track_id,
        frame_index=frame,
        detection_confidence=det_conf,
        quality=quality,
        sharpness=quality,
        sharpness_raw=100.0,
        brightness=0.8,
        brightness_raw=128.0,
        size=0.5,
        crop_width=100,
        crop_height=40,
        crop_area=4_000,
        aspect_ratio=2.5,
        box=box,
        crop=f"crop-{track_id}-{frame}",
        raw_text=text,
        plate_text=text,
        ocr_confidence=ocr_conf,
    )


def tracklet(
    track_id: int,
    first: int,
    last: int,
    text: str,
    *,
    first_box: tuple[int, int, int, int] = (10, 10, 110, 50),
    last_box: tuple[int, int, int, int] | None = None,
    ocr_conf: float = 0.9,
    quality: float = 0.7,
    det_conf: float = 0.8,
) -> Tracklet:
    last_box = last_box or first_box
    selected = candidate(
        track_id,
        last,
        text,
        ocr_conf=ocr_conf,
        quality=quality,
        det_conf=det_conf,
        box=last_box,
    )
    return Tracklet(
        track_id=track_id,
        first_detected_frame=first,
        last_detected_frame=last,
        first_box=first_box,
        last_box=last_box,
        hits=last - first + 1,
        plate_text=text,
        ocr_confidence=ocr_conf,
        candidates=(selected,),
        selected_candidate=selected,
    )


def stitch(*items: Tracklet, fps: float = 30.0):
    return TrackletStitcher().stitch(list(items), fps)


def test_levenshtein_and_exact_merge() -> None:
    assert levenshtein_distance("59N304864", "59N304864") == 0
    assert levenshtein_distance("59N304864", "59N304B64") == 1
    result = stitch(
        tracklet(13, 938, 965, "59N304864"),
        tracklet(14, 971, 991, "59N304864", first_box=(20, 10, 120, 50)),
    )
    assert len(result.events) == 1
    assert result.events[0].member_track_ids == [13, 14]
    assert result.events[0].first_detected_frame == 938
    assert result.events[0].last_detected_frame == 991
    assert result.metrics["exact_ocr_merges"] == 1
    assert result.decisions[-1]["reason"] == "MERGED_EXACT_OCR"


def test_fuzzy_merge_chooses_stronger_ocr_evidence() -> None:
    result = stitch(
        tracklet(1, 100, 130, "59N304864", ocr_conf=0.95, quality=0.8),
        tracklet(
            2,
            135,
            160,
            "59N304B64",
            ocr_conf=0.60,
            quality=0.7,
            first_box=(15, 10, 115, 50),
        ),
    )
    event = result.events[0]
    assert len(result.events) == 1
    assert event.canonical_plate_text == "59N304864"
    assert event.best_candidate.track_id == 1
    assert result.metrics["fuzzy_distance1_merges"] == 1


def test_ocr_distance_and_shared_prefix_do_not_merge() -> None:
    far_text = stitch(
        tracklet(1, 10, 20, "59N304864"),
        tracklet(2, 22, 30, "59N314B72"),
    )
    assert len(far_text.events) == 2
    assert any(item["reason"] == "REJECT_OCR_DISTANCE" for item in far_text.decisions)

    shared_prefix = stitch(
        tracklet(1, 10, 20, "51A123456"),
        tracklet(2, 22, 30, "51A123789"),
    )
    assert len(shared_prefix.events) == 2

    short_fuzzy = stitch(
        tracklet(1, 10, 20, "12345"),
        tracklet(2, 22, 30, "12346"),
    )
    assert len(short_fuzzy.events) == 2


def test_time_spatial_and_overlap_conflicts() -> None:
    time_gap = stitch(
        tracklet(1, 100, 150, "59N304864"),
        tracklet(2, 241, 260, "59N304864"),
    )
    assert len(time_gap.events) == 2
    assert time_gap.decisions[-1]["reason"] == "REJECT_TIME_GAP"

    spatial = stitch(
        tracklet(1, 100, 150, "59N304864", last_box=(0, 0, 50, 20)),
        tracklet(2, 153, 170, "59N304864", first_box=(900, 0, 950, 20)),
    )
    assert len(spatial.events) == 2
    assert spatial.decisions[-1]["reason"] == "REJECT_SPATIAL"

    overlap = stitch(
        tracklet(1, 100, 150, "59N304864"),
        tracklet(2, 120, 160, "59N304864"),
    )
    assert len(overlap.events) == 2
    assert overlap.decisions[-1]["reason"] == "REJECT_OVERLAP"


def test_chain_transitive_safety_and_best_crop_metadata() -> None:
    chain = stitch(
        tracklet(1, 100, 130, "59N304864", quality=0.4, det_conf=0.75),
        tracklet(
            2,
            135,
            160,
            "59N304864",
            quality=0.9,
            det_conf=0.88,
            first_box=(20, 10, 120, 50),
        ),
        tracklet(
            3,
            165,
            200,
            "59N304864",
            quality=0.6,
            first_box=(30, 10, 130, 50),
        ),
    )
    event = chain.events[0]
    assert len(chain.events) == 1 and event.member_track_ids == [1, 2, 3]
    assert event.first_detected_frame == 100 and event.last_detected_frame == 200
    assert event.best_candidate.track_id == 2
    assert event.best_candidate.frame_index == 160
    assert event.best_candidate.detection_confidence == 0.88

    transitive_safe = stitch(
        tracklet(1, 100, 130, "59N304864"),
        tracklet(2, 135, 160, "59N304864", first_box=(20, 10, 120, 50)),
        tracklet(3, 165, 200, "59N304B64", first_box=(900, 10, 1000, 50)),
    )
    assert [event.member_track_ids for event in transitive_safe.events] == [[1, 2], [3]]
    assert any(item["reason"] == "REJECT_SPATIAL" for item in transitive_safe.decisions)


def test_feature_toggle_keeps_one_event_per_tracklet() -> None:
    disabled = TrackletStitcher(
        TrackletStitchConfig(enabled=False)
    ).stitch(
        [
            tracklet(1, 10, 20, "59N304864"),
            tracklet(2, 22, 30, "59N304864"),
        ],
        30.0,
    )
    assert len(disabled.events) == 2
    assert disabled.metrics["number_of_merges"] == 0
    assert disabled.decisions == ()


def main() -> int:
    test_levenshtein_and_exact_merge()
    test_fuzzy_merge_chooses_stronger_ocr_evidence()
    test_ocr_distance_and_shared_prefix_do_not_merge()
    test_time_spatial_and_overlap_conflicts()
    test_chain_transitive_safety_and_best_crop_metadata()
    test_feature_toggle_keeps_one_event_per_tracklet()
    print("T4 TrackletStitcher unit tests: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
