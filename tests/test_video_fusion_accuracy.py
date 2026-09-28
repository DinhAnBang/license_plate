"""Regression tests for video OCR fusion without VN postprocessing."""

import json
from pathlib import Path

import pytest

from src.ocr_fusion import (
    GAP, OCRFusionCandidate, candidate_weight, fuse_candidates, fuse_track,
)
from tools.diagnostics.run_video_fusion_accuracy import run_video_fusion_accuracy


def _candidate(rank, text, confidence=0.9, quality=0.9, frame=None):
    return OCRFusionCandidate(
        track_id=3, frame_index=rank if frame is None else frame,
        rank=rank, raw_text=text, ocr_confidence=confidence,
        quality_score=quality,
    )


def test_single_candidate_returns_original_text_without_fake_consensus():
    result = fuse_track(3, [_candidate(1, "51b77569")])
    assert result.raw_text == "51b77569"
    assert result.method == "single_candidate"
    assert result.support_count == result.valid_candidate_count == 1


def test_four_identical_candidates_use_exact_vote():
    result = fuse_track(3, [_candidate(rank, "72a16231") for rank in range(1, 5)])
    assert result.raw_text == "72a16231"
    assert result.method == "weighted_exact_vote"
    assert result.support_count == 4


def test_three_good_one_bad_does_not_change_result():
    items = [_candidate(rank, "72a16231", 0.82, 0.9) for rank in range(1, 4)]
    items.append(_candidate(4, "72a1671", 0.73, 0.8))
    result = fuse_track(3, items)
    assert result.raw_text == "72a16231"
    assert result.method == "weighted_exact_vote"
    assert result.exact_votes[0].weight > result.exact_votes[1].weight


def test_missing_character_aligns_to_gap_not_shifted_suffix():
    items = [_candidate(1, "64a20633"), _candidate(2, "64a2033"),
             _candidate(3, "64a20533"), _candidate(4, "64a20633")]
    result = fuse_track(3, items)
    assert result.method == "sequence_alignment"
    assert result.raw_text == "64a20633"
    assert result.alignment_rows[1][2] == tuple("64a20") + (GAP,) + tuple("33")
    assert result.alignment_column_votes[5].selected_token == "6"


def test_character_confidences_explain_six_vs_five_vs_gap():
    specs = [
        (1, "64a20633", 0.712, 0.935, 0.58),
        (2, "64a2033", 0.770, 0.923, None),
        (3, "64a20533", 0.763, 0.903, 0.61),
        (4, "64a20633", 0.794, 0.901, 0.86),
    ]
    items = []
    for rank, text, ocr, quality, disputed_conf in specs:
        char_conf = [0.8] * len(text)
        if disputed_conf is not None:
            char_conf[5] = disputed_conf
        items.append(OCRFusionCandidate(
            3, rank, rank, text, ocr, quality, tuple(char_conf),
        ))
    result = fuse_track(3, items)
    vote = result.alignment_column_votes[5]
    weights = dict(vote.character_weights)
    assert result.raw_text == "64a20633"
    assert weights["6"] > vote.gap_weight > weights["5"]


def test_extra_character_aligns_to_insertion_and_is_outvoted_by_gaps():
    items = [_candidate(1, "ABCDEF", 0.95), _candidate(2, "ABCDEF", 0.9),
             _candidate(3, "ABXCDEF", 0.8), _candidate(4, "ABCDZF", 0.8)]
    result = fuse_track(3, items)
    assert result.method == "sequence_alignment"
    assert result.raw_text == "ABCDEF"
    insertion = next(vote for vote in result.alignment_column_votes
                     if vote.reference_token == GAP)
    assert insertion.selected_token == GAP
    assert insertion.gap_weight > dict(insertion.character_weights)["X"]


def test_one_character_disagreement_uses_weighted_character_votes():
    items = [_candidate(1, "64a20633"), _candidate(2, "64a20633"),
             _candidate(3, "64a20533"), _candidate(4, "64a20633")]
    result = fuse_track(3, items)
    assert result.raw_text == "64a20633"
    assert result.exact_votes[0].count == 3


def test_two_weak_matching_votes_do_not_beat_one_strong_vote():
    items = [_candidate(1, "ABC", 0.95, 1.0),
             _candidate(2, "AXC", 0.30, 0.4),
             _candidate(3, "AXC", 0.28, 0.4)]
    result = fuse_track(3, items)
    assert result.raw_text == "ABC"
    assert candidate_weight(items[0]) > sum(candidate_weight(item) for item in items[1:])


def test_all_different_candidates_take_alignment_path():
    items = [_candidate(1, "ABCDEF", 0.9),
             _candidate(2, "ABXDEF", 0.8),
             _candidate(3, "ABCDE", 0.7)]
    result = fuse_track(3, items)
    assert result.method == "sequence_alignment"
    assert result.reference_text in {item.raw_text for item in items}
    assert len(result.alignment_rows) == 3


def test_zero_candidates_yield_explicit_no_valid_ocr():
    report = fuse_candidates([], track_ids=[4])
    result = report.results[0]
    assert result.track_id == 4
    assert result.method == "no_valid_ocr"
    assert result.raw_text == "" and result.valid_candidate_count == 0


def test_empty_candidate_is_ignored_without_crashing():
    result = fuse_track(3, [_candidate(1, ""), _candidate(2, "ABC")])
    assert result.raw_text == "ABC"
    assert result.method == "single_candidate"


def test_fusion_does_not_apply_vietnam_plate_format():
    result = fuse_track(3, [_candidate(1, "72a16231")])
    assert result.raw_text == "72a16231"
    assert "-" not in result.raw_text and "." not in result.raw_text
    source = Path(__file__).resolve().parents[1] / "src/ocr_fusion.py"
    assert "postprocess_vietnam_plate" not in source.read_text(encoding="utf-8")


def test_plate_confidence_is_off_by_default():
    candidate = OCRFusionCandidate(3, 1, 1, "ABC", 0.9, 0.8,
                                   plate_confidence=0.1)
    assert candidate_weight(candidate) == pytest.approx(0.72)


def test_diagnostic_uses_production_fusion_and_keeps_empty_track(tmp_path):
    crop_result = tmp_path / "crop.json"
    crop_result.write_text(json.dumps({
        "ocr": {"text": "72a16231", "confidence": 0.9, "status": "ok"},
        "characters": [{"confidence": 0.9}] * 8,
    }), encoding="utf-8")
    summary = tmp_path / "ocr_summary.json"
    summary.write_text(json.dumps({"tracks": [
        {"track_id": 1, "crops": [{
            "rank": 1, "frame_index": 2, "ocr_text": "72a16231",
            "ocr_confidence": 0.9, "quality_score": 0.8,
            "plate_confidence": 0.7, "result_path": str(crop_result),
        }]},
        {"track_id": 4, "crops": []},
    ]}), encoding="utf-8")
    result = run_video_fusion_accuracy(summary, tmp_path / "fusion")
    assert [track["raw_text"] for track in result["tracks"]] == ["72a16231", None]
    track1 = json.loads((tmp_path / "fusion/track_1/report.json").read_text())
    track4 = json.loads((tmp_path / "fusion/track_4/report.json").read_text())
    assert track1["inputs"][0]["weight"] == pytest.approx(0.72)
    assert track1["fusion"]["method"] == "single_candidate"
    assert track4["status"] == "no_ocr_candidates"
    assert track4["fusion"]["raw_text"] is None
    assert (tmp_path / "fusion/summary.md").is_file()
