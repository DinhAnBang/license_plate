"""Regression coverage for the video-only VN plate postprocess adapter."""

import json
from unittest.mock import patch

import pytest

from src.video.postprocess_stage import postprocess_video_fused
from tools.diagnostics.run_video_postprocess_accuracy import run_video_postprocess_accuracy


@pytest.mark.parametrize("raw, expected", [
    ("72a16231", "72A-162.31"),
    ("51b77569", "51B-775.69"),
    ("64a20633", "64A-206.33"),
])
def test_real_phone_fusion_text_normalized_and_formatted(raw, expected):
    result = postprocess_video_fused(raw, 0.8, "car")
    assert result.valid
    assert result.status == "ok"
    assert result.formatted_text == expected
    assert result.postprocessed.raw_text == raw
    assert result.postprocessed.normalized_text == raw.upper()
    assert result.postprocessed.corrected_text == raw.upper()
    assert result.postprocessed.corrections == ()
    assert result.postprocessed.format_family == "car_common"
    assert result.postprocessed.fusion_confidence == 0.8


def test_missing_character_is_never_filled_in():
    result = postprocess_video_fused("72A162", 0.8, "car")
    assert not result.valid and result.formatted_text is None
    assert result.reason == "invalid_length"
    assert result.postprocessed.corrected_text == "72A162"


def test_seven_character_car_plate_is_ambiguous_not_extended():
    result = postprocess_video_fused("72A1623", 0.8, "car")
    # Current rules also accept a genuine four-digit registration number.
    assert result.postprocessed.corrected_text == "72A1623"
    assert result.postprocessed.corrections == ()
    assert result.formatted_text == "72A-16.23"


def test_extra_character_is_not_deleted_or_reinterpreted_as_motorbike():
    result = postprocess_video_fused("72A162311", 0.8, "car")
    assert not result.valid and result.formatted_text is None
    assert result.status == "vehicle_plate_family_mismatch"
    assert result.postprocessed.normalized_text == result.postprocessed.corrected_text
    assert result.postprocessed.corrections == ()


def test_digit_position_allows_ocr_o_to_zero_with_trace():
    result = postprocess_video_fused("72A16O31", 0.8, "car")
    assert result.valid and result.formatted_text == "72A-160.31"
    assert [(c.index, c.from_char, c.to_char, c.reason)
            for c in result.postprocessed.corrections] == [
                (5, "O", "0", "digit_expected")]


def test_letter_position_allows_eight_to_b_with_trace():
    result = postprocess_video_fused("72816231", 0.8, "car")
    assert result.valid and result.formatted_text == "72B-162.31"
    assert [(c.index, c.from_char, c.to_char, c.reason)
            for c in result.postprocessed.corrections] == [
                (2, "8", "B", "letter_expected")]


def test_zero_to_o_at_letter_position_is_rejected_by_current_serial_rules():
    result = postprocess_video_fused("72016231", 0.8, "car")
    assert not result.valid and result.formatted_text is None
    assert result.reason == "invalid_serial"
    assert result.postprocessed.corrections == ()


def test_unconstrained_motorbike_serial_position_keeps_eight():
    result = postprocess_video_fused("72A816231", 0.8, "motorcycle")
    assert result.valid and result.formatted_text == "72A8-162.31"
    assert result.postprocessed.corrections == ()


@pytest.mark.parametrize("raw", ["XXA12345", "72A#6231", ""])
def test_unrecognized_text_not_given_a_plausible_formatted_plate(raw):
    result = postprocess_video_fused(raw, 0.8, "car")
    assert not result.valid and result.formatted_text is None


def test_no_ocr_does_not_call_postprocessor():
    with patch("src.video.postprocess_stage.postprocess_vietnam_plate") as postprocess:
        result = postprocess_video_fused(None, 0.0, "car")
    assert result.status == "no_ocr" and result.postprocessed is None
    postprocess.assert_not_called()


def test_video_never_passes_character_confidences_for_deletion():
    real_result = postprocess_video_fused("72A16231", 0.8, "car").postprocessed
    with patch("src.video.postprocess_stage.postprocess_vietnam_plate") as postprocess:
        postprocess.return_value = real_result
        postprocess_video_fused("72A162311", 0.8, "car")
    assert postprocess.call_args.kwargs["char_confidences"] is None


def test_fusion_input_remains_raw_and_confidence_is_not_promoted():
    raw = " 72a-162.31 "
    result = postprocess_video_fused(raw, 0.37, "car")
    assert result.postprocessed.raw_text == raw
    assert result.postprocessed.normalized_text == "72A16231"
    assert result.postprocessed.fusion_confidence == 0.37


def test_too_many_position_corrections_are_not_formatted():
    result = postprocess_video_fused("7ZA16O3B", 0.8, "car")
    assert not result.valid and result.formatted_text is None
    assert result.status == "low_format_confidence"
    assert result.reason == "too_many_position_corrections"


def test_diagnostic_reuses_adapter_and_keeps_empty_track(tmp_path):
    fusion = tmp_path / "fusion.json"
    fusion.write_text(json.dumps({"tracks": [
        {"track_id": 2, "raw_text": "72a16231", "confidence": 0.8,
         "method": "weighted_exact_vote"},
        {"track_id": 4, "raw_text": None, "confidence": 0.0,
         "method": "no_valid_ocr"},
    ]}), encoding="utf-8")
    topk = tmp_path / "topk.json"
    topk.write_text(json.dumps({"frames": [{"vehicles": [
        {"track_id": 2, "class_name": "car"},
        {"track_id": 4, "class_name": "car"},
    ]}]}), encoding="utf-8")
    output = tmp_path / "postprocess"
    summary = run_video_postprocess_accuracy(fusion, topk, output)
    assert summary["tracks_total"] == 2
    assert summary["tracks_with_ocr"] == summary["structurally_valid"] == 1
    assert summary["no_ocr"] == 1
    report2 = json.loads((output / "track_2/report.json").read_text(encoding="utf-8"))
    report4 = json.loads((output / "track_4/report.json").read_text(encoding="utf-8"))
    assert report2["raw_text"] == "72a16231"
    assert report2["normalized_text"] == "72A16231"
    assert report2["corrections"] == []
    assert report2["format_family"] == "car_common"
    assert report2["formatted"] == "72A-162.31"
    assert report4["status"] == "no_ocr" and report4["formatted"] is None
    assert (output / "summary.md").is_file()


def test_diagnostic_rejects_conflicting_vehicle_classes(tmp_path):
    fusion = tmp_path / "fusion.json"
    fusion.write_text(json.dumps({"tracks": [
        {"track_id": 2, "raw_text": "72a16231", "confidence": 0.8,
         "method": "single_candidate"},
    ]}), encoding="utf-8")
    topk = tmp_path / "topk.json"
    topk.write_text(json.dumps({"frames": [
        {"vehicles": [{"track_id": 2, "class_name": "car"}]},
        {"vehicles": [{"track_id": 2, "class_name": "motorcycle"}]},
    ]}), encoding="utf-8")
    with pytest.raises(ValueError, match="Inconsistent vehicle families"):
        run_video_postprocess_accuracy(fusion, topk, tmp_path / "output")


def test_diagnostic_records_class_drift_within_same_plate_family(tmp_path):
    fusion = tmp_path / "fusion.json"
    fusion.write_text(json.dumps({"tracks": [
        {"track_id": 1, "raw_text": "51b77569", "confidence": 0.7,
         "method": "single_candidate"},
    ]}), encoding="utf-8")
    topk = tmp_path / "topk.json"
    topk.write_text(json.dumps({"frames": [
        {"vehicles": [{"track_id": 1, "class_name": name}]}
        for name in ("car", "truck", "car")
    ]}), encoding="utf-8")
    output = tmp_path / "output"
    run_video_postprocess_accuracy(fusion, topk, output)
    report = json.loads((output / "track_1/report.json").read_text(encoding="utf-8"))
    assert report["vehicle_class_name"] == "car"
    assert report["vehicle_class_counts"] == {"car": 2, "truck": 1}
    assert report["format_family"] == "car_common"
