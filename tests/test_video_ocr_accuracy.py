"""Regression tests for independent video-crop OCR and its diagnostic trace."""

import json

import cv2
import numpy as np
import pytest

from src.microcharnet_ocr import (
    MicroCharNetOCR, OCRCharacter, OutputFormat, _OCRCounters, _RawCharacter,
    _Transform, _class_agnostic_nms, _group_and_sort_characters,
)
from tools.diagnostics.run_video_ocr_accuracy import run_video_ocr_accuracy


def _decoder(names=("A", "B"), threshold=0.25, iou=0.70):
    model = object.__new__(MicroCharNetOCR)
    model.class_names = names
    model.num_classes = len(names)
    model.conf_threshold = threshold
    model.iou_threshold = iou
    return model


def test_decode_end2end_records_raw_filter_nms_and_reading_order():
    model = _decoder()
    rows = np.asarray([[
        [20, 2, 30, 20, 0.9, 1],
        [2, 2, 12, 20, 0.8, 0],
        [2, 2, 12, 20, 0.7, 1],
        [40, 2, 50, 20, 0.24, 0],
    ]], dtype=np.float32)
    trace = {}
    result, characters = model._decode_end2end(rows, _Transform(1, 1, 0, 0, 60, 30), trace)
    assert result.text == "AB"
    assert [item.raw_index for item in characters] == [1, 0]
    assert [trace[key] for key in ("raw_character_count", "after_confidence_count",
                                   "after_nms_count", "final_character_count")] == [4, 3, 2, 2]
    assert [item["status"] for item in trace["rows"]] == [
        "final", "final", "suppressed_nms", "below_confidence",
    ]


def test_confidence_filter_keeps_value_at_threshold():
    model = _decoder(threshold=0.25)
    rows = np.asarray([[[2, 2, 12, 18, 0.25, 0], [20, 2, 30, 18, 0.24, 1]]], dtype=np.float32)
    trace = {}
    result, _ = model._decode_end2end(rows, _Transform(1, 1, 0, 0, 50, 30), trace)
    assert result.text == "A"
    assert trace["after_confidence_count"] == 1
    assert trace["rows"][1]["char"] == "B"


def test_nms_preserves_adjacent_distinct_characters():
    chars = [_RawCharacter("A", 0, 0.9, (0, 0, 10, 20)),
             _RawCharacter("B", 1, 0.8, (10, 0, 20, 20))]
    assert [item.char for item in _class_agnostic_nms(chars, 0.70)] == ["A", "B"]


def test_nms_keeps_best_of_duplicate_bbox_even_if_class_differs():
    chars = [_RawCharacter("A", 0, 0.7, (0, 0, 10, 20)),
             _RawCharacter("B", 1, 0.9, (0, 0, 10, 20))]
    assert [item.char for item in _class_agnostic_nms(chars, 0.70)] == ["B"]


def test_one_row_sorts_left_to_right():
    chars = [OCRCharacter("B", 1, 0.9, (20, 4, 30, 22)),
             OCRCharacter("A", 0, 0.9, (1, 5, 11, 23))]
    trace = {}
    lines = _group_and_sort_characters(chars, trace)
    assert ["".join(item.char for item in line) for line in lines] == ["AB"]
    assert trace["row_count"] == 1


def test_two_rows_sort_top_before_bottom_then_left_to_right():
    chars = [OCRCharacter("D", 3, 0.9, (25, 30, 35, 45)),
             OCRCharacter("B", 1, 0.9, (25, 3, 35, 18)),
             OCRCharacter("C", 2, 0.9, (3, 30, 13, 45)),
             OCRCharacter("A", 0, 0.9, (3, 3, 13, 18))]
    trace = {}
    lines = _group_and_sort_characters(chars, trace)
    assert ["".join(item.char for item in line) for line in lines] == ["AB", "CD"]
    assert trace["split_index"] == 2


def test_letterbox_restores_coordinates_and_rgb_without_stretch():
    model = _decoder()
    model.input_height = model.input_width = 128
    model.preprocess_mode = "letterbox"
    crop = np.zeros((40, 100, 3), dtype=np.uint8)
    crop[:, :] = (10, 20, 200)  # BGR
    tensor, transform = model.preprocess_plate(crop)
    assert tensor.shape == (1, 3, 128, 128)
    assert transform.scale == pytest.approx(transform.scale_y)
    assert transform.pad_y > 0
    assert tensor[0, 0, transform.pad_y + 2, 2] == pytest.approx(200 / 255)
    box = [10 * transform.scale + transform.pad_x,
           5 * transform.scale_y + transform.pad_y,
           30 * transform.scale + transform.pad_x,
           25 * transform.scale_y + transform.pad_y, 0.9, 0]
    result, chars = model._decode_end2end(
        np.asarray([[box]], dtype=np.float32), transform,
    )
    assert result.text == "A"
    assert chars[0].bbox == (10, 5, 30, 25)


def _inference_model(outputs):
    model = _decoder()
    model.input_height = model.input_width = 128
    model.preprocess_mode = "letterbox"
    model.input_name = "images"
    model.output_name = "output0"
    model.output_format = OutputFormat.END2END
    model._counters = _OCRCounters()
    iterator = iter(outputs)

    class Session:
        def run(self, _outputs, _feed):
            return [next(iterator)]

    model.session = Session()
    return model


def test_tiny_crop_does_not_crash_and_no_detection_is_explicitly_empty():
    model = _inference_model([np.zeros((1, 3, 6), dtype=np.float32)] * 2)
    assert model.recognize(np.zeros((1, 1, 3), dtype=np.uint8)).status == "empty"
    result = model.recognize(np.zeros((30, 60, 3), dtype=np.uint8))
    assert result.text == "" and result.confidence == 0.0 and result.status == "empty"


def test_one_crop_does_not_change_another_or_its_trace():
    first = np.asarray([[[10, 10, 30, 30, 0.9, 0]]], dtype=np.float32)
    second = np.asarray([[[10, 10, 30, 30, 0.8, 1]]], dtype=np.float32)
    model = _inference_model([first, second])
    crop_a = np.full((20, 20, 3), 40, dtype=np.uint8)
    crop_b = np.full((20, 20, 3), 100, dtype=np.uint8)
    trace_a, trace_b = {}, {}
    result_a, _, _ = model.recognize_with_debug(crop_a, trace_a)
    result_b, _, _ = model.recognize_with_debug(crop_b, trace_b)
    assert (result_a.text, result_b.text) == ("A", "B")
    assert (trace_a["rows"][0]["char"], trace_b["rows"][0]["char"]) == ("A", "B")
    assert np.all(crop_a == 40) and np.all(crop_b == 100)


def test_trace_mode_does_not_change_production_ocr_result():
    output = np.asarray([[[10, 10, 30, 30, 0.9, 0]]], dtype=np.float32)
    model = _inference_model([output, output])
    crop = np.full((20, 20, 3), 80, dtype=np.uint8)
    plain = model.recognize(crop)
    trace = {}
    debug, characters, _ = model.recognize_with_debug(crop, trace)
    assert (plain.text, plain.confidence, plain.status) == (
        debug.text, debug.confidence, debug.status,
    )
    assert len(characters) == trace["final_character_count"] == 1


def test_raw_fallback_trace_uses_same_filter_and_nms():
    model = _decoder()
    # Two raw xywh locations; the second is below the confidence threshold.
    output = np.asarray([[[20, 40], [20, 20], [10, 10], [10, 10],
                          [0.8, 0.2], [0.1, 0.1]]], dtype=np.float32)
    trace = {}
    result, _ = model._decode_raw(output, _Transform(1, 1, 0, 0, 60, 40), trace)
    assert result.text == "A"
    assert (trace["raw_character_count"], trace["after_confidence_count"],
            trace["after_nms_count"]) == (2, 1, 1)


def test_diagnostic_writes_one_crop_and_no_eligible_track(tmp_path):
    video = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    assert writer.isOpened()
    writer.write(np.full((48, 64, 3), 150, dtype=np.uint8))
    writer.release()
    topk = tmp_path / "topk.json"
    topk.write_text(json.dumps({"tracks": [
        {"track_id": 1, "all_evidence_count": 1, "eligible_evidence_count": 1,
         "top_k": [{"rank": 1, "frame_index": 0, "plate_bbox_xyxy": [4, 5, 44, 30],
                    "plate_class_name": "vuong", "plate_confidence": 0.8,
                    "quality": {"width": 40, "height": 25, "total_score": 0.8,
                                "sharpness_score": 0.7, "size_score": 0.6,
                                "exposure_score": 0.9}}]},
        {"track_id": 2, "all_evidence_count": 3, "eligible_evidence_count": 0,
         "top_k": []},
    ]}), encoding="utf-8")
    summary = run_video_ocr_accuracy(topk, video, tmp_path / "review")
    assert len(summary["tracks"][0]["crops"]) == 1
    result = json.loads((tmp_path / "review/track_1/crop_1/result.json").read_text())
    assert result["ocr"]["final_character_count"] == len(result["characters"])
    assert (tmp_path / "review/track_1/crop_1/preprocessed.jpg").is_file()
    assert (tmp_path / "review/track_1/crop_1/characters.jpg").is_file()
    assert (tmp_path / "review/track_2/no_eligible_crop.json").is_file()
    assert (tmp_path / "review/summary.md").is_file()
