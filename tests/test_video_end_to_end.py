"""Video production routing, finalization, and two-pass rendering regressions."""

import json
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

import src.video.pipeline as video_pipeline
from src.alpr_pipeline import ALPRPipeline
from src.config import PipelineConfig
from src.core.ocr import OCRResult
from src.core.ocr_fusion import OCRFusionCandidate, fuse_candidates
from src.core.plate_detector import PlateDetection
from src.core.vehicle_detector import VehicleDetection
from src.video.customer_output import build_video_customer_payload
from src.video.renderer import plate_label, render_video_frame
from src.video.result_finalizer import finalize_video_tracks


def _row(track_id, formatted, *, status="ok", raw="72a16231", valid=True):
    return {"track_id": track_id, "vehicle": {"class_name": "car"}, "plate": {
        "status": status, "raw_text": raw, "normalized_text": raw.upper() if raw else None,
        "formatted": formatted, "format_valid": valid, "confidence": 0.9,
    }}


def _record(frame_index, vehicle_bbox=(5, 5, 100, 100), plate_bbox=None, track_id=2):
    return {"frame_index": frame_index,
            "vehicles": [{"track_id": track_id, "bbox_xyxy": list(vehicle_bbox),
                          "class_name": "car", "confidence": 0.9}],
            "plates": ([] if plate_bbox is None else [{
                "track_id": track_id, "bbox_xyxy": list(plate_bbox),
                "candidate_index": 0, "class_name": "vuong", "confidence": 0.91,
            }])}


def _draw_calls(monkeypatch):
    boxes, labels = [], []
    monkeypatch.setattr(cv2, "rectangle", lambda _im, p1, p2, color, _width:
                        boxes.append((p1, p2, color)))
    monkeypatch.setattr(cv2, "putText", lambda _im, text, *_args:
                        labels.append(text))
    return boxes, labels


def test_customer_plate_text_is_keyed_by_track_not_vehicle_order(monkeypatch):
    boxes, labels = _draw_calls(monkeypatch)
    record = _record(0, track_id=2, plate_bbox=(20, 30, 60, 50))
    record["vehicles"].insert(0, {"track_id": 1, "bbox_xyxy": [0, 0, 10, 10],
                                  "class_name": "car", "confidence": 0.8})
    record["plates"].insert(0, {"track_id": 1, "bbox_xyxy": [2, 2, 8, 7],
                                "candidate_index": 1, "class_name": "dai", "confidence": 0.7})
    render_video_frame(np.zeros((120, 120, 3), np.uint8), record,
                       {1: _row(1, "51B-775.69"), 2: _row(2, "72A-162.31")})
    assert labels == ["ID 1", "ID 2", "51B-775.69", "72A-162.31"]
    assert "vuong" not in " ".join(labels) and "dai" not in " ".join(labels)
    assert "P0" not in " ".join(labels)
    assert boxes[2][2] == boxes[3][2] == (0, 0, 255)


def test_debug_mode_can_show_candidate_detail(monkeypatch):
    _, labels = _draw_calls(monkeypatch)
    render_video_frame(np.zeros((120, 120, 3), np.uint8),
                       _record(0, plate_bbox=(20, 30, 60, 50)),
                       {2: _row(2, "72A-162.31")}, mode="debug")
    assert "P0" in labels[1] and "vuong" in labels[1]


def test_no_ocr_draws_red_box_without_fake_label(monkeypatch):
    boxes, labels = _draw_calls(monkeypatch)
    render_video_frame(np.zeros((120, 120, 3), np.uint8),
                       _record(0, plate_bbox=(20, 30, 60, 50)),
                       {2: _row(2, None, status="no_ocr", raw=None, valid=False)})
    assert len(boxes) == 2 and labels == ["ID 2"]


def test_invalid_format_label_is_only_normalized_raw():
    row = _row(2, None, status="unrecognized_format", raw="abc123", valid=False)
    assert plate_label(row) == "ABC123"
    assert row["plate"]["formatted"] is None


def test_frame_without_plate_does_not_carry_old_bbox(monkeypatch):
    boxes, labels = _draw_calls(monkeypatch)
    final = {2: _row(2, "72A-162.31")}
    frame = np.zeros((120, 120, 3), np.uint8)
    render_video_frame(frame, _record(0, plate_bbox=(20, 30, 60, 50)), final)
    boxes.clear(); labels.clear()
    render_video_frame(frame, _record(1, vehicle_bbox=(8, 6, 103, 101)), final)
    assert boxes == [((8, 6), (103, 101), (0, 200, 0))]
    assert labels == ["ID 2"]


def test_each_frame_uses_its_own_vehicle_and_plate_bbox(monkeypatch):
    boxes, labels = _draw_calls(monkeypatch)
    final = {2: _row(2, "72A-162.31")}
    frame = np.zeros((120, 120, 3), np.uint8)
    render_video_frame(frame, _record(0, (5, 5, 100, 100), (20, 30, 60, 50)), final)
    boxes.clear(); labels.clear()
    render_video_frame(frame, _record(1, (9, 6, 105, 101), (24, 31, 64, 51)), final)
    assert boxes[0][:2] == ((9, 6), (105, 101))
    assert boxes[1][:2] == ((24, 31), (64, 51))
    assert labels[1] == "72A-162.31"


def _finalize(raw=None, plate_count=1, confidence=0.8, duplicate=False):
    ids = [1, 2] if duplicate else [1]
    candidates = ([] if raw is None else [OCRFusionCandidate(
        1, 0, 1, raw, confidence, 0.9,
    )])
    fusion = fuse_candidates(candidates, track_ids=ids)
    history = {track_id: {"first_frame": 0, "last_frame": 2, "hits": 3,
                          "removal_reason": "duplicate" if track_id == 2 else None}
               for track_id in ids}
    votes = {track_id: Counter({"car": 3}) for track_id in ids}
    buffer = SimpleNamespace(selected=lambda _id: ())
    counts = {track_id: plate_count for track_id in ids}
    return finalize_video_tracks(history, votes, counts, buffer, fusion, PipelineConfig())


def test_core_finalizer_preserves_no_plate():
    assert _finalize(plate_count=0)[0]["plate"]["status"] == "no_plate"


def test_core_finalizer_preserves_no_ocr():
    assert _finalize()[0]["plate"]["status"] == "no_ocr"


def test_core_finalizer_preserves_unrecognized_format():
    row = _finalize("ABC123")[0]
    assert row["plate"]["status"] == "unrecognized_format"
    assert row["plate"]["formatted"] is None


def test_core_finalizer_preserves_low_confidence():
    row = _finalize("72a16231", confidence=0.0)[0]
    assert row["plate"]["status"] == "low_confidence"
    assert row["plate"]["formatted"] == "72A-162.31"


def test_duplicate_track_is_not_independent_final_result():
    rows = _finalize("72a16231", duplicate=True)
    assert [item["track_id"] for item in rows] == [1]


def test_customer_json_filters_only_at_serializer_boundary():
    core = {"status": "ok", "vehicles": [
        _row(1, "51B-775.69"),
        _row(2, None, status="no_ocr", raw=None, valid=False),
        _row(3, None, status="unrecognized_format", raw="abc123", valid=False),
    ]}
    public = build_video_customer_payload(core, min_confidence=0.5)
    assert len(core["vehicles"]) == 3
    assert [item["track_id"] for item in public["vehicles"]] == [1]


class _VehicleDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, _frame):
        self.calls += 1
        offset = self.calls - 1
        return [VehicleDetection(2, "car", 0.9, (10 + offset, 10, 150 + offset, 110))]


class _PlateDetector:
    def __init__(self):
        self.calls = 0

    def detect(self, _roi):
        self.calls += 1
        if self.calls == 2:
            return []
        return [PlateDetection(0, "vuong", 0.91, (30, 40, 110, 75))]


class _OCR:
    def __init__(self):
        self.calls = 0

    def recognize(self, _crop):
        self.calls += 1
        return OCRResult("72a16231", 0.9, (0.9,) * 8)


def test_production_two_pass_end_to_end_does_not_rerun_models(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (160, 120))
    assert writer.isOpened()
    rng = np.random.default_rng(2026)
    for _ in range(3):
        writer.write(rng.integers(50, 220, (120, 160, 3), dtype=np.uint8))
    writer.release()
    vehicle, plate, ocr = _VehicleDetector(), _PlateDetector(), _OCR()
    pipeline = ALPRPipeline(vehicle_detector=vehicle, plate_detector=plate, ocr_engine=ocr)
    result = pipeline.process_video(source, output=tmp_path / "result.json", save_annotated=True)
    assert result["summary"]["frames_read"] == result["summary"]["frames_rendered"] == 3
    assert vehicle.calls == plate.calls == 3  # pass 2 did not invoke either model
    assert ocr.calls == 1  # Top-K temporal gap retained one crop
    assert [row["track_id"] for row in result["vehicles"]] == [1]
    assert result["vehicles"][0]["plate"]["formatted"] == "72A-162.31"
    assert result["frames"][1]["plates"] == []
    assert result["frames"][0]["plates"][0]["bbox_xyxy"] != result["frames"][2]["plates"][0]["bbox_xyxy"]
    assert result["summary"]["successful_results"] == 1
    assert json.loads((tmp_path / "result.json").read_text())["vehicles"][0]["track_id"] == 1
    capture = cv2.VideoCapture(result["annotated_path"])
    assert capture.isOpened()
    frames = 0
    while capture.read()[0]:
        frames += 1
    capture.release()
    assert frames == 3


@pytest.mark.parametrize("top_k,min_crop_width,expected", [(2, 40, 2), (4, 81, 0)])
def test_video_topk_config_reaches_production_pipeline(tmp_path, top_k, min_crop_width, expected):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (160, 120))
    assert writer.isOpened()
    rng = np.random.default_rng(2026)
    for _ in range(3):
        writer.write(rng.integers(50, 220, (120, 160, 3), dtype=np.uint8))
    writer.release()

    plate = _PlateDetector()
    plate.detect = lambda _roi: [PlateDetection(0, "vuong", 0.91, (30, 40, 110, 75))]
    ocr = _OCR()
    base = PipelineConfig()
    topk = replace(base.video.topk, top_k=top_k, min_frame_gap=0,
                   min_crop_width=min_crop_width)
    config = replace(base, video=replace(base.video, topk=topk))
    pipeline = ALPRPipeline(config=config, vehicle_detector=_VehicleDetector(),
                            plate_detector=plate, ocr_engine=ocr)
    result = pipeline.process_video(source, output=tmp_path / "result.json",
                                    save_annotated=False)
    assert result["vehicles"][0]["evidence"]["topk_selected"] == expected
    assert ocr.calls == expected


def test_video_validation_tracking_quality_and_ownership_config_reach_production(
    tmp_path, monkeypatch,
):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"),
                             10.0, (160, 120))
    assert writer.isOpened()
    rng = np.random.default_rng(2026)
    for _ in range(3):
        writer.write(rng.integers(50, 220, (120, 160, 3), dtype=np.uint8))
    writer.release()

    base = PipelineConfig()
    video = replace(
        base.video,
        validation=replace(base.video.validation, min_width_pixels=9),
        tracking=replace(base.video.tracking, max_lost_seconds=0.60),
        quality=replace(base.video.quality, sharpness_reference=210.0),
        topk=replace(base.video.topk, top_k=2, min_frame_gap=0),
        ownership=replace(base.video.ownership, conflict_iou_threshold=0.66,
                          overlap_over_smaller_threshold=0.86),
    )
    seen = {}
    original_iter = video_pipeline.iter_tracked_video_frames
    original_resolve = video_pipeline.resolve_frame_plate_ownership
    original_score = video_pipeline.score_plate_quality

    def capture_iter(*args, **kwargs):
        seen["validation"] = kwargs["validation_config"]
        seen["tracking"] = kwargs["tracking_config"]
        yield from original_iter(*args, **kwargs)

    def capture_resolve(candidates, **kwargs):
        seen["ownership"] = kwargs
        return original_resolve(candidates, **kwargs)

    def capture_score(crop, confidence, quality_config):
        seen["quality"] = quality_config
        return original_score(crop, confidence, quality_config)

    monkeypatch.setattr(video_pipeline, "iter_tracked_video_frames", capture_iter)
    monkeypatch.setattr(video_pipeline, "resolve_frame_plate_ownership", capture_resolve)
    monkeypatch.setattr(video_pipeline, "score_plate_quality", capture_score)
    pipeline = ALPRPipeline(
        config=replace(base, video=video), vehicle_detector=_VehicleDetector(),
        plate_detector=_PlateDetector(), ocr_engine=_OCR(),
    )
    result = pipeline.process_video(source, output=tmp_path / "result.json",
                                    save_annotated=False)
    assert seen["validation"] is video.validation
    assert seen["tracking"] is video.tracking
    assert seen["quality"] is video.quality
    assert seen["ownership"] == {
        "conflict_iou_threshold": video.ownership.conflict_iou_threshold,
        "overlap_over_smaller_threshold": video.ownership.overlap_over_smaller_threshold,
    }
    assert result["vehicles"][0]["evidence"]["topk_selected"] == 2


def test_main_cli_routes_video_to_production_pipeline(tmp_path, monkeypatch):
    import main

    source = tmp_path / "sample.mp4"
    source.write_bytes(b"test-input")
    calls = []

    class FakePipeline:
        def __init__(self, **_kwargs):
            self.config = PipelineConfig()

        def process_video(self, path, **kwargs):
            calls.append((path, kwargs))
            return {"status": "ok", "summary": {
                "vehicles": 0, "vehicles_with_plate": 0,
                "vehicles_with_ocr": 0, "successful_results": 0,
            }, "input": {"type": "video"}, "vehicles": [],
                "output_path": str(tmp_path / "result.json")}

    monkeypatch.setattr(main, "ALPRPipeline", FakePipeline)
    assert main.main(["--input", str(source), "--output", str(tmp_path / "result.json")]) == 0
    assert len(calls) == 1 and calls[0][0] == source
    assert calls[0][1]["save_annotated"] is True
    assert calls[0][1]["output"] == tmp_path / "result/result.json"


def test_main_release_video_filters_only_customer_json(tmp_path, monkeypatch):
    import main

    source = tmp_path / "sample.mp4"
    source.write_bytes(b"test-input")
    release_dir = tmp_path / "release"
    core = {"status": "ok", "input": {"type": "video"}, "summary": {
        "vehicles": 2, "vehicles_with_plate": 1,
        "vehicles_with_ocr": 1, "successful_results": 1,
    }, "vehicles": [_row(1, "51B-775.69"),
                     _row(2, None, status="no_ocr", raw=None, valid=False)]}

    class FakePipeline:
        def __init__(self, **_kwargs):
            self.config = PipelineConfig()

        def process_video(self, _path, **kwargs):
            assert kwargs["debug"] is False
            assert kwargs["save_annotated"] is True
            return {**core, "output_path": str(kwargs["output"])}

    monkeypatch.setattr(main, "ALPRPipeline", FakePipeline)
    monkeypatch.setattr(main, "_release_paths", lambda _source:
                        (release_dir, release_dir / "sample.json"))
    assert main.main(["--input", str(source), "--release", "--debug"]) == 0
    public = json.loads((release_dir / "sample.json").read_text(encoding="utf-8"))
    assert [item["track_id"] for item in public["vehicles"]] == [1]
    assert len(core["vehicles"]) == 2
