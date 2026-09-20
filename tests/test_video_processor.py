"""Run Phase 3, 4, or 5 video processing on the supplied input video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from core.detector import DetectorError, PlateDetector
from core.quality import PlateQualityEvaluator
from core.result_writer import VideoResultWriter
from core.tracker import PlateTracker
from core.video_processor import VideoProcessingError, VideoProcessor


def validate_output(output_path: Path, expected: dict[str, object]) -> tuple[bool, str]:
    """Reopen the output and check its basic video metadata and readability."""

    cap = cv2.VideoCapture(str(output_path))
    if not cap.isOpened():
        cap.release()
        return False, "output video could not be opened"
    try:
        width = int(round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        height = int(round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        output_frame_count = int(round(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        ok, frame = cap.read()
    finally:
        cap.release()

    if not ok or frame is None or frame.size == 0:
        return False, "output video has no readable first frame"
    if (width, height) != (expected["width"], expected["height"]):
        return False, f"resolution is {width}x{height}, expected {expected['width']}x{expected['height']}"
    if fps <= 0.0:
        return False, "output FPS is invalid"
    if output_frame_count != expected["processed_frames"]:
        return False, (
            f"output frame count is {output_frame_count}, "
            f"expected {expected['processed_frames']}"
        )
    return True, f"opened, {width}x{height}, {fps:.2f} FPS, {output_frame_count} frames"


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Phase 3, 4, or 5 video processing.")
    parser.add_argument(
        "--tracking",
        action="store_true",
        help="enable Phase 4 IoU tracking and write a _tracked.mp4 output",
    )
    parser.add_argument(
        "--quality",
        action="store_true",
        help="enable Phase 5 best-crop quality evaluation; tracking is enabled automatically",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="write and validate the official Phase 6 video JSON",
    )
    parser.add_argument("--model", type=Path, help="Detector ONNX path for testing.")
    args = parser.parse_args()
    if args.quality or args.json:
        args.quality = True
        args.tracking = True

    project_dir = Path(__file__).resolve().parents[1]
    model_path = args.model or project_dir / "models" / "best.onnx"
    source_path = (
        project_dir
        / "input"
        / "1788115615491-5334873298567571839-5334873298567571839_txDxN1HN.mp4"
    )

    print("====================================")
    print("VIDEO PROCESSOR")
    print("====================================")
    print(f"Source: {source_path.relative_to(project_dir)}")
    print(f"Tracking: {'ON' if args.tracking else 'OFF'}")
    print(f"Quality: {'ON' if args.quality else 'OFF'}")
    print(f"Official JSON: {'ON' if args.json else 'OFF'}")

    try:
        detector = PlateDetector(
            model_path=model_path,
            conf_threshold=0.7,
            iou_threshold=0.45,
            providers=["CPUExecutionProvider"],
        )
        tracker = PlateTracker(iou_threshold=0.25, max_missed=10) if args.tracking else None
        quality_evaluator = (
            PlateQualityEvaluator(
                confidence_weight=0.30,
                sharpness_weight=0.35,
                brightness_weight=0.15,
                size_weight=0.20,
                sharpness_reference=500.0,
                brightness_target=127.5,
                reference_area=12_000.0,
            )
            if args.quality
            else None
        )
        result_writer = (
            VideoResultWriter(
                output_dir=project_dir / "output" / "json",
                project_root=project_dir,
            )
            if args.json
            else None
        )
        processor = VideoProcessor(
            detector,
            output_dir=project_dir / "output",
            project_root=project_dir,
            tracker=tracker,
            tracker_mode="legacy" if tracker is not None else "disabled",
            quality_evaluator=quality_evaluator,
            result_writer=result_writer,
        )
        result = processor.process(source_path)
    except (DetectorError, VideoProcessingError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print("\n------------------------------------")
    print(f"Resolution: {result['width']}x{result['height']}")
    print(f"FPS: {result['fps']:.2f}")
    print(f"Frames: {result['frame_count']}")
    print(f"Duration: {result['duration']:.2f} s")
    print(f"Processed frames: {result['processed_frames']}")
    print(f"Total frame detections: {result['total_detections']}")
    print("(total frame detections is not the number of unique vehicles/plates)")
    print(f"Average detect: {result['avg_detect_ms']:.2f} ms/frame")
    print(f"Detector FPS: {result['detector_fps']:.2f}")
    print(f"Total processing: {result['total_processing_time']:.2f} s")
    print(f"Processing FPS: {result['processing_fps']:.2f}")

    if args.tracking:
        print(f"Average tracking: {result['avg_tracking_ms']:.4f} ms/frame")
        print(f"Total tracks: {result['total_tracks']}")
        print(f"Tracks with 1 hit: {result['tracks_with_1_hit']}")
        print(f"Tracks with <=2 hits: {result['tracks_with_le_2_hits']}")
        print(f"Tracks with >=3 hits: {result['tracks_with_ge_3_hits']}")
        longest_track = result["longest_track"]
        if longest_track is None:
            print("Longest track: none")
        else:
            print("Longest track:")
            print(f"  ID: {longest_track['track_id']}")
            print(f"  First frame: {longest_track['first_frame']}")
            print(f"  Last frame: {longest_track['last_frame']}")
            print(f"  Hits: {longest_track['hits']}")
        print("\nID  First  Last  Hits")
        for track in result["tracks"]:
            print(
                f"{track['track_id']:>2}  {track['first_frame']:>5}  "
                f"{track['last_frame']:>4}  {track['hits']:>4}"
            )

    if args.quality:
        print("\nQuality configuration:")
        print("  Confidence weight: 0.30")
        print("  Sharpness weight: 0.35")
        print("  Brightness weight: 0.15")
        print("  Size weight: 0.20")
        print("  Sharpness reference: 500.00")
        print("  Brightness target: 127.50")
        print("  Reference area: 12000.00 px^2")
        print(f"Quality candidates: {result['quality_candidates']}")
        print(f"Invalid crops: {result['invalid_crops']}")
        print(f"Tracks with best crop: {result['tracks_with_best_crop']}")
        print(f"Tracks without valid crop: {result['tracks_without_valid_crop']}")
        print(f"Average quality: {result['avg_quality_ms']:.4f} ms/frame")
        if result["min_best_quality"] is None:
            print("Best quality: no valid best crops")
        else:
            print(
                f"Best quality: min={result['min_best_quality']:.4f}, "
                f"max={result['max_best_quality']:.4f}, "
                f"average={result['avg_best_quality']:.4f}"
            )
        print(
            "\nID | First | Last | Hits | BestFrame | Conf | Quality | Sharp | "
            "SharpRaw | Bright | BrightRaw | Size | WxH | Area | Crop"
        )
        for best in result["best_crops"]:
            print(
                f"{best['track_id']} | {best['first_frame']} | {best['last_frame']} | "
                f"{best['hits']} | {best['best_frame']} | {best['conf']:.3f} | "
                f"{best['quality']:.3f} | {best['sharpness']:.3f} | "
                f"{best['sharpness_raw']:.1f} | {best['brightness']:.3f} | "
                f"{best['brightness_raw']:.1f} | {best['size']:.3f} | "
                f"{best['crop_width']}x{best['crop_height']} | {best['crop_area']} | "
                f"{best['crop']}"
            )

    output_path = project_dir / Path(str(result["output"]))
    valid, validation_message = validate_output(output_path, result)
    print("\n------------------------------------")
    print(f"Output: {result['output']}")
    print(f"Output validation: {'OK - ' if valid else 'FAILED - '}{validation_message}")
    if args.json:
        json_path = project_dir / Path(str(result["json"]))
        print(f"JSON: {result['json']}")
        print("\nOfficial JSON:")
        print(json_path.read_text(encoding="utf-8"))
        official = json.loads(json_path.read_text(encoding="utf-8"))
        assert official["status"] == "success" and official["input_type"] == "video"
        assert official["count"] == len(official["plates"])
        assert official["processing"]["frames"] == result["processed_frames"]
        assert official["processing"]["average_frame_ms"] == round(
            official["processing"]["total_ms"] / official["processing"]["frames"], 6
        )
        for plate in official["plates"]:
            assert set(plate) == {
                "plate_text", "detection_confidence", "ocr_confidence",
                "first_detected_frame", "last_detected_frame",
                "first_detected_time_sec", "last_detected_time_sec",
                "best_frame_index", "best_frame_time_sec", "crop_path",
            }
            assert plate["first_detected_frame"] <= plate["best_frame_index"] <= plate["last_detected_frame"]
            assert plate["first_detected_time_sec"] <= plate["best_frame_time_sec"] <= plate["last_detected_time_sec"]
            assert Path(plate["crop_path"]).is_file()
        assert result["ocr_calls"] <= len(official["plates"]) * processor.top_k
        assert processor.ocr.session_creation_count == 1
        print("\nTOP-K OCR REPORT")
        for track in result["ocr_report"]:
            print(f"Track ID: {track['track_id']} | Candidates retained: {track['candidates_retained']}")
            for index, candidate in enumerate(track["candidates"], start=1):
                print(f"  Candidate {index}: frame={candidate['frame']} quality={candidate['quality']:.4f} det_conf={candidate['det_conf']:.4f} raw={candidate['raw_text']} text={candidate['text']} ocr_conf={candidate['ocr_conf']:.4f} vote_weight={candidate['vote_weight']:.4f}")
            print(f"  Winner: text={track['winner']['text']} frame={track['winner']['frame']} weight={track['winner']['weight']:.4f}")
        print(f"JSON: PASS | Detector sessions: 1 | OCR sessions: 1 | OCR calls: {result['ocr_calls']}")
        print(f"Average OCR: {result['avg_ocr_ms']:.2f} ms")
    print("Status: OK" if valid else "Status: FAILED")
    return 0 if valid and (not args.json or json_path.is_file()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
