"""Run the Phase 3 video processor on the supplied input video."""

from __future__ import annotations

from pathlib import Path

import cv2

from core.detector import DetectorError, PlateDetector
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
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "best.onnx"
    source_path = (
        project_dir
        / "input"
        / "1788115615491-5334873298567571839-5334873298567571839_txDxN1HN.mp4"
    )

    print("====================================")
    print("VIDEO PROCESSOR")
    print("====================================")
    print(f"Source: {source_path.relative_to(project_dir)}")

    try:
        detector = PlateDetector(
            model_path=model_path,
            conf_threshold=0.7,
            iou_threshold=0.45,
            providers=["CPUExecutionProvider"],
        )
        processor = VideoProcessor(
            detector,
            output_dir=project_dir / "output",
            project_root=project_dir,
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

    output_path = project_dir / Path(str(result["output"]))
    valid, validation_message = validate_output(output_path, result)
    print("\n------------------------------------")
    print(f"Output: {result['output']}")
    print(f"Output validation: {'OK - ' if valid else 'FAILED - '}{validation_message}")
    print("Status: OK" if valid else "Status: FAILED")
    return 0 if valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
