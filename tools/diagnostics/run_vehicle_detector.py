"""Run the Phase 1 vehicle detector on an image or video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.vehicle_detector import VehicleDetection, VehicleDetector


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}
COLORS = {
    2: (0, 200, 0),
    3: (255, 160, 0),
    5: (0, 180, 255),
    7: (0, 0, 255),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input image or video")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/vehicle/yolo26n.onnx"),
        help="Vehicle ONNX model",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Image JSON path (default: output/test_vehicle.json)",
    )
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--preview", action="store_true", help="Show result window")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def draw_detections(
    image: np.ndarray, detections: list[VehicleDetection]
) -> np.ndarray:
    result = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = COLORS[detection.class_id]
        label = f"{detection.class_name} {detection.confidence:.2f}"
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            result,
            (x1, label_top),
            (min(result.shape[1], x1 + text_width + 6), y1),
            color,
            -1,
        )
        cv2.putText(
            result,
            label,
            (x1 + 3, max(text_height, y1 - baseline - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return result


def print_detections(detections: list[VehicleDetection]) -> None:
    print(f"Detected vehicles: {len(detections)}")
    for index, detection in enumerate(detections):
        bbox = ",".join(str(value) for value in detection.bbox)
        print(f"\n[{index}] {detection.class_name}")
        print(f"    class_id={detection.class_id}")
        print(f"    confidence={detection.confidence:.4f}")
        print(f"    bbox=[{bbox}]")


def save_json(path: Path, detections: list[VehicleDetection]) -> None:
    payload = {
        "status": "ok",
        "input_type": "image",
        "vehicle_count": len(detections),
        "vehicles": [
            {
                "vehicle_index": index,
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "confidence": round(detection.confidence, 4),
                "bbox_xyxy": list(detection.bbox),
            }
            for index, detection in enumerate(detections)
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def process_image(
    input_path: Path,
    output_dir: Path,
    json_path: Path,
    detector: VehicleDetector,
    preview: bool,
) -> None:
    image = cv2.imread(str(input_path))
    if image is None:
        raise ValueError(f"Could not read image: {input_path}")

    detections = detector.detect(image)
    print_detections(detections)
    annotated = draw_detections(image, detections)

    output_dir.mkdir(parents=True, exist_ok=True)
    image_output = output_dir / f"{input_path.stem}_vehicle{input_path.suffix}"
    if not cv2.imwrite(str(image_output), annotated):
        raise OSError(f"Could not write result image: {image_output}")
    save_json(json_path, detections)
    print(f"\nSaved image: {image_output}")
    print(f"Saved JSON: {json_path}")

    if preview:
        cv2.imshow("Vehicle detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def process_video(
    input_path: Path,
    output_dir: Path,
    detector: VehicleDetector,
    preview: bool,
) -> None:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    if width <= 0 or height <= 0:
        capture.release()
        raise ValueError(f"Invalid video dimensions: {width}x{height}")
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    output_dir.mkdir(parents=True, exist_ok=True)
    video_output = output_dir / f"{input_path.stem}_vehicle.mp4"
    writer = cv2.VideoWriter(
        str(video_output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise OSError(f"Could not create result video: {video_output}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.detect(frame)
            annotated = draw_detections(frame, detections)
            writer.write(annotated)
            frame_index += 1

            if frame_index == 1 or frame_index % 30 == 0:
                print(
                    f"Frame {frame_index}: {len(detections)} vehicle(s)",
                    flush=True,
                )

            if preview:
                cv2.imshow("Vehicle detection - press q to stop", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        capture.release()
        writer.release()
        if preview:
            cv2.destroyAllWindows()

    print(f"Processed frames: {frame_index}")
    print(f"Saved video: {video_output}")


def main() -> None:
    args = parse_args()
    input_path = args.input
    if not input_path.is_file():
        raise FileNotFoundError(f"Input not found: {input_path}")

    detector = VehicleDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        debug=args.debug,
    )

    extension = input_path.suffix.lower()
    if extension in IMAGE_EXTENSIONS:
        json_path = args.json or args.output_dir / "test_vehicle.json"
        process_image(input_path, args.output_dir, json_path, detector, args.preview)
    elif extension in VIDEO_EXTENSIONS:
        if args.json is not None:
            print("--json is ignored for video input in Phase 1")
        process_video(input_path, args.output_dir, detector, args.preview)
    else:
        raise ValueError(f"Unsupported input extension: {extension or '<none>'}")


if __name__ == "__main__":
    main()
