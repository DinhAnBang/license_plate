"""Run only Module 2: YOLO26 vehicle detection on a video."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import application_directory
from src.video.vehicle_stage import run_video_vehicle_detection


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=application_directory() / "models" / "vehicle" / "yolo26n.onnx",
    )
    parser.add_argument("--confidence", type=float, default=0.50)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    result = run_video_vehicle_detection(
        args.input,
        args.output_dir,
        model_path=args.model,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
    )
    print(f"Frames: {result['summary']['frames_read']}")
    print(f"Vehicle boxes: {result['summary']['total_vehicle_detections']}")
    print(f"JSON: {result['output_path']}")
    if "annotated_path" in result:
        print(f"Annotated: {result['annotated_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
