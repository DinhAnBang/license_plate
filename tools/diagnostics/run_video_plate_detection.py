"""Run video vehicle tracking followed by plate-candidate detection."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import application_directory
from src.video.plate_stage import run_video_plate_detection
from src.video.tracking import VehicleTrackingConfig
from src.video.vehicle_validation import VehicleValidationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--vehicle-model",
        type=Path,
        default=application_directory() / "models" / "vehicle" / "yolo26n.onnx",
    )
    parser.add_argument(
        "--plate-model",
        type=Path,
        default=application_directory() / "models" / "plate" / "best.onnx",
    )
    parser.add_argument("--vehicle-confidence", type=float, default=0.10)
    parser.add_argument("--vehicle-min-confidence", type=float, default=0.50)
    parser.add_argument("--new-track-confidence", type=float, default=0.55)
    parser.add_argument("--vehicle-iou", type=float, default=0.45)
    parser.add_argument("--plate-confidence", type=float, default=0.25)
    parser.add_argument("--plate-iou", type=float, default=0.45)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-lost-seconds", type=float, default=0.5)
    parser.add_argument("--archive-seconds", type=float, default=10.0)
    parser.add_argument("--reid-similarity", type=float, default=0.80)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()

    result = run_video_plate_detection(
        args.input,
        args.output_dir,
        vehicle_model_path=args.vehicle_model,
        plate_model_path=args.plate_model,
        vehicle_confidence=args.vehicle_confidence,
        vehicle_iou=args.vehicle_iou,
        plate_confidence=args.plate_confidence,
        plate_iou=args.plate_iou,
        device=args.device,
        validation_config=VehicleValidationConfig(
            min_confidence=args.vehicle_min_confidence,
        ),
        tracking_config=VehicleTrackingConfig(
            high_confidence=args.vehicle_min_confidence,
            new_track_confidence=args.new_track_confidence,
            max_lost_seconds=args.max_lost_seconds,
            archive_seconds=args.archive_seconds,
            reidentification_similarity=args.reid_similarity,
        ),
        max_frames=args.max_frames,
    )
    summary = result["summary"]
    print(f"Frames: {summary['frames_read']}")
    print(f"Accepted vehicle detections: {summary['accepted_vehicle_detections']}")
    print(f"Raw plate candidates: {summary['raw_plate_candidates']}")
    print(f"Selected plate candidates: {summary['selected_plate_candidates']}")
    print(f"Rejected plate conflicts: {summary['rejected_plate_candidates']}")
    print(f"JSON: {result['output_path']}")
    if "annotated_path" in result:
        print(f"Annotated: {result['annotated_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
