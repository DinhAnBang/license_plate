"""Run Module 4: validated YOLO26 vehicle detections plus stable tracking."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.paths import application_directory
from src.video.tracking_stage import run_video_vehicle_tracking
from src.video.tracking import VehicleTrackingConfig
from src.video.vehicle_validation import VehicleValidationConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--model",
        type=Path,
        default=application_directory() / "models" / "vehicle" / "yolo26n.onnx",
    )
    parser.add_argument("--confidence", type=float, default=0.10)
    parser.add_argument("--min-confidence", type=float, default=0.50)
    parser.add_argument("--new-track-confidence", type=float, default=0.55)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    parser.add_argument("--max-lost-seconds", type=float, default=0.5)
    parser.add_argument("--archive-seconds", type=float, default=10.0)
    parser.add_argument("--reid-similarity", type=float, default=0.80)
    parser.add_argument("--wide-edge-ratio", type=float, default=0.98)
    parser.add_argument("--wide-edge-min-confidence", type=float, default=0.50)
    args = parser.parse_args()

    result = run_video_vehicle_tracking(
        args.input,
        args.output_dir,
        model_path=args.model,
        confidence=args.confidence,
        iou=args.iou,
        device=args.device,
        validation_config=VehicleValidationConfig(
            min_confidence=args.min_confidence,
            wide_edge_bbox_ratio=args.wide_edge_ratio,
            wide_edge_min_confidence=args.wide_edge_min_confidence,
        ),
        tracking_config=VehicleTrackingConfig(
            high_confidence=args.min_confidence,
            new_track_confidence=args.new_track_confidence,
            max_lost_seconds=args.max_lost_seconds,
            archive_seconds=args.archive_seconds,
            reidentification_similarity=args.reid_similarity,
        ),
    )
    print(f"Frames: {result['summary']['frames_read']}")
    print(f"Accepted vehicle detections: {result['summary']['accepted_vehicle_detections']}")
    print(f"Unique track IDs: {result['summary']['unique_track_ids_created']}")
    print(f"Reidentified: {result['summary']['reidentified_count']}")
    print(f"JSON: {result['output_path']}")
    if "annotated_path" in result:
        print(f"Annotated: {result['annotated_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
