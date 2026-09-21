"""Run the V3 plate detector on one vehicle crop image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.plate_detector import PlateDetection, PlateDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--model", type=Path, default=Path("models/plate/best.onnx")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.45)
    parser.add_argument("--json", action="store_true", help="Also save JSON")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def draw_plates(image, plates: list[PlateDetection]):
    annotated = image.copy()
    for plate in plates:
        x1, y1, x2, y2 = plate.bbox
        label = f"{plate.class_name} | {plate.confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            annotated,
            label,
            (x1, max(18, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
    return annotated


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input image not found: {args.input}")
    image = cv2.imread(str(args.input))
    if image is None:
        raise ValueError(f"Could not read image: {args.input}")

    detector = PlateDetector(
        model_path=args.model,
        confidence_threshold=args.confidence,
        iou_threshold=args.iou,
        debug=args.debug,
    )
    plates = detector.detect(image)
    print(f"Detected plates: {len(plates)}")
    for index, plate in enumerate(plates):
        bbox = ",".join(str(value) for value in plate.bbox)
        print(f"\n[{index}]")
        print(f"class_id={plate.class_id}") 
        print(f"class_name={plate.class_name}")
        print(f"confidence={plate.confidence:.4f}")
        print(f"bbox=[{bbox}]")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    image_output = args.output_dir / f"{args.input.stem}_plate{args.input.suffix}"
    if not cv2.imwrite(str(image_output), draw_plates(image, plates)):
        raise OSError(f"Could not write output image: {image_output}")
    print(f"\nSaved image: {image_output}")

    if args.json:
        json_output = args.output_dir / f"{args.input.stem}_plate.json"
        payload = {
            "status": "ok",
            "input_type": "vehicle_roi",
            "plate_count": len(plates),
            "plates": [
                {
                    "plate_index": index,
                    "class_id": plate.class_id,
                    "class_name": plate.class_name,
                    "confidence": round(plate.confidence, 4),
                    "bbox_xyxy": list(plate.bbox),
                }
                for index, plate in enumerate(plates)
            ],
        }
        json_output.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Saved JSON: {json_output}")

    if args.preview:
        cv2.imshow("Plate detection", draw_plates(image, plates))
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
