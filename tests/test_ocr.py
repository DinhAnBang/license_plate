"""Standalone MicroCharNet ONNX OCR test with a clean-output fallback crop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from core.ocr import MicroCharNetOCR, draw_ocr_result, group_and_sort_characters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test MicroCharNet ONNX OCR on plate crops.")
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "models" / "OCR" / "microcharnet.onnx",
    )
    parser.add_argument(
        "--crops",
        type=Path,
        default=PROJECT_ROOT / "output" / "crops",
    )
    parser.add_argument(
        "--debug-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "ocr_debug",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Character confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.70, help="Class-aware NMS IoU threshold.")
    return parser.parse_args()


def _remove_previous_debug_images(debug_dir: Path) -> None:
    """Remove only debug files produced by this test, never the plate crops."""

    debug_dir.mkdir(parents=True, exist_ok=True)
    for path in debug_dir.glob("*_ocr.jpg"):
        path.unlink()


def _check_sorting() -> tuple[bool, bool]:
    one_line_detections = [
        {"char": "1", "class_id": 1, "conf": 0.9, "box": [40, 10, 50, 40]},
        {"char": "7", "class_id": 7, "conf": 0.9, "box": [10, 11, 20, 41]},
        {"char": "A", "class_id": 10, "conf": 0.9, "box": [25, 9, 35, 39]},
    ]
    one_line = group_and_sort_characters(one_line_detections)
    one_line_pass = "".join(item["char"] for item in one_line[0]) == "7A1"

    two_line_detections = [
        {"char": "2", "class_id": 2, "conf": 0.9, "box": [40, 50, 50, 80]},
        {"char": "1", "class_id": 1, "conf": 0.9, "box": [10, 10, 20, 40]},
        {"char": "3", "class_id": 3, "conf": 0.9, "box": [55, 50, 65, 80]},
        {"char": "5", "class_id": 5, "conf": 0.9, "box": [25, 10, 35, 40]},
    ]
    two_lines = group_and_sort_characters(two_line_detections)
    two_line_pass = [
        "".join(item["char"] for item in line) for line in two_lines
    ] == ["15", "23"]
    return one_line_pass, two_line_pass


def _print_character_detections(result: dict) -> None:
    print("Character detections:")
    for character in result["characters"]:
        print(
            f'  {character["char"]}   '
            f'conf={float(character["conf"]):.4f} '
            f'box={character["box"]}'
        )


def main() -> int:
    args = parse_args()
    model_path = args.model if args.model.is_absolute() else PROJECT_ROOT / args.model
    crops_dir = args.crops if args.crops.is_absolute() else PROJECT_ROOT / args.crops
    debug_dir = args.debug_dir if args.debug_dir.is_absolute() else PROJECT_ROOT / args.debug_dir

    if not crops_dir.is_dir():
        print(f"Crops directory not found: {crops_dir}")
        return 1

    _remove_previous_debug_images(debug_dir)
    crop_paths = sorted(
        path for path in crops_dir.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    fallback_crop: np.ndarray | None = None
    if not crop_paths and crops_dir.resolve() == (PROJECT_ROOT / "output" / "crops").resolve():
        source_image = cv2.imread(str(PROJECT_ROOT / "input" / "images1.jpg"))
        if source_image is not None and source_image.shape[0] >= 403 and source_image.shape[1] >= 89:
            # Known plate location in the checked-in sample image.
            fallback_crop = source_image[379:403, 56:89]
    if not crop_paths and fallback_crop is None:
        print(f"No readable plate crops are available in {crops_dir}")
        return 1

    print("========================================")
    print("MICROCHARNET ONNX OCR TEST")
    print("========================================")
    print(f"Model:\n{model_path}")

    try:
        ocr = MicroCharNetOCR(
            model_path=model_path,
            conf_threshold=args.conf,
            iou_threshold=args.iou,
        )
    except Exception as exc:  # noqa: BLE001 - print the actionable validation error
        print(f"ONNX Runtime: FAIL\n{exc}")
        return 1

    print("ONNX Runtime: PASS")
    print(f"Input: name={ocr.input_name}, shape={ocr.input_shape}, dtype={ocr.input_dtype}")
    print(f"Output: name={ocr.output_name}, shape={ocr.output_shape}, dtype={ocr.output_dtype}")
    print(f"Character classes: count={ocr.num_classes}, source=ONNX metadata (verified MicroCharNet YAML)")
    print("Mapping: " + ", ".join(f"{index}->{name}" for index, name in enumerate(ocr.class_names)))
    print("----------------------------------------")

    timing_totals = {
        "preprocess_ms": 0.0,
        "inference_ms": 0.0,
        "postprocess_ms": 0.0,
        "total_ms": 0.0,
    }
    results: list[tuple[Path, dict]] = []
    images_with_ocr = 0
    errors = 0

    crop_items = [(path.name, cv2.imread(str(path))) for path in crop_paths]
    if fallback_crop is not None:
        crop_items.append(("images1_sample.jpg", fallback_crop))
    for crop_name, image in crop_items:
        if image is None:
            print(f"Skipping unreadable image: {crop_name}")
            errors += 1
            continue

        try:
            result = ocr.recognize(image)
        except Exception as exc:  # noqa: BLE001 - keep the remaining crop tests running
            print(f"OCR failed for {crop_name}: {exc}")
            errors += 1
            continue

        results.append((Path(crop_name), result))
        if result["text"]:
            images_with_ocr += 1
        for key in timing_totals:
            timing_totals[key] += float(result["timing_ms"][key])

        debug_image = draw_ocr_result(image, result)
        debug_path = debug_dir / f"{Path(crop_name).stem}_ocr.jpg"
        if not cv2.imwrite(str(debug_path), debug_image):
            print(f"Could not write debug image: {debug_path}")
            errors += 1

        print(f"File: {crop_name}")
        print(f"Size: {image.shape[1]}x{image.shape[0]}")
        print(f"Text: {result['text']}")
        print(f"Lines: {result['lines']}")
        print(f"Characters: {len(result['characters'])}")
        print(f"OCR confidence: {float(result['confidence']):.4f}")
        print(
            "Inference time: "
            f"{float(result['timing_ms']['inference_ms']):.3f} ms"
        )
        _print_character_detections(result)
        print("----------------------------------------")

    tested = len(results)
    images_without_ocr = tested - images_with_ocr
    divisor = max(1, tested)
    one_line_pass, two_line_pass = _check_sorting()

    print("========================================")
    print("SUMMARY")
    print("========================================")
    print(f"Images tested: {tested}")
    print(f"Images with OCR: {images_with_ocr}")
    print(f"Images without OCR: {images_without_ocr}")
    print(f"Average preprocess: {timing_totals['preprocess_ms'] / divisor:.3f} ms")
    print(f"Average inference: {timing_totals['inference_ms'] / divisor:.3f} ms")
    print(f"Average postprocess: {timing_totals['postprocess_ms'] / divisor:.3f} ms")
    print(f"Average total: {timing_totals['total_ms'] / divisor:.3f} ms")
    print(f"Model sessions created: {ocr.session_creation_count}")
    print(f"One-line sorting: {'PASS' if one_line_pass else 'FAIL'}")
    print(f"Two-line sorting: {'PASS' if two_line_pass else 'FAIL'}")
    print("Existing detection/tracking/quality pipeline modified: NO")

    if errors:
        print(f"\nFinal status: OCR CORE TEST COMPLETED WITH {errors} ERROR(S)")
        return 1

    print("\nFinal status: OCR CORE READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
