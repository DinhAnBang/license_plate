"""Run the Phase 2 image processor on input/images1.jpg."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from core.detector import DetectorError, PlateDetector
from core.image_processor import ImageProcessingError, ImageProcessor


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path)
    args = parser.parse_args()
    project_dir = Path(__file__).resolve().parents[1]
    model_path = args.model or project_dir / "models" / "best.onnx"
    source_path = project_dir / "input" / "images1.jpg"

    print("====================================")
    print("IMAGE PROCESSOR")
    print("====================================")
    print(f"Source: {source_path.relative_to(project_dir)}")

    try:
        detector = PlateDetector(
            model_path=model_path,
            conf_threshold=0.5,
            iou_threshold=0.45,
            providers=["CPUExecutionProvider"],
        )
        processor = ImageProcessor(
            detector,
            output_dir=project_dir / "output",
            project_root=project_dir,
        )
        started = time.perf_counter()
        result = processor.process(source_path)
        total_ms = (time.perf_counter() - started) * 1000.0
    except (DetectorError, ImageProcessingError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Size: {result['size']['w']}x{result['size']['h']}")
    print(f"\nPlates: {result['count']}")
    for plate in result["plates"]:
        print(f"\n#{plate['id']}")
        print(f"Conf: {plate['conf']:.6f}")
        print(f"Box: {plate['box']}")
        print(f"Crop: {plate['crop']}")
        print(f"OCR: raw={plate['raw_text']} text={plate['text']} conf={plate['ocr_conf']:.4f}")

    json_path = project_dir / "output" / "json" / f"{source_path.stem}.json"
    saved = json.loads(json_path.read_text(encoding="utf-8"))
    assert saved["status"] == "success" and saved["input_type"] == "image"
    assert saved["count"] == len(saved["plates"])
    for plate in saved["plates"]:
        assert {"plate_text", "detection_confidence", "ocr_confidence", "crop_path"} == set(plate)
        assert Path(plate["crop_path"]).is_file()
    assert processor.ocr.session_creation_count == 1
    assert processor.ocr_calls == len(saved["plates"])
    print(f"JSON: PASS | OCR calls: {processor.ocr_calls} | OCR sessions: 1")
    print(f"Image total: {total_ms:.2f} ms | Average OCR: {processor.ocr_total_ms / max(1, processor.ocr_calls):.2f} ms")

    print("\n------------------------------------")
    print("Image:")
    print(f"output/images/{source_path.stem}_result.jpg")
    print("\nJSON:")
    print(f"output/json/{source_path.stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
