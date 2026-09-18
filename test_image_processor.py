"""Run the Phase 2 image processor on input/images1.jpg."""

from __future__ import annotations

from pathlib import Path

from core.detector import DetectorError, PlateDetector
from core.image_processor import ImageProcessingError, ImageProcessor


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    model_path = project_dir / "models" / "best.onnx"
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
        result = processor.process(source_path)
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

    print("\n------------------------------------")
    print("Image:")
    print(f"output/images/{source_path.stem}_result.jpg")
    print("\nJSON:")
    print(f"output/json/{source_path.stem}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
