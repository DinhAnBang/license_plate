"""Run the Phase 1 detector on input/images1.jpg and save an annotated image."""

from __future__ import annotations

from pathlib import Path

import cv2

from core.detector import DetectorError, PlateDetector


def main() -> int:
    project_dir = Path(__file__).resolve().parents[1]
    model_path = project_dir / "models" / "best.onnx"
    image_path = project_dir / "input" / "images1.jpg"
    output_path = project_dir / "output" / "images" / "images1_result.jpg"

    print("====================================")
    print("ONNX LICENSE PLATE DETECTOR")
    print("====================================")
    print(f"Model: {model_path.relative_to(project_dir)}")

    try:
        detector = PlateDetector(
            model_path=model_path,
            conf_threshold=0.5,
            iou_threshold=0.45,
            providers=["CPUExecutionProvider"],
        )
    except (DetectorError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1

    print(f"Provider: {', '.join(detector.providers)}")
    print("\nInput:")
    print(f"  name: {detector.input_name}")
    print(f"  shape: {detector.input_shape}")
    print(f"  type: {detector.input_type}")
    print("\nOutput:")
    for output in detector.output_info:
        print(f"  name: {output['name']}")
        print(f"  shape: {output['shape']}")
        print(f"  type: {output['type']}")

    if not image_path.is_file():
        print(f"ERROR: image does not exist: {image_path}")
        return 1
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        print(f"ERROR: OpenCV could not read image: {image_path}")
        return 1

    height, width = image.shape[:2]
    print("\n------------------------------------")
    print(f"Image: {image_path.relative_to(project_dir)}")
    print(f"Size: {width}x{height}")

    try:
        detections, timings = detector.detect_with_timing(image)
    except DetectorError as exc:
        print(f"ERROR: {exc}")
        return 1

    annotated = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = detection["box"]
        confidence = detection["conf"]
        label = f"Plate {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        label_top = max(0, y1 - text_height - baseline - 4)
        cv2.rectangle(
            annotated,
            (x1, label_top),
            (min(width, x1 + text_width + 6), y1),
            (0, 255, 0),
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 3, max(text_height + 1, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), annotated):
        print(f"ERROR: OpenCV could not save output image: {output_path}")
        return 1

    print(f"\nDetections: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        print(f"\n#{index}")
        print(f"Conf: {detection['conf']:.4f}")
        print(f"Box: {detection['box']}")

    print("\n------------------------------------")
    print(f"Preprocess: {timings['preprocess_ms']:.2f} ms")
    print(f"Inference: {timings['inference_ms']:.2f} ms")
    print(f"Postprocess: {timings['postprocess_ms']:.2f} ms")
    print(f"Total: {timings['total_ms']:.2f} ms")
    print("\nSaved:")
    print(output_path.relative_to(project_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
