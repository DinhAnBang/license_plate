"""Compare intended PyTorch one-to-one output with the exported ONNX output."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from .microcharnet_export import register_custom_layers
from src.microcharnet_ocr import MicroCharNetOCR, OCRCharacter, OutputFormat, _iou


def _default_crops(root: Path) -> list[Path]:
    patterns = (
        "output/*_topk/track_5/*frame_76*.jpg",
        "output/*_topk/track_5/*frame_69*.jpg",
        "output/*_topk/track_5/*frame_64*.jpg",
        "output/*_topk/track_5/*frame_66*.jpg",
        "output/*_topk/track_5/*frame_74*.jpg",
    )
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(sorted(root.glob(pattern)))
    return paths


def _format_characters(characters: tuple[OCRCharacter, ...]) -> str:
    return " ".join(
        f"{item.char}(class={item.class_id},conf={item.confidence:.4f},box={item.bbox})"
        for item in characters
    )


def _mean_iou(
    first: tuple[OCRCharacter, ...], second: tuple[OCRCharacter, ...]
) -> float:
    if not first or not second:
        return 0.0
    values = []
    for left, right in zip(first, second):
        values.append(float(_iou(left.bbox, np.asarray([right.bbox], dtype=np.float32))[0]))
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pt", type=Path, default=Path("models/OCR/microcharnet.pt"))
    parser.add_argument("--onnx", type=Path, default=Path("models/OCR/microcharnet.onnx"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("crops", nargs="*", type=Path)
    args = parser.parse_args()

    crops = [path.expanduser().resolve() for path in args.crops]
    if not crops:
        crops = _default_crops(Path.cwd())
    if not crops:
        raise SystemExit("No regression crops found; pass crop paths explicitly.")
    if any(not path.is_file() for path in crops):
        raise SystemExit("One or more benchmark crops do not exist.")

    register_custom_layers()
    import torch
    from ultralytics import YOLO

    pt = YOLO(str(args.pt.expanduser().resolve()))
    pt_model = pt.model.float().eval()
    pt_model.end2end = True
    pt_model.model[-1].export = True

    onnx = MicroCharNetOCR(
        args.onnx,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
    )
    if onnx.output_format is not OutputFormat.END2END:
        raise SystemExit(
            f"Expected end-to-end ONNX [1,K,6], found {onnx.output_shape} "
            f"({onnx.output_format.value})."
        )

    print(f"pt={args.pt.resolve()}")
    print(f"onnx={args.onnx.resolve()}")
    print(f"onnx_output_shape={onnx.output_shape}")
    print(f"onnx_class_count={onnx.num_classes}")

    for crop_path in crops:
        image = cv2.imread(str(crop_path))
        if image is None:
            print(f"\n{crop_path.name}: unreadable")
            continue
        tensor, transform = onnx.preprocess_plate(image)
        with torch.inference_mode():
            pt_output = pt_model(torch.from_numpy(tensor)).detach().cpu().numpy()
        onnx_output = onnx.session.run(
            [onnx.output_name], {onnx.input_name: tensor}
        )[0]

        pt_result, pt_chars = onnx._decode_end2end(pt_output, transform)
        onnx_result, onnx_chars = onnx._decode_end2end(onnx_output, transform)
        print(f"\nCROP {crop_path.name} size={image.shape[1]}x{image.shape[0]}")
        print(f"PT text={pt_result.text!r} confidence={pt_result.confidence:.4f}")
        print(f"PT chars={_format_characters(pt_chars)}")
        print(f"ONNX text={onnx_result.text!r} confidence={onnx_result.confidence:.4f}")
        print(f"ONNX chars={_format_characters(onnx_chars)}")
        print(
            "semantic_compare="
            f"count_pt={len(pt_chars)} count_onnx={len(onnx_chars)} "
            f"mean_ordered_bbox_iou={_mean_iou(pt_chars, onnx_chars):.4f} "
            f"max_conf_diff={max((abs(a.confidence - b.confidence) for a, b in zip(pt_chars, onnx_chars)), default=0.0):.6f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
