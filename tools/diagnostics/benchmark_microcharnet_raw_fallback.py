"""Compare the old raw class-aware result with the V5.1 raw fallback."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.microcharnet_ocr import (
    MicroCharNetOCR,
    OCRCharacter,
    _RawCharacter,
    _class_agnostic_nms,
    _group_and_sort_characters,
    _iou,
)


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


def _debug_class_aware_nms(
    detections: list[_RawCharacter], iou_threshold: float
) -> list[_RawCharacter]:
    """Reference-only reproduction of the pre-V5.1 production bug."""

    kept: list[_RawCharacter] = []
    for class_id in sorted({item.class_id for item in detections}):
        remaining = sorted(
            [item for item in detections if item.class_id == class_id],
            key=lambda item: item.confidence,
            reverse=True,
        )
        while remaining:
            best = remaining.pop(0)
            kept.append(best)
            if not remaining:
                continue
            boxes = np.asarray([item.box for item in remaining], dtype=np.float32)
            overlaps = _iou(best.box, boxes)
            remaining = [
                item
                for item, overlap in zip(remaining, overlaps)
                if float(overlap) <= iou_threshold
            ]
    kept.sort(key=lambda item: item.confidence, reverse=True)
    return kept


def _raw_candidates(
    ocr: MicroCharNetOCR, output: np.ndarray, transform
) -> list[_RawCharacter]:
    predictions = output[0].T.astype(np.float32, copy=False)
    candidates: list[_RawCharacter] = []
    for row in predictions:
        scores = row[4:]
        class_id = int(np.argmax(scores))
        confidence = float(scores[class_id])
        if confidence < ocr.conf_threshold:
            continue
        center_x, center_y, width, height = [float(value) for value in row[:4]]
        if width <= 0.0 or height <= 0.0:
            continue
        box = (
            float(np.clip((center_x - width / 2.0 - transform.pad_x) / transform.scale, 0, transform.source_width)),
            float(np.clip((center_y - height / 2.0 - transform.pad_y) / transform.scale_y, 0, transform.source_height)),
            float(np.clip((center_x + width / 2.0 - transform.pad_x) / transform.scale, 0, transform.source_width)),
            float(np.clip((center_y + height / 2.0 - transform.pad_y) / transform.scale_y, 0, transform.source_height)),
        )
        if box[2] > box[0] and box[3] > box[1]:
            candidates.append(_RawCharacter(ocr.class_names[class_id], class_id, confidence, box))
    return candidates


def _text(kept: list[_RawCharacter]) -> str:
    chars = [
        OCRCharacter(item.char, item.class_id, item.confidence, tuple(round(v) for v in item.box))
        for item in kept
    ]
    return "".join(item.char for line in _group_and_sort_characters(chars) for item in line)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-model",
        type=Path,
        default=Path("models/OCR/backup/microcharnet_raw.onnx"),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("crops", nargs="*", type=Path)
    args = parser.parse_args()

    crops = [path.expanduser().resolve() for path in args.crops] or _default_crops(Path.cwd())
    if not crops or any(not path.is_file() for path in crops):
        raise SystemExit("No valid raw fallback regression crops found.")

    raw = MicroCharNetOCR(args.raw_model, args.conf, args.iou)
    if raw.output_shape != (1, 40, 1024):
        raise SystemExit(f"Expected the backed-up raw model, found {raw.output_shape}")

    for path in crops:
        image = cv2.imread(str(path))
        tensor, transform = raw.preprocess_plate(image)
        output = raw.session.run([raw.output_name], {raw.input_name: tensor})[0]
        candidates = _raw_candidates(raw, output, transform)
        aware = _debug_class_aware_nms(candidates, args.iou)
        agnostic = _class_agnostic_nms(candidates, args.iou)
        production_result, _ = raw._decode_raw(output, transform)
        print(
            f"{path.name}: old_class_aware={_text(aware)!r} "
            f"raw_class_agnostic={_text(agnostic)!r} "
            f"production_raw={production_result.text!r} "
            f"aware_count={len(aware)} agnostic_count={len(agnostic)} "
            f"removed={len(aware) - len(agnostic)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
