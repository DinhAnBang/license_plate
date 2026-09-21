"""Experimental A/B benchmark for MicroCharNet preprocessing hypotheses.

This intentionally creates one session per mode and is not the production V5
pipeline.  It exists so direct resize is never silently substituted for the
Ultralytics-compatible letterbox transform.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.microcharnet_ocr import MicroCharNetOCR


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("crops_dir", type=Path)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.70)
    args = parser.parse_args()
    paths = sorted(
        path
        for path in args.crops_dir.iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not paths:
        raise SystemExit(f"No image crops found in {args.crops_dir}")

    for mode in ("letterbox", "direct_resize"):
        ocr = MicroCharNetOCR(
            conf_threshold=args.conf,
            iou_threshold=args.iou,
            preprocess_mode=mode,
        )
        print(f"mode={mode} (EXPERIMENTAL A/B; session_init_count={ocr.session_init_count})")
        for path in paths:
            crop = cv2.imread(str(path))
            if crop is None:
                print(f"  {path.name}: unreadable")
                continue
            result = ocr.recognize(crop)
            print(
                f"  {path.name}: crop={crop.shape[1]}x{crop.shape[0]} "
                f"raw_text={result.text!r} confidence={result.confidence:.4f}"
            )
        timing = ocr.timing_totals
        print(
            "  averages_ms_per_crop: "
            f"preprocess={timing['preprocess_ms_per_crop']:.3f}, "
            f"inference={timing['inference_ms_per_crop']:.3f}, "
            f"decode={timing['decode_ms_per_crop']:.3f}, "
            f"total={timing['total_ms_per_crop']:.3f}"
        )


if __name__ == "__main__":
    main()
