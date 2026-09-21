"""Run the complete ONNX ALPR pipeline for one image or video."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from src.alpr_pipeline import ALPRPipeline, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image or video path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: output/<input-stem>_<YYYYMMDD_HHMMSS>.json)",
    )
    parser.add_argument("--save-annotated", action="store_true", help="Write annotated image/video")
    parser.add_argument("--save-topk-crops", action="store_true", help="Write retained plate crops")
    parser.add_argument("--debug", action="store_true", help="Print diagnostics and add per-candidate evidence")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.input
    output = args.output
    try:
        if not source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported input type: {suffix or '<none>'}")
        pipeline = ALPRPipeline(device=args.device, debug=args.debug)
        kwargs = {
            "output": output,
            "save_annotated": args.save_annotated,
            "save_topk_crops": args.save_topk_crops,
            "debug": args.debug,
        }
        if suffix in IMAGE_EXTENSIONS:
            result = pipeline.process_image(source, **kwargs)
        else:
            result = pipeline.process_video(source, **kwargs)
        summary = result["summary"]
        print(f"Processed {result['input']['type']}: {source}")
        print(
            f"Vehicles={summary['vehicles']}, with_plate={summary['vehicles_with_plate']}, "
            f"with_ocr={summary['vehicles_with_ocr']}, status_ok={summary['successful_results']}"
        )
        print(f"JSON: {result['output_path']}")
        if "annotated_path" in result:
            print(f"Annotated: {result['annotated_path']}")
        return 0
    except (FileNotFoundError, ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        if args.debug:
            traceback.print_exc()
        else:
            print(f"ALPR error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # ONNX/OpenCV errors should still be user readable
        if args.debug:
            traceback.print_exc()
        else:
            print(f"ALPR error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
