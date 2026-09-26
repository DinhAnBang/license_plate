"""Run the complete ONNX ALPR pipeline for one image or video."""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

from src.alpr_pipeline import ALPRPipeline, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS
from src.customer_output import (
    build_customer_payload,
    write_customer_annotated_image,
    write_customer_json,
)
from src.paths import application_directory


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Image or video path")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON filename; CLI groups it under output/<filename-stem>/",
    )
    parser.add_argument("--save-annotated", action="store_true", help="Write annotated image/video")
    parser.add_argument("--save-topk-crops", action="store_true", help="Write retained plate crops")
    parser.add_argument("--debug", action="store_true", help="Print diagnostics and add per-candidate evidence")
    parser.add_argument(
        "--release",
        action="store_true",
        help="Customer/EXE output: write one annotated media file and one compact JSON beside the app",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def _application_directory() -> Path:
    """Return the directory containing the executable, or the source app."""

    return application_directory()


def _release_paths(source: Path) -> tuple[Path, Path]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = _application_directory() / f"{source.stem}_{timestamp}"
    return output_directory, output_directory / f"{source.stem}.json"


def _development_output_path(output: Path | None) -> Path | None:
    """Keep one CLI input's JSON and media artifacts in one named bundle."""

    if output is None or output.suffix.lower() != ".json":
        return output
    return output.parent / output.stem / output.name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    source = args.input
    output = args.output
    release_mode = bool(args.release or getattr(sys, "frozen", False))
    try:
        if not source.is_file():
            raise FileNotFoundError(f"Input file does not exist: {source}")
        suffix = source.suffix.lower()
        if suffix not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported input type: {suffix or '<none>'}")
        if release_mode and output is not None:
            raise ValueError("--release chooses its output folder automatically; omit --output")
        if release_mode and args.save_topk_crops:
            raise ValueError("--save-topk-crops is a development-only option and cannot be used with --release")

        pipeline = ALPRPipeline(device=args.device, debug=args.debug)
        release_directory = None
        if release_mode:
            release_directory, output = _release_paths(source)
            release_directory.mkdir(parents=True, exist_ok=True)
        else:
            output = _development_output_path(output)
        kwargs = {
            "output": output,
            "save_annotated": (
                args.save_annotated or release_mode
            ) and not (release_mode and suffix in IMAGE_EXTENSIONS),
            "save_topk_crops": args.save_topk_crops and not release_mode,
            "debug": args.debug,
        }
        if suffix in IMAGE_EXTENSIONS:
            result = pipeline.process_image(source, **kwargs)
        else:
            result = pipeline.process_video(source, **kwargs)
        if release_mode:
            assert release_directory is not None
            if suffix in IMAGE_EXTENSIONS:
                annotated_path = release_directory / f"{source.stem}_annotated.jpg"
                write_customer_annotated_image(
                    source,
                    annotated_path,
                    result,
                    min_confidence=pipeline.config.low_confidence_threshold,
                )
                result["annotated_path"] = str(annotated_path)
            write_customer_json(
                output,
                build_customer_payload(
                    result,
                    min_confidence=pipeline.config.low_confidence_threshold,
                ),
            )
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
