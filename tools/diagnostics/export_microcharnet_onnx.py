"""Export MicroCharNet through the intended Ultralytics end-to-end branch."""

from __future__ import annotations

import argparse
from pathlib import Path

from .microcharnet_export import register_custom_layers


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/OCR/microcharnet.pt"),
    )
    parser.add_argument("--imgsz", type=int, default=128)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Keep the unsimplified graph for exporter debugging.",
    )
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"Model not found: {model_path}")
    if args.imgsz <= 0 or args.batch <= 0:
        parser.error("--imgsz and --batch must be positive")

    register_custom_layers()
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    print(f"ultralytics_version={__import__('ultralytics').__version__}")
    print(f"model_type={type(model.model).__name__}")
    print(f"before_export_end2end={getattr(model.model, 'end2end', None)}")
    print(
        "before_export_one2one_branch="
        f"{getattr(model.model.model[-1], 'one2one_cv2', None) is not None}"
    )

    # This must be explicit.  Leaving nms=None preserves the serialized raw
    # one-to-many branch even when the checkpoint was trained end-to-end.
    export_kwargs = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dynamic": False,
        "simplify": not args.no_simplify,
        "opset": args.opset,
        "nms": False,
    }
    print(f"export_kwargs={export_kwargs}")
    exported_path = model.export(**export_kwargs)
    print(f"exported_path={exported_path}")
    print(f"after_export_end2end={getattr(model.model, 'end2end', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
