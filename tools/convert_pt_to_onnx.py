"""Convert an Ultralytics YOLO .pt checkpoint to ONNX.

This utility is only for model conversion. The application can continue to
use the generated .onnx file with ONNX Runtime as before.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn


class HSigmoid(nn.Module):
    """Compatibility implementation for the custom h_sigmoid layer."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + 3.0) / 6.0


class HSwish(nn.Module):
    """Compatibility implementation for the custom h_swish layer."""

    def __init__(self, inplace: bool = True) -> None:
        super().__init__()
        self.sigmoid = HSigmoid(inplace=inplace)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    """Coordinate Attention layer used by microcharnet.pt.

    The checkpoint stores this class under
    ``ultralytics.nn.modules.block.CoordAtt``. The implementation here is
    registered under that name before Ultralytics loads the checkpoint.
    """

    def __init__(self, inp: int, oup: int, reduction: int = 32) -> None:
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = HSwish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        _, _, height, width = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.act(self.bn1(self.conv1(y)))

        x_h, x_w = torch.split(y, [height, width], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        attention_h = self.conv_h(x_h).sigmoid()
        attention_w = self.conv_w(x_w).sigmoid()
        return identity * attention_w * attention_h


def register_custom_layers() -> None:
    """Register layers required by checkpoints trained with custom modules."""

    import ultralytics.nn.modules.block as block

    # The names must match the module path serialized inside the checkpoint.
    block.CoordAtt = CoordAtt
    block.h_swish = HSwish
    block.h_sigmoid = HSigmoid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an Ultralytics YOLO .pt model to ONNX."
    )
    parser.add_argument(
        "--model",
        required=True,
        type=Path,
        help="Path to the input .pt model, for example models/best.pt",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Input image size used during export (default: 640).",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Export batch size (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help='Export device, for example "cpu" or "0" (default: cpu).',
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic ONNX input dimensions.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Do not simplify the exported ONNX graph.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=None,
        help="Optional ONNX opset version.",
    )
    parser.add_argument(
        "--nms",
        action="store_true",
        help="Include NMS in the exported model when supported.",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export an INT8-quantized ONNX model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        help="Calibration dataset YAML; required when --int8 is used.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        parser.error(f"Model file not found: {model_path}")

    if model_path.suffix.lower() != ".pt":
        parser.error(f"Input model must be a .pt file: {model_path}")

    if args.imgsz <= 0:
        parser.error("--imgsz must be greater than 0")

    if args.batch <= 0:
        parser.error("--batch must be greater than 0")

    data_path: Path | None = None
    if args.int8:
        if args.data is None:
            parser.error("--data is required when --int8 is used")
        data_path = args.data.expanduser().resolve()
        if not data_path.is_file():
            parser.error(f"Calibration YAML not found: {data_path}")

    try:
        register_custom_layers()
        from ultralytics import YOLO
    except ImportError as exc:
        print(
            "Ultralytics is not installed. Install it in the conversion "
            "environment with: python -m pip install ultralytics"
        )
        return 1

    export_args: dict[str, Any] = {
        "format": "onnx",
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "dynamic": args.dynamic,
        "simplify": not args.no_simplify,
    }

    if args.opset is not None:
        export_args["opset"] = args.opset

    if args.nms:
        export_args["nms"] = True

    if args.int8:
        export_args["quantize"] = 8
        export_args["data"] = str(data_path)

    print(f"Loading model: {model_path}")
    print("Export options:")
    for key, value in export_args.items():
        print(f"  {key}: {value}")

    try:
        model = YOLO(str(model_path))
        exported_path = model.export(**export_args)
    except Exception as exc:  # noqa: BLE001 - show a useful CLI error
        print(f"Export failed: {exc}")
        return 1

    print(f"Export completed: {exported_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
