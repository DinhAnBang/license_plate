"""Inspect an ONNX model with ONNX Runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort


def format_shape(shape: Any) -> str:
    """Format static and symbolic ONNX dimensions without losing information."""
    return str(list(shape) if shape is not None else shape)


def make_probe_shape(shape: Any) -> list[int]:
    """Replace dynamic dimensions with common probe values for a metadata check."""
    defaults = [1, 3, 640, 640]
    result: list[int] = []
    for index, dimension in enumerate(shape):
        if isinstance(dimension, int) and dimension > 0:
            result.append(dimension)
        else:
            result.append(defaults[index] if index < len(defaults) else 1)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect an ONNX model using ONNX Runtime.")
    parser.add_argument("model", nargs="?", default="models/best.onnx")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model does not exist: {model_path}")
        return 1

    print("====================================")
    print("ONNX MODEL INSPECTION")
    print("====================================")
    print(f"Model: {model_path}")
    print(f"Available providers: {ort.get_available_providers()}")

    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
    except Exception as exc:
        print(f"ERROR: ONNX Runtime could not load the model: {exc}")
        return 1

    print(f"Session providers: {session.get_providers()}")
    print("\nInputs:")
    for value in session.get_inputs():
        print(f"  name: {value.name}")
        print(f"  shape: {format_shape(value.shape)}")
        print(f"  type: {value.type}")

    print("\nOutputs:")
    for value in session.get_outputs():
        print(f"  name: {value.name}")
        print(f"  shape: {format_shape(value.shape)}")
        print(f"  type: {value.type}")

    inputs = session.get_inputs()
    if not inputs:
        print("\nNo inputs found; cannot run a probe inference.")
        return 0

    input_meta = inputs[0]
    if input_meta.type != "tensor(float)":
        print(f"\nProbe skipped: unsupported probe input type {input_meta.type!r}.")
        return 0

    try:
        probe_shape = make_probe_shape(input_meta.shape)
        probe = np.zeros(probe_shape, dtype=np.float32)
        probe_outputs = session.run(None, {input_meta.name: probe})
    except Exception as exc:
        print(f"\nProbe inference failed: {exc}")
        return 0

    print("\nProbe output details:")
    for meta, output in zip(session.get_outputs(), probe_outputs):
        array = np.asarray(output)
        flattened = array.reshape(-1)
        sample = flattened[: min(10, flattened.size)].tolist()
        print(f"  name: {meta.name}")
        print(f"  actual shape: {list(array.shape)}")
        print(f"  dtype: {array.dtype}")
        print(f"  sample values: {sample}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
