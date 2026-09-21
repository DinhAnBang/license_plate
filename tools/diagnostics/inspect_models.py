"""Inspect the input/output metadata of an ONNX model."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="Path to an ONNX model")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = args.model.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    print(f"model path: {model_path}")
    print(f"execution providers available: {ort.get_available_providers()}")
    print(f"execution providers selected: {session.get_providers()}")

    for index, model_input in enumerate(session.get_inputs()):
        print(f"input[{index}] name: {model_input.name}")
        print(f"input[{index}] shape: {model_input.shape}")
        print(f"input[{index}] datatype: {model_input.type}")

    for index, model_output in enumerate(session.get_outputs()):
        print(f"output[{index}] name: {model_output.name}")
        print(f"output[{index}] shape: {model_output.shape}")
        print(f"output[{index}] datatype: {model_output.type}")


if __name__ == "__main__":
    main()
