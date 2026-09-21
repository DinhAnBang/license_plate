"""Inspect MicroCharNet checkpoint and ONNX output contracts before use."""

from __future__ import annotations

import argparse
import ast
from collections import Counter
from pathlib import Path
from typing import Any

import onnx
import onnxruntime as ort


def _shape(value_info: Any) -> list[int | str]:
    return [
        int(dimension.dim_value)
        if dimension.dim_value
        else str(dimension.dim_param or "?")
        for dimension in value_info.type.tensor_type.shape.dim
    ]


def _dtype(value_info: Any) -> str:
    return onnx.helper.tensor_dtype_to_string(value_info.type.tensor_type.elem_type)


def _parse_names(raw: str | None) -> dict[int, str] | None:
    if not raw:
        return None
    try:
        value = ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return None
    if isinstance(value, dict):
        try:
            return {int(key): str(char) for key, char in value.items()}
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        return {index: str(char) for index, char in enumerate(value)}
    return None


def _print_mapping(names: dict[int, str] | None) -> None:
    print(f"class_count: {len(names) if names is not None else None}")
    print(f"class_names: {names}")


def inspect_pt(model_path: Path) -> None:
    from microcharnet_export import register_custom_layers

    register_custom_layers()
    import torch
    import ultralytics
    from ultralytics import YOLO

    loaded = YOLO(str(model_path))
    model = loaded.model
    checkpoint = torch.load(str(model_path), map_location="cpu", weights_only=False)
    train_args = checkpoint.get("train_args", {}) if isinstance(checkpoint, dict) else {}
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        names = {int(key): str(value) for key, value in names.items()}
    head = model.model[-1]

    print(f"model_path: {model_path}")
    print(f"model_type: {type(model).__name__}")
    print(f"ultralytics_version: {ultralytics.__version__}")
    print(f"yaml_end2end: {getattr(model, 'yaml', {}).get('end2end')}")
    print(f"train_args_end2end: {train_args.get('end2end')}")
    print(f"model_end2end: {getattr(model, 'end2end', None)}")
    print(f"detect_head_type: {type(head).__name__}")
    print(f"detect_head_end2end: {getattr(head, 'end2end', None)}")
    print(f"one2one_branch: {getattr(head, 'one2one_cv2', None) is not None}")
    _print_mapping(names)


def _node_summary(model: onnx.ModelProto) -> str:
    final = model.graph.node[-8:]
    return "; ".join(
        f"{node.op_type}({','.join(node.input)} -> {','.join(node.output)})"
        for node in final
    )


def inspect_onnx(model_path: Path) -> None:
    model = onnx.load(str(model_path))
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers()
        else ["CPUExecutionProvider"]
    )
    session = ort.InferenceSession(str(model_path), providers=providers)
    metadata = {item.key: item.value for item in model.metadata_props}
    names = _parse_names(metadata.get("names"))
    op_counts = Counter(node.op_type for node in model.graph.node)

    print(f"model_path: {model_path}")
    print(f"onnxruntime_available_providers: {ort.get_available_providers()}")
    print(f"onnxruntime_selected_providers: {session.get_providers()}")
    print(f"input_count: {len(model.graph.input)}")
    for index, info in enumerate(model.graph.input):
        runtime = session.get_inputs()[index]
        print(f"input[{index}].name: {info.name}")
        print(f"input[{index}].shape: {_shape(info)}")
        print(f"input[{index}].dtype: {_dtype(info)} / runtime={runtime.type}")
    print(f"output_count: {len(model.graph.output)}")
    for index, info in enumerate(model.graph.output):
        runtime = session.get_outputs()[index]
        print(f"output[{index}].name: {info.name}")
        print(f"output[{index}].shape: {_shape(info)}")
        print(f"output[{index}].dtype: {_dtype(info)} / runtime={runtime.type}")

    output_shape = _shape(model.graph.output[0]) if model.graph.output else []
    output_format = "unsupported"
    output_semantics = "not inferred"
    if len(output_shape) == 3 and output_shape[0] == 1 and output_shape[2] == 6:
        output_format = "end2end"
        output_semantics = "[x1, y1, x2, y2, confidence, class_id]"
    elif (
        len(output_shape) == 3
        and names is not None
        and output_shape[1] == 4 + len(names)
    ):
        output_format = "raw"
        output_semantics = "[cx, cy, w, h, class_scores...] per detector location"

    print(f"ir_version: {model.ir_version}")
    print(f"producer: {model.producer_name} {model.producer_version}")
    print(f"metadata: {metadata}")
    print(f"embedded_character_mapping: {names}")
    print(f"output_format: {output_format}")
    print(f"output_semantics: {output_semantics}")
    print(f"op_counts: {dict(sorted(op_counts.items()))}")
    print(f"initializer_count: {len(model.graph.initializer)}")
    print(f"graph_node_count: {len(model.graph.node)}")
    print(f"final_nodes: {_node_summary(model)}")


def inspect(model_path: Path) -> None:
    model_path = model_path.expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if model_path.suffix.lower() == ".pt":
        inspect_pt(model_path)
    elif model_path.suffix.lower() == ".onnx":
        inspect_onnx(model_path)
    else:
        raise ValueError(f"Unsupported model extension: {model_path.suffix}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "models",
        nargs="*",
        type=Path,
        default=[
            Path("models/OCR/microcharnet.pt"),
            Path("models/OCR/microcharnet.onnx"),
        ],
    )
    args = parser.parse_args()
    for index, model_path in enumerate(args.models):
        if index:
            print("\n" + "=" * 72)
        inspect(model_path)


if __name__ == "__main__":
    main()

