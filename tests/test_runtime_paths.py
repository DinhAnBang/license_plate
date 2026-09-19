"""Source and PyInstaller one-file filesystem root contracts."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

from engine import AIPlateEngine
from engine.runtime_paths import get_app_root, get_model_path, get_output_root, get_resource_root
from tests.test_engine import FakeDetector, FakeOCR


ROOT = Path(__file__).resolve().parents[1]


def test_source_mode() -> None:
    with patch.object(sys, "frozen", False, create=True):
        assert get_resource_root() == ROOT
        assert get_app_root() == ROOT
        assert get_model_path("best.onnx") == ROOT / "models" / "best.onnx"
        assert get_output_root() == ROOT / "output" / "requests"

        engine = AIPlateEngine(detector=FakeDetector(), ocr=FakeOCR())
        assert engine.resource_root == ROOT
        assert engine.app_root == ROOT
        assert engine.detector_model == ROOT / "models" / "best.onnx"
        assert engine.ocr_model == ROOT / "models" / "OCR" / "microcharnet.onnx"
        assert engine.output_manager.root == ROOT / "output" / "requests"
        assert engine._resolve_input_path("input/images1.jpg") == ROOT / "input" / "images1.jpg"


def test_frozen_mode() -> None:
    frozen_resource = Path(r"C:\Temp\_MEI123").resolve()
    frozen_executable = Path(r"D:\AI_Plate\LicensePlateEngine.exe").resolve()
    frozen_app = frozen_executable.parent

    with (
        patch.object(sys, "frozen", True, create=True),
        patch.object(sys, "_MEIPASS", str(frozen_resource), create=True),
        patch.object(sys, "executable", str(frozen_executable)),
    ):
        assert get_resource_root() == frozen_resource
        assert get_app_root() == frozen_app
        assert get_model_path("best.onnx") == frozen_resource / "models" / "best.onnx"
        assert get_model_path("OCR", "microcharnet.onnx") == frozen_resource / "models" / "OCR" / "microcharnet.onnx"
        assert get_output_root() == frozen_app / "output" / "requests"
        assert not get_output_root().is_relative_to(frozen_resource)

        engine = AIPlateEngine(detector=FakeDetector(), ocr=FakeOCR())
        assert engine.resource_root == frozen_resource
        assert engine.app_root == frozen_app
        assert engine.detector_model == frozen_resource / "models" / "best.onnx"
        assert engine.ocr_model == frozen_resource / "models" / "OCR" / "microcharnet.onnx"
        assert engine.output_manager.root == frozen_app / "output" / "requests"
        assert engine._resolve_input_path("input/car.jpg") == frozen_app / "input" / "car.jpg"
        absolute_input = Path(r"D:\ParkingData\car001.jpg").resolve()
        assert engine._resolve_input_path(absolute_input) == absolute_input


def main() -> int:
    test_source_mode()
    test_frozen_mode()
    print("Source and frozen resource/app/output path separation: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
