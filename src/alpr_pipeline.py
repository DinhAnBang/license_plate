"""Production image/video ALPR pipeline using the three ONNX models."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import onnxruntime as ort

from .config import PipelineConfig
from .image_pipeline import run_image
from .microcharnet_ocr import MicroCharNetOCR
from .plate_detector import PlateDetector
from .vehicle_detector import VehicleDetector
from .video_pipeline import run_video


IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})
VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv"})


class ALPRPipeline:
    """Load detector/OCR sessions once, then process independent inputs."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        device: str = "auto",
        debug: bool = False,
        vehicle_detector: VehicleDetector | None = None,
        plate_detector: PlateDetector | None = None,
        ocr_engine: MicroCharNetOCR | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.debug = debug
        if device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be auto, cpu or cuda")
        available = ort.get_available_providers()
        if device == "cuda" and "CUDAExecutionProvider" not in available:
            raise RuntimeError("CUDAExecutionProvider is not available")
        providers = (["CPUExecutionProvider"] if device == "cpu" else
                     ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else None)
        for model in (self.config.vehicle.model, self.config.plate.model, self.config.ocr.model):
            if Path(model).suffix.lower() != ".onnx":
                raise ValueError(f"Production model must be ONNX: {model}")

        def load_detector(factory: Any, *args: Any) -> Any:
            try:
                return factory(*args, debug=debug, providers=providers)
            except Exception:
                if device != "auto" or "CUDAExecutionProvider" not in available:
                    raise
                return factory(*args, debug=debug, providers=["CPUExecutionProvider"])

        self.vehicle_detector = vehicle_detector or load_detector(
            VehicleDetector, self.config.vehicle.model,
            self.config.vehicle.confidence, self.config.vehicle.iou,
        )
        self.plate_detector = plate_detector or load_detector(
            PlateDetector, self.config.plate.model,
            self.config.plate.confidence, self.config.plate.iou,
        )
        self.ocr_engine = ocr_engine or MicroCharNetOCR(
            self.config.ocr.model, conf_threshold=self.config.ocr.confidence,
            iou_threshold=self.config.ocr.iou, providers=providers,
        )
        if device == "cuda" and not (vehicle_detector or plate_detector or ocr_engine):
            selected = (
                self.vehicle_detector.session.get_providers()[0],
                self.plate_detector.session.get_providers()[0],
                self.ocr_engine.provider,
            )
            if any(provider != "CUDAExecutionProvider" for provider in selected):
                raise RuntimeError(f"CUDA execution was requested but providers are {selected}")

    @property
    def models(self) -> dict[str, str]:
        return {
            "vehicle": str(self.config.vehicle.model),
            "plate": str(self.config.plate.model),
            "ocr": str(self.config.ocr.model),
        }

    @property
    def session_init_count(self) -> dict[str, int]:
        return {
            "vehicle": getattr(self.vehicle_detector, "session_init_count", 1),
            "plate": getattr(self.plate_detector, "session_init_count", 1),
            "ocr": self.ocr_engine.session_init_count,
        }

    def _output_path(self, source: Path, output: str | Path | None) -> Path:
        if output is not None:
            return Path(output)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return Path("output") / f"{source.stem}_{timestamp}.json"

    @staticmethod
    def _artifact_stem(source: Path, output_path: Path, output: str | Path | None) -> str:
        """Use one shared prefix for default JSON and media artifacts."""

        return output_path.stem if output is None else source.stem

    def process_image(
        self, path: str | Path, *, output: str | Path | None = None,
        save_annotated: bool = False, save_topk_crops: bool = False,
        debug: bool | None = None,
    ) -> dict[str, Any]:
        return run_image(
            self, path, output=output, save_annotated=save_annotated,
            save_topk_crops=save_topk_crops, debug=debug,
        )

    def process_video(
        self, path: str | Path, *, output: str | Path | None = None,
        save_annotated: bool = False, save_topk_crops: bool = False,
        debug: bool | None = None,
    ) -> dict[str, Any]:
        return run_video(
            self, path, output=output, save_annotated=save_annotated,
            save_topk_crops=save_topk_crops, debug=debug,
        )


__all__ = ["ALPRPipeline", "IMAGE_EXTENSIONS", "VIDEO_EXTENSIONS"]
