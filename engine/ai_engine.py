"""One-process, sequential ONNX engine for image and video requests."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.detector import PlateDetector
from core.image_processor import ImageProcessor
from core.ocr import MicroCharNetOCR
from core.quality import PlateQualityEvaluator
from core.result_writer import VideoResultWriter
from core.tracker import PlateTracker
from core.video_processor import VideoProcessor

from .protocol import ProtocolError, error_response, validate_request


LOGGER = logging.getLogger(__name__)


class AIPlateEngine:
    """Own one detector and one OCR session for the entire engine lifetime."""

    STARTING = "starting"
    READY = "ready"
    PROCESSING = "processing"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"

    def __init__(
        self,
        project_root: str | Path | None = None,
        detector_model: str | Path = "models/best.onnx",
        ocr_model: str | Path = "models/OCR/microcharnet.onnx",
        *,
        detector: PlateDetector | None = None,
        ocr: MicroCharNetOCR | None = None,
        write_json: bool = True,
    ) -> None:
        self.project_root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
        self.detector_model = self._resolve_path(detector_model)
        self.ocr_model = self._resolve_path(ocr_model)
        self.detector = detector
        self.ocr = ocr
        self.write_json = write_json
        self.image_processor: ImageProcessor | None = None
        self.video_processor: VideoProcessor | None = None
        self.state = self.STOPPED
        self.startup_ms = 0.0
        self.detector_warmed = False
        self.ocr_warmed = False

    def _resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.project_root / path).resolve()

    def startup(self) -> None:
        if self.state != self.STOPPED:
            raise RuntimeError(f"Engine cannot start from state {self.state}")
        started = time.perf_counter()
        self.state = self.STARTING
        try:
            if self.detector is None:
                if not self.detector_model.is_file():
                    raise FileNotFoundError(f"Detector model not found: {self.detector_model}")
                self.detector = PlateDetector(
                    self.detector_model,
                    conf_threshold=0.5,
                    iou_threshold=0.45,
                    providers=["CPUExecutionProvider"],
                )
            if self.ocr is None:
                if not self.ocr_model.is_file():
                    raise FileNotFoundError(f"OCR model not found: {self.ocr_model}")
                self.ocr = MicroCharNetOCR(self.ocr_model)

            self._warm_up_detector()
            self.detector_warmed = True
            self._warm_up_ocr()
            self.ocr_warmed = True

            self.image_processor = ImageProcessor(
                self.detector,
                output_dir=self.project_root / "output",
                project_root=self.project_root,
                ocr=self.ocr,
                write_json=self.write_json,
            )
            self.video_processor = VideoProcessor(
                self.detector,
                output_dir=self.project_root / "output",
                project_root=self.project_root,
                tracker=PlateTracker(iou_threshold=0.25, max_missed=10),
                quality_evaluator=PlateQualityEvaluator(
                    confidence_weight=0.30,
                    sharpness_weight=0.35,
                    brightness_weight=0.15,
                    size_weight=0.20,
                    sharpness_reference=500.0,
                    brightness_target=127.5,
                    reference_area=12_000.0,
                ),
                result_writer=VideoResultWriter(
                    output_dir=self.project_root / "output" / "json",
                    project_root=self.project_root,
                ),
                ocr=self.ocr,
                top_k=3,
                write_json=self.write_json,
            )
            self.startup_ms = (time.perf_counter() - started) * 1000.0
            self.state = self.READY
        except Exception:
            self.shutdown()
            raise

    def _warm_up_detector(self) -> None:
        assert self.detector is not None
        shape = self.detector.input_shape
        if len(shape) != 4 or self.detector.input_type != "tensor(float)":
            raise RuntimeError(f"Unsupported detector warm-up input: {shape}")
        probe_shape = [dim if isinstance(dim, int) and dim > 0 else 1 for dim in shape]
        probe = np.zeros(probe_shape, dtype=np.float32)
        names = [item["name"] for item in self.detector.output_info]
        self.detector.session.run(names, {self.detector.input_name: probe})

    def _warm_up_ocr(self) -> None:
        assert self.ocr is not None
        probe = np.zeros((1, 3, self.ocr.input_height, self.ocr.input_width), dtype=np.float32)
        self.ocr.session.run([self.ocr.output_name], {self.ocr.input_name: probe})

    def handle_request(self, value: Any) -> dict[str, Any]:
        request_id = value.get("id") if isinstance(value, dict) else None
        try:
            request = validate_request(value)
        except ProtocolError as exc:
            return error_response(request_id, exc.code, exc.message)

        if self.state != self.READY:
            return error_response(request_id, "ENGINE_NOT_READY", f"Engine state is {self.state}.")

        action = request["action"]
        if action == "ping":
            return {"id": request_id, "status": "ok", "state": self.READY}
        if action == "shutdown":
            self.state = self.SHUTTING_DOWN
            return {"id": request_id, "status": "ok", "state": self.SHUTTING_DOWN}

        source = self._resolve_path(request["path"])
        if not source.is_file():
            return error_response(request_id, "INPUT_NOT_FOUND", f"Input file not found: {source}")

        self.state = self.PROCESSING
        try:
            assert self.detector is not None
            if request["type"] == "image":
                self.detector.conf_threshold = 0.5
                assert self.image_processor is not None
                result = self.image_processor.process(source)
            else:
                self.detector.conf_threshold = 0.7
                assert self.video_processor is not None
                video_result = self.video_processor.process(source)
                result = video_result["official_result"]
            return {"id": request_id, "status": "ok", "result": result}
        except Exception as exc:
            LOGGER.exception("Request %r failed", request_id)
            return error_response(request_id, "PROCESSING_ERROR", str(exc))
        finally:
            self.state = self.READY

    def shutdown(self) -> None:
        if self.state == self.STOPPED:
            return
        self.state = self.SHUTTING_DOWN
        self.image_processor = None
        self.video_processor = None
        self.detector = None
        self.ocr = None
        self.state = self.STOPPED
