"""One-process, sequential ONNX engine for image and video requests."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np

from core.detector import PlateDetector
from core.config import (
    BYTE_HIGH_THRESHOLD,
    DETECTOR_CONF_THRESHOLD,
    DETECTOR_NMS_IOU_THRESHOLD,
    QUALITY_BRIGHTNESS_TARGET,
    QUALITY_BRIGHTNESS_WEIGHT,
    QUALITY_CONFIDENCE_WEIGHT,
    QUALITY_REFERENCE_AREA,
    QUALITY_SHARPNESS_REFERENCE,
    QUALITY_SHARPNESS_WEIGHT,
    QUALITY_SIZE_WEIGHT,
    RECOGNITION_TOP_K,
)
from core.image_processor import ImageProcessor
from core.ocr import MicroCharNetOCR
from core.quality import PlateQualityEvaluator
from core.result_writer import VideoResultWriter
from core.video_processor import VideoProcessor

from .protocol import ProtocolError, error_response, validate_request
from .output_manager import RequestOutputManager
from .runtime_paths import get_app_root, get_output_root, get_resource_root


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
        resource_root: str | Path | None = None,
        tracker_mode: str = "sort",
        t5_enabled: bool = True,
        vietnam_plate_validation_enabled: bool | None = None,
        vietnam_plate_correction_enabled: bool | None = None,
        final_invalid_filter_enabled: bool | None = None,
        final_duplicate_merge_enabled: bool | None = None,
        overlap_duplicate_merge_enabled: bool | None = None,
    ) -> None:
        if tracker_mode not in {"legacy", "sort", "byte"}:
            raise ValueError("tracker_mode must be 'legacy', 'sort', or 'byte'")
        self.resource_root = Path(resource_root).resolve() if resource_root is not None else get_resource_root()
        self.app_root = Path(project_root).resolve() if project_root is not None else get_app_root()
        # Keep this alias for the Phase 1-10 processors, where it means the
        # persistent root used to resolve inputs and generated artifacts.
        self.project_root = self.app_root
        self.detector_model = self._resolve_model_path(detector_model)
        self.ocr_model = self._resolve_model_path(ocr_model)
        self.detector = detector
        self.ocr = ocr
        self.write_json = write_json
        self.tracker_mode = tracker_mode
        self.t5_enabled = bool(t5_enabled)
        self.t5_flags = {
            "vietnam_plate_validation_enabled": vietnam_plate_validation_enabled,
            "vietnam_plate_correction_enabled": vietnam_plate_correction_enabled,
            "final_invalid_filter_enabled": final_invalid_filter_enabled,
            "final_duplicate_merge_enabled": final_duplicate_merge_enabled,
            "overlap_duplicate_merge_enabled": overlap_duplicate_merge_enabled,
        }
        self.output_manager = RequestOutputManager(self.app_root)
        self.image_processor: ImageProcessor | None = None
        self.video_processor: VideoProcessor | None = None
        self.state = self.STOPPED
        self.startup_ms = 0.0
        self.detector_warmed = False
        self.ocr_warmed = False

    def _resolve_model_path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.resource_root / path).resolve()

    def _resolve_input_path(self, value: str | Path) -> Path:
        path = Path(value)
        return (path if path.is_absolute() else self.app_root / path).resolve()

    def startup(self) -> None:
        if self.state != self.STOPPED:
            raise RuntimeError(f"Engine cannot start from state {self.state}")
        started = time.perf_counter()
        self.state = self.STARTING
        try:
            LOGGER.info("Resource root: %s", self.resource_root)
            LOGGER.info("App root: %s", self.app_root)
            LOGGER.info("Detector model: %s", self.detector_model)
            LOGGER.info("OCR model: %s", self.ocr_model)
            LOGGER.info("Output root: %s", get_output_root(self.app_root))
            if self.detector is None:
                if not self.detector_model.is_file():
                    raise FileNotFoundError(f"Detector model not found: {self.detector_model}")
                self.detector = PlateDetector(
                    self.detector_model,
                    conf_threshold=DETECTOR_CONF_THRESHOLD,
                    iou_threshold=DETECTOR_NMS_IOU_THRESHOLD,
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
                quality_evaluator=PlateQualityEvaluator(
                    confidence_weight=QUALITY_CONFIDENCE_WEIGHT,
                    sharpness_weight=QUALITY_SHARPNESS_WEIGHT,
                    brightness_weight=QUALITY_BRIGHTNESS_WEIGHT,
                    size_weight=QUALITY_SIZE_WEIGHT,
                    sharpness_reference=QUALITY_SHARPNESS_REFERENCE,
                    brightness_target=QUALITY_BRIGHTNESS_TARGET,
                    reference_area=QUALITY_REFERENCE_AREA,
                ),
                result_writer=VideoResultWriter(
                    output_dir=self.project_root / "output" / "json",
                    project_root=self.project_root,
                ),
                ocr=self.ocr,
                top_k=RECOGNITION_TOP_K,
                write_json=self.write_json,
                tracker_mode=self.tracker_mode,
                t5_enabled=self.t5_enabled,
                **self.t5_flags,
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
        probe_shape = getattr(
            self.detector,
            "inference_input_shape",
            [dim if isinstance(dim, int) and dim > 0 else 1 for dim in shape],
        )
        probe = np.zeros(probe_shape, dtype=np.float32)
        names = [item["name"] for item in self.detector.output_info]
        self.detector.session.run(names, {self.detector.input_name: probe})

    def _warm_up_ocr(self) -> None:
        assert self.ocr is not None
        probe = np.zeros((1, 3, self.ocr.input_height, self.ocr.input_width), dtype=np.float32)
        self.ocr.session.run([self.ocr.output_name], {self.ocr.input_name: probe})

    def handle_request(self, value: Any) -> dict[str, Any]:
        request_id = value.get("id") if isinstance(value, dict) else None
        input_type = value.get("type") if isinstance(value, dict) else None
        try:
            request = validate_request(value)
        except ProtocolError as exc:
            return error_response(request_id, exc.code, exc.message, input_type)

        if self.state != self.READY:
            return error_response(
                request_id,
                "ENGINE_NOT_READY",
                f"Engine state is {self.state}.",
                input_type,
            )

        action = request["action"]
        if action == "ping":
            return {"id": request_id, "status": "ok", "state": self.READY}
        if action == "shutdown":
            self.state = self.SHUTTING_DOWN
            return {"id": request_id, "status": "ok", "state": self.SHUTTING_DOWN}

        source = self._resolve_input_path(request["path"])
        if not source.is_file():
            return error_response(
                request_id,
                "INPUT_NOT_FOUND",
                f"Input file not found: {source}",
                request["type"],
            )

        self.state = self.PROCESSING
        output_paths = None
        try:
            output_paths = self.output_manager.paths_for(request_id, request["type"])
            self.output_manager.prepare(output_paths)
            assert self.detector is not None
            if request["type"] == "image":
                self.detector.conf_threshold = DETECTOR_CONF_THRESHOLD
                assert self.image_processor is not None
                internal_result = self.image_processor.process(source, output_paths=output_paths)
                result = internal_result["production_result"]
            else:
                self.detector.conf_threshold = BYTE_HIGH_THRESHOLD
                assert self.video_processor is not None
                video_result = self.video_processor.process(source, output_paths=output_paths)
                result = video_result["official_result"]
            return result
        except Exception as exc:
            LOGGER.exception("Request %r failed", request_id)
            if output_paths is not None:
                try:
                    self.output_manager.cleanup(output_paths)
                except Exception:
                    LOGGER.exception("Could not clean partial output for request %r", request_id)
            return error_response(request_id, "PROCESSING_ERROR", str(exc), request["type"])
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
