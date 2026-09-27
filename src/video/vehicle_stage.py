"""Per-frame YOLO vehicle detection for the rebuilt video pipeline."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from ..config import VehicleConfig
from ..vehicle_detector import VehicleDetection, VehicleDetector
from .source import VideoReader


def detect_frame_vehicles(
    frame: np.ndarray,
    detector: VehicleDetector,
) -> tuple[VehicleDetection, ...]:
    """Run YOLO26 once on one frame and return its vehicle detections."""

    return tuple(detector.detect(frame))


def _providers_for_device(device: str) -> list[str] | None:
    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be auto, cpu or cuda")
    if device == "cpu":
        return ["CPUExecutionProvider"]
    if device == "cuda":
        import onnxruntime as ort

        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("CUDAExecutionProvider is not available")
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return None


def _draw_detections(
    frame: np.ndarray,
    detections: Sequence[VehicleDetection],
) -> None:
    for vehicle_index, detection in enumerate(detections):
        x1, y1, x2, y2 = detection.bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"{vehicle_index} {detection.class_name} {detection.confidence:.2f}"
        cv2.putText(
            frame,
            label,
            (max(0, x1), max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 200, 0),
            2,
            cv2.LINE_AA,
        )


def _detection_to_json(
    vehicle_index: int,
    detection: VehicleDetection,
) -> dict[str, object]:
    return {
        "vehicle_index": vehicle_index,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": round(float(detection.confidence), 6),
        "bbox_xyxy": list(detection.bbox),
    }


def run_video_vehicle_detection(
    source: str | Path,
    output_dir: str | Path,
    *,
    model_path: str | Path | None = None,
    confidence: float | None = None,
    iou: float | None = None,
    device: str = "auto",
    detector: VehicleDetector | None = None,
    save_annotated: bool = True,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Detect vehicles in every frame and write an inspectable result bundle."""

    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{source_path.stem}_vehicles.json"
    annotated_path = output_path / f"{source_path.stem}_vehicles_annotated.mp4"

    defaults = VehicleConfig()
    if detector is None:
        selected_model = Path(model_path) if model_path is not None else defaults.model
        selected_confidence = defaults.confidence if confidence is None else confidence
        selected_iou = defaults.iou if iou is None else iou
        detector = VehicleDetector(
            selected_model,
            confidence_threshold=selected_confidence,
            iou_threshold=selected_iou,
            providers=_providers_for_device(device),
        )

    records: list[dict[str, object]] = []
    class_counts: Counter[str] = Counter()
    frames_read = 0
    frames_with_detections = 0
    total_detections = 0
    writer: cv2.VideoWriter | None = None
    started = cv2.getTickCount()

    try:
        with VideoReader(source_path) as reader:
            metadata = reader.metadata
            if save_annotated:
                writer = cv2.VideoWriter(
                    str(annotated_path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    metadata.fps,
                    (metadata.width, metadata.height),
                )
                if not writer.isOpened():
                    raise OSError(f"Could not create annotated video: {annotated_path}")

            for video_frame in reader:
                if max_frames is not None and frames_read >= max_frames:
                    break
                detections = detect_frame_vehicles(video_frame.image, detector)
                if detections:
                    frames_with_detections += 1
                total_detections += len(detections)
                class_counts.update(item.class_name for item in detections)
                records.append(
                    {
                        "frame_index": video_frame.frame_index,
                        "timestamp_seconds": round(video_frame.timestamp_seconds, 6),
                        "detections": [
                            _detection_to_json(index, item)
                            for index, item in enumerate(detections)
                        ],
                    }
                )
                if writer is not None:
                    annotated = video_frame.image.copy()
                    _draw_detections(annotated, detections)
                    writer.write(annotated)
                frames_read += 1
    finally:
        if writer is not None:
            writer.release()

    if frames_read == 0:
        raise ValueError(f"Video contains no decodable frames: {source_path}")

    elapsed_seconds = (cv2.getTickCount() - started) / cv2.getTickFrequency()
    payload: dict[str, object] = {
        "status": "ok",
        "input": {
            "path": str(source_path),
            "type": "video",
            "fps": round(metadata.fps, 6),
            "frame_count_metadata": metadata.frame_count,
            "width": metadata.width,
            "height": metadata.height,
        },
        "summary": {
            "frames_read": frames_read,
            "frames_with_detections": frames_with_detections,
            "total_vehicle_detections": total_detections,
            "class_counts": dict(sorted(class_counts.items())),
        },
        "frames": records,
        "performance": {
            "total_seconds": round(elapsed_seconds, 6),
            "ms_per_frame": round(elapsed_seconds * 1000.0 / frames_read, 6),
            "detector_calls": int(getattr(detector, "detect_call_count", frames_read)),
            "model_path": str(getattr(detector, "model_path", model_path or "injected")),
        },
    }
    if save_annotated:
        payload["annotated_path"] = str(annotated_path)
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload["output_path"] = str(json_path)
    return payload


__all__ = ["detect_frame_vehicles", "run_video_vehicle_detection"]
