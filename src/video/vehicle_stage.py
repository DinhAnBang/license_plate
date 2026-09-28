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
from .vehicle_validation import (
    ValidatedVehicleDetection,
    VehicleValidationConfig,
    rejection_reason_counts,
    validate_frame_detections,
)


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


def _draw_validated_detections(
    frame: np.ndarray,
    detections: Sequence[ValidatedVehicleDetection],
    *,
    show_rejected: bool = False,
) -> None:
    for vehicle_index, item in enumerate(detections):
        if not item.accepted and not show_rejected:
            continue
        detection = item.detection
        x1, y1, x2, y2 = detection.bbox
        color = (0, 200, 0) if item.accepted else (0, 0, 220)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if item.accepted:
            label = f"{vehicle_index} {detection.class_name} {detection.confidence:.2f} OK"
        else:
            label = (
                f"R{vehicle_index} {detection.class_name} "
                f"{detection.confidence:.2f} {','.join(item.reasons)}"
            )
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
    item: ValidatedVehicleDetection,
) -> dict[str, object]:
    detection = item.detection
    return {
        "vehicle_index": vehicle_index,
        "class_id": detection.class_id,
        "class_name": detection.class_name,
        "confidence": round(float(detection.confidence), 6),
        "bbox_xyxy": list(detection.bbox),
        "accepted": item.accepted,
        "rejection_reasons": list(item.reasons),
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
    validation_config: VehicleValidationConfig | None = None,
    save_annotated: bool = True,
    show_rejected: bool = False,
    max_frames: int | None = None,
) -> dict[str, object]:
    """Detect vehicles in every frame and write an inspectable result bundle."""

    source_path = Path(source)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    json_path = output_path / f"{source_path.stem}_vehicles.json"
    annotated_path = output_path / f"{source_path.stem}_vehicles_annotated.mp4"

    defaults = VehicleConfig()
    selected_validation = validation_config or VehicleValidationConfig()
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
    raw_frames_with_detections = 0
    frames_with_detections = 0
    raw_detections_count = 0
    total_detections = 0
    rejected_detections_count = 0
    rejection_reasons: Counter[str] = Counter()
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
                validated = validate_frame_detections(
                    detections,
                    frame_width=metadata.width,
                    frame_height=metadata.height,
                    config=selected_validation,
                )
                accepted = tuple(item for item in validated if item.accepted)
                rejected = tuple(item for item in validated if not item.accepted)
                if validated:
                    raw_frames_with_detections += 1
                if accepted:
                    frames_with_detections += 1
                raw_detections_count += len(validated)
                total_detections += len(accepted)
                rejected_detections_count += len(rejected)
                rejection_reasons.update(rejection_reason_counts(validated))
                class_counts.update(item.detection.class_name for item in accepted)
                raw_json = [
                    _detection_to_json(index, item)
                    for index, item in enumerate(validated)
                ]
                records.append(
                    {
                        "frame_index": video_frame.frame_index,
                        "timestamp_seconds": round(video_frame.timestamp_seconds, 6),
                        "raw_detections": raw_json,
                        "accepted_detections": [
                            item for item in raw_json if item["accepted"]
                        ],
                        "detections": [
                            item for item in raw_json if item["accepted"]
                        ],
                        "validation": {
                            "raw_count": len(validated),
                            "accepted_count": len(accepted),
                            "rejected_count": len(rejected),
                            "rejection_reasons": rejection_reason_counts(validated),
                        },
                    }
                )
                if writer is not None:
                    annotated = video_frame.image.copy()
                    _draw_validated_detections(
                        annotated,
                        validated,
                        show_rejected=show_rejected,
                    )
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
            "raw_frames_with_detections": raw_frames_with_detections,
            "frames_with_detections": frames_with_detections,
            "raw_vehicle_detections": raw_detections_count,
            "accepted_vehicle_detections": total_detections,
            "rejected_vehicle_detections": rejected_detections_count,
            "total_vehicle_detections": total_detections,
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "class_counts": dict(sorted(class_counts.items())),
        },
        "frames": records,
        "performance": {
            "total_seconds": round(elapsed_seconds, 6),
            "ms_per_frame": round(elapsed_seconds * 1000.0 / frames_read, 6),
            "detector_calls": int(getattr(detector, "detect_call_count", frames_read)),
            "model_path": str(getattr(detector, "model_path", model_path or "injected")),
            "detector_confidence": float(
                getattr(
                    detector,
                    "confidence_threshold",
                    confidence if confidence is not None else defaults.confidence,
                )
            ),
            "validation_min_confidence": selected_validation.min_confidence,
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
