"""Second-pass video renderer; reads stored per-frame geometry only."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import cv2
import numpy as np

from .source import VideoReader


def plate_label(row: Mapping[str, object]) -> str | None:
    """Valid format wins; uncertain OCR remains visibly unformatted."""
    plate = row["plate"]
    if plate["format_valid"] and plate["formatted"]:
        return str(plate["formatted"])
    raw = plate.get("normalized_text")
    return str(raw) if isinstance(raw, str) and raw else None


def render_video_frame(
    image: np.ndarray,
    frame_record: Mapping[str, object],
    final_by_track: Mapping[int, Mapping[str, object]],
    *,
    mode: str = "customer",
) -> np.ndarray:
    if mode not in {"customer", "debug"}:
        raise ValueError("render mode must be customer or debug")
    frame = image.copy()
    for vehicle in frame_record["vehicles"]:
        track_id = int(vehicle["track_id"])
        if track_id not in final_by_track:
            continue  # duplicate removed by the existing tracker
        x1, y1, x2, y2 = (int(value) for value in vehicle["bbox_xyxy"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
        label = f"ID {track_id}"
        if mode == "debug":
            label += f" {vehicle['class_name']} {vehicle['confidence']:.2f}"
        cv2.putText(frame, label, (max(0, x1), max(18, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2, cv2.LINE_AA)
    for plate in frame_record["plates"]:
        track_id = int(plate["track_id"])
        row = final_by_track.get(track_id)
        if row is None:
            continue
        x1, y1, x2, y2 = (int(value) for value in plate["bbox_xyxy"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = plate_label(row)
        if mode == "debug":
            label = (label or "unread") + f" P{plate['candidate_index']} {plate['class_name']} {plate['confidence']:.2f}"
        if label:
            cv2.putText(frame, label, (max(0, x1), max(18, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 255), 2, cv2.LINE_AA)
    return frame


def write_annotated_video(
    source: str | Path,
    destination: str | Path,
    frame_records: Sequence[Mapping[str, object]],
    final_by_track: Mapping[int, Mapping[str, object]],
    *,
    mode: str = "customer",
) -> int:
    """Decode again to draw; no detector/OCR/fusion calls in this pass."""
    if mode not in {"customer", "debug"}:
        raise ValueError("render mode must be customer or debug")
    output_path = Path(destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    frames_written = 0
    try:
        with VideoReader(source) as reader:
            metadata = reader.metadata
            writer = cv2.VideoWriter(
                str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                metadata.fps, (metadata.width, metadata.height),
            )
            if not writer.isOpened():
                raise OSError(f"Could not create annotated video: {output_path}")
            for frame in reader:
                if frame.frame_index >= len(frame_records):
                    raise ValueError("Pass 2 has more frames than pass 1")
                record = frame_records[frame.frame_index]
                if int(record["frame_index"]) != frame.frame_index:
                    raise ValueError("Pass 2 frame index does not match pass 1")
                writer.write(render_video_frame(
                    frame.image, record, final_by_track, mode=mode,
                ))
                frames_written += 1
        if frames_written != len(frame_records):
            raise ValueError("Pass 2 has fewer frames than pass 1")
    finally:
        if writer is not None:
            writer.release()
    return frames_written


__all__ = ["plate_label", "render_video_frame", "write_annotated_video"]
