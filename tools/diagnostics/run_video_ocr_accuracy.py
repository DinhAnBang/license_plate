"""OCR each retained video plate crop independently, with decoder evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import OCRConfig
from src.microcharnet_ocr import MicroCharNetOCR, OCRCharacter


def _write_image(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not save image: {path}")


def _original_crop(capture: cv2.VideoCapture, item: dict) -> np.ndarray:
    frame_index = int(item["frame_index"])
    if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
        raise OSError(f"Could not seek to frame {frame_index}")
    ok, frame = capture.read()
    if not ok:
        raise OSError(f"Could not decode frame {frame_index}")
    x1, y1, x2, y2 = (int(value) for value in item["plate_bbox_xyxy"])
    if x1 < 0 or y1 < 0 or x2 > frame.shape[1] or y2 > frame.shape[0] or x2 <= x1 or y2 <= y1:
        raise ValueError(f"Invalid retained plate bbox at frame {frame_index}")
    crop = frame[y1:y2, x1:x2].copy()
    quality = item["quality"]
    if crop.shape[:2] != (int(quality["height"]), int(quality["width"])):
        raise ValueError(f"Saved Top-K dimensions disagree with source frame {frame_index}")
    return crop


def _preprocessed_preview(tensor: np.ndarray) -> np.ndarray:
    rgb = np.clip(np.rint(tensor[0].transpose(1, 2, 0) * 255.0), 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _character_preview(crop: np.ndarray, characters: tuple[OCRCharacter, ...]) -> np.ndarray:
    scale = min(4, max(2, int(640 / max(1, crop.shape[1]))))
    enlarged = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                          interpolation=cv2.INTER_NEAREST)
    legend_height = max(34, 25 * (len(characters) + 1))
    canvas = np.full((enlarged.shape[0] + legend_height, enlarged.shape[1], 3),
                     30, dtype=np.uint8)
    canvas[:enlarged.shape[0], :enlarged.shape[1]] = enlarged
    if not characters:
        cv2.putText(canvas, "NO CHARACTERS", (5, enlarged.shape[0] + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    for order, item in enumerate(characters, start=1):
        color = (0, 255, 255) if order <= len(characters) // 2 else (255, 255, 0)
        x1, y1, x2, y2 = item.bbox
        cv2.rectangle(canvas, (x1 * scale, y1 * scale), (x2 * scale, y2 * scale), color, 2)
        cv2.putText(canvas, str(order), (x1 * scale, max(17, y1 * scale - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        cv2.putText(canvas, f"{order}: {item.char}  {item.confidence:.3f}",
                    (5, enlarged.shape[0] + order * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return canvas


def run_video_ocr_accuracy(
    plate_json: Path,
    video_path: Path,
    output_dir: Path,
    *,
    ocr: MicroCharNetOCR | None = None,
) -> dict[str, object]:
    """Use production Top-K metadata and OCR; never fuse or edit characters."""
    payload = json.loads(plate_json.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    ocr = ocr or MicroCharNetOCR(
        OCRConfig().model,
        conf_threshold=OCRConfig().confidence,
        iou_threshold=OCRConfig().iou,
    )
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    track_reports: list[dict[str, object]] = []
    try:
        for track in payload["tracks"]:
            track_id = int(track["track_id"])
            track_dir = output_dir / f"track_{track_id}"
            track_dir.mkdir(parents=True, exist_ok=True)
            top_k = track["top_k"]
            if not top_k:
                reason = {
                    "track_id": track_id,
                    "status": "no_eligible_crop",
                    "all_evidence_count": track["all_evidence_count"],
                    "eligible_evidence_count": track["eligible_evidence_count"],
                }
                (track_dir / "no_eligible_crop.json").write_text(
                    json.dumps(reason, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                track_reports.append({**reason, "crops": []})
                continue

            crop_reports: list[dict[str, object]] = []
            for item in top_k:
                rank = int(item["rank"])
                crop_dir = track_dir / f"crop_{rank}"
                crop_dir.mkdir(parents=True, exist_ok=True)
                crop = _original_crop(capture, item)
                _write_image(crop_dir / "original.jpg", crop)
                tensor, transform = ocr.preprocess_plate(crop)
                _write_image(crop_dir / "preprocessed.jpg", _preprocessed_preview(tensor))
                trace: dict[str, object] = {}
                result, characters, timing = ocr.recognize_with_debug(crop, trace_out=trace)
                _write_image(crop_dir / "characters.jpg", _character_preview(crop, characters))
                report = {
                    "track_id": track_id,
                    "rank": rank,
                    "frame_index": int(item["frame_index"]),
                    "layout": item["plate_class_name"],
                    "quality": item["quality"],
                    "crop_size": {"width": crop.shape[1], "height": crop.shape[0]},
                    "preprocess": {
                        "mode": ocr.preprocess_mode,
                        "input_shape": list(tensor.shape),
                        "scale_x": transform.scale,
                        "scale_y": transform.scale_y,
                        "pad_x": transform.pad_x,
                        "pad_y": transform.pad_y,
                    },
                    "ocr": {
                        "text": result.text,
                        "confidence": result.confidence,
                        "status": result.status,
                        "raw_character_count": trace["raw_character_count"],
                        "raw_above_0_01_count": trace["raw_above_0_01_count"],
                        "raw_above_0_10_count": trace["raw_above_0_10_count"],
                        "after_confidence_count": trace["after_confidence_count"],
                        "after_geometry_count": trace["after_geometry_count"],
                        "after_nms_count": trace["after_nms_count"],
                        "final_character_count": trace["final_character_count"],
                    },
                    "characters": [
                        {"order": order, "char": char.char, "class_id": char.class_id,
                         "confidence": char.confidence, "bbox_xyxy": list(char.bbox),
                         "raw_index": char.raw_index}
                        for order, char in enumerate(characters, start=1)
                    ],
                    "decode_trace": trace,
                    "timing_ms": {
                        "preprocess": timing.preprocess_ms,
                        "inference": timing.inference_ms,
                        "decode": timing.decode_ms,
                    },
                }
                (crop_dir / "result.json").write_text(
                    json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
                crop_reports.append({
                    "rank": rank, "frame_index": report["frame_index"],
                    "quality_score": item["quality"]["total_score"],
                    "sharpness_score": item["quality"]["sharpness_score"],
                    "size_score": item["quality"]["size_score"],
                    "exposure_score": item["quality"]["exposure_score"],
                    "plate_confidence": item["plate_confidence"],
                    "ocr_text": result.text, "ocr_confidence": result.confidence,
                    "ocr_status": result.status, "result_path": str(crop_dir / "result.json"),
                })
            track_reports.append({"track_id": track_id, "status": "ok", "crops": crop_reports})
    finally:
        capture.release()

    summary: dict[str, object] = {
        "video": str(video_path),
        "plate_topk_json": str(plate_json),
        "ocr_model": ocr.model_info,
        "tracks": track_reports,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# OCR từng crop video (chưa Fusion)", "",
        "| Track | Crop | Frame | Quality | Sharpness | OCR thô | OCR confidence |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for track in track_reports:
        if not track["crops"]:
            lines.append(f"| {track['track_id']} | — | — | — | — | không có crop hợp lệ | — |")
        for candidate in track["crops"]:
            lines.append(
                f"| {track['track_id']} | {candidate['rank']} | {candidate['frame_index']} "
                f"| {candidate['quality_score']:.3f} | {candidate['sharpness_score']:.3f} "
                f"| `{candidate['ocr_text']}` | {candidate['ocr_confidence']:.3f} |"
            )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plate-json", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run_video_ocr_accuracy(args.plate_json, args.video, args.output_dir)
    for track in summary["tracks"]:
        print(f"Track {track['track_id']}: " +
              (", ".join(f"Top{crop['rank']}={crop['ocr_text']!r} ({crop['ocr_confidence']:.3f})"
                         for crop in track["crops"]) or "no eligible crop"))
    print(f"Summary: {args.output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
