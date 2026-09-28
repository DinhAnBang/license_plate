"""Visualize every production plate decision without rerunning its algorithms."""

from __future__ import annotations

import argparse
import json
import sys
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.plate_buffer_stage import run_video_plate_buffer


def _put_label(image: np.ndarray, text: str, y: int = 22) -> None:
    cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, text, (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 0, 0), 1, cv2.LINE_AA)


def _montage(images: list[tuple[str, np.ndarray]], target: Path) -> None:
    if not images:
        return
    tiles = []
    for label, image in images:
        tile = np.zeros((180, 360, 3), dtype=np.uint8)
        height, width = image.shape[:2]
        scale = min(340 / width, 135 / height)
        resized = cv2.resize(image, (max(1, int(width * scale)), max(1, int(height * scale))))
        y = 32 + (135 - resized.shape[0]) // 2
        x = (360 - resized.shape[1]) // 2
        tile[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
        _put_label(tile, label[:47])
        tiles.append(tile)
    while len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[index:index + 2]) for index in range(0, len(tiles), 2)]
    if not cv2.imwrite(str(target), np.vstack(rows)):
        raise OSError(f"Could not write montage: {target}")


def build_accuracy_review(video_path: Path, result: dict, output_dir: Path) -> dict:
    """Render production JSON and exact source-frame crops; never re-score them."""
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks: dict[int, dict] = {}
    frames = {int(frame["frame_index"]): frame for frame in result["frames"]}
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise OSError(f"Could not open video: {video_path}")
    try:
        frame_index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            record = frames.get(frame_index)
            if record is not None:
                for candidate in record["plate_candidates"]:
                    track_id = int(candidate["track_id"])
                    track = tracks.setdefault(track_id, {"candidates": [], "reasons": Counter()})
                    track_dir = output_dir / f"track_{track_id}"
                    reason = candidate.get("final_rejection_reason") or candidate.get("buffer_rejection_reason")
                    category = "selected" if candidate["ownership_status"] == "selected" else "rejected"
                    target_dir = track_dir / category
                    target_dir.mkdir(parents=True, exist_ok=True)
                    name = f"frame{frame_index:06d}_candidate{candidate['candidate_index']:03d}.jpg"
                    x1, y1, x2, y2 = candidate["plate_bbox_xyxy"]
                    context = image.copy()
                    vx1, vy1, vx2, vy2 = candidate["vehicle_bbox_xyxy"]
                    cv2.rectangle(context, (vx1, vy1), (vx2, vy2), (0, 210, 0), 2)
                    cv2.rectangle(context, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    raw = candidate.get("plate_bbox_before_clamp_xyxy")
                    if candidate.get("coordinate_clamped") and raw:
                        cv2.rectangle(context, (raw[0], raw[1]), (raw[2], raw[3]), (255, 0, 255), 2)
                    label = "TOPK" if candidate.get("topk_status") == "retained" else (reason or category)
                    _put_label(context, f"ID{track_id} P{candidate['candidate_index']} {label}")
                    context_path = target_dir / name
                    if not cv2.imwrite(str(context_path), context):
                        raise OSError(f"Could not write: {context_path}")
                    candidate["review_context_path"] = str(context_path)
                    if candidate.get("crop_valid") and 0 <= x1 < x2 <= image.shape[1] and 0 <= y1 < y2 <= image.shape[0]:
                        crop_path = target_dir / name.replace(".jpg", "_crop.jpg")
                        if not cv2.imwrite(str(crop_path), image[y1:y2, x1:x2]):
                            raise OSError(f"Could not write: {crop_path}")
                        candidate["review_crop_path"] = str(crop_path)
                        all_dir = track_dir / "candidates"
                        all_dir.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(crop_path, all_dir / crop_path.name)
                    track["candidates"].append({"frame_index": frame_index, **candidate})
                    if reason:
                        track["reasons"][reason] += 1
            frame_index += 1
    finally:
        capture.release()

    final_by_id = {int(item["track_id"]): item for item in result["tracks"]}
    reports = []
    for track_id, track in sorted(tracks.items()):
        track_dir = output_dir / f"track_{track_id}"
        topk = final_by_id.get(track_id, {}).get("top_k", [])
        images = []
        for item in topk:
            crop = cv2.imread(str(item["path"]))
            if crop is not None:
                images.append((f"Top{item['rank']} frame{item['frame_index']} score{item['quality']['total_score']:.3f}", crop))
                topk_dir = track_dir / "topk"
                topk_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item["path"], topk_dir / Path(item["path"]).name)
        if images:
            _montage(images, track_dir / "topk_montage.jpg")
        report = {
            "track_id": track_id,
            "candidate_count": len(track["candidates"]),
            "topk_count": len(topk),
            "rejection_reasons": dict(track["reasons"]),
            "top_k": topk,
            "candidates": track["candidates"],
        }
        (track_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        reports.append({"track_id": track_id, "candidate_count": report["candidate_count"],
                        "topk_count": report["topk_count"], "rejection_reasons": report["rejection_reasons"],
                        "report_path": str(track_dir / "report.json")})
    summary = {"source": str(video_path), "production_result": result.get("output_path"),
               "tracks": reports}
    (output_dir / "accuracy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--result-json", type=Path, help="Existing production Top-K JSON; avoid model rerun")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    if args.result_json:
        result = json.loads(args.result_json.read_text(encoding="utf-8"))
        result["output_path"] = str(args.result_json)
    else:
        result = run_video_plate_buffer(args.input, args.output_dir / "core", device=args.device)
    summary = build_accuracy_review(args.input, result, args.output_dir)
    print(json.dumps({"tracks": len(summary["tracks"]),
                      "summary_path": str(args.output_dir / "accuracy_summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
