"""Inspect video plate formatting from saved fusion and tracking evidence."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.video.postprocess_stage import postprocess_video_fused
from src.image.vn_plate_postprocessor import preferred_family_for_vehicle_class


def _vehicle_classes(topk_source: dict) -> dict[int, tuple[str, dict[str, int]]]:
    classes: dict[int, Counter[str]] = defaultdict(Counter)
    for frame in topk_source["frames"]:
        for vehicle in frame["vehicles"]:
            classes[int(vehicle["track_id"])][vehicle["class_name"]] += 1
    result: dict[int, tuple[str, dict[str, int]]] = {}
    for track_id, counts in classes.items():
        families = {preferred_family_for_vehicle_class(name) for name in counts}
        if len(families) != 1:
            raise ValueError(f"Inconsistent vehicle families across frames: {track_id}: {dict(counts)}")
        maximum = max(counts.values())
        winners = [name for name, count in counts.items() if count == maximum]
        if len(winners) != 1:
            raise ValueError(f"Tied vehicle classes across frames: {track_id}: {dict(counts)}")
        result[track_id] = (winners[0], dict(counts))
    return result


def run_video_postprocess_accuracy(
    fusion_summary: Path,
    topk_json: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Call the production postprocessor; never rerun detectors, OCR, or fusion."""
    fused_source = json.loads(fusion_summary.read_text(encoding="utf-8"))
    topk_source = json.loads(topk_json.read_text(encoding="utf-8"))
    classes = _vehicle_classes(topk_source)
    tracks: list[dict[str, object]] = []
    seen: set[int] = set()
    for fused in fused_source["tracks"]:
        track_id = int(fused["track_id"])
        if track_id in seen:
            raise ValueError(f"Duplicate fusion track ID: {track_id}")
        seen.add(track_id)
        if track_id not in classes:
            raise ValueError(f"No vehicle class for fusion track {track_id}")
        raw_text = fused["raw_text"]
        confidence = float(fused["confidence"])
        vehicle_class, class_counts = classes[track_id]
        result = postprocess_video_fused(raw_text, confidence, vehicle_class)
        plate = result.postprocessed
        corrections = (
            [
                {"index": item.index, "from": item.from_char,
                 "to": item.to_char, "reason": item.reason}
                for item in plate.corrections
            ]
            if plate else []
        )
        report: dict[str, object] = {
            "track_id": track_id,
            "vehicle_class_name": vehicle_class,
            "vehicle_class_counts": class_counts,
            "fusion_method": fused["method"],
            "fusion_confidence": confidence,
            "raw_text": raw_text,
            "normalized_text": plate.normalized_text if plate else None,
            "corrected_text": plate.corrected_text if plate else None,
            "corrections": corrections,
            "format_family": plate.format_family if plate else None,
            "formatted": result.formatted_text,
            "format_valid": plate.format_valid if plate else False,
            "valid": result.valid,
            "status": result.status,
            "reason": result.reason,
            "unknown_characters": (
                [{"index": index, "character": char} for index, char in plate.unknown_characters]
                if plate else []
            ),
            "char_confidences_passed": False,
            "trace": {
                "normalize": plate.normalized_text if plate else None,
                "position_correction": plate.corrected_text if plate else None,
                "validate": {"family": plate.format_family if plate else None,
                             "structural_valid": plate.format_valid if plate else False,
                             "reason": plate.format_reason if plate else result.reason},
                "format": result.formatted_text,
            },
        }
        track_dir = output_dir / f"track_{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        report_path = track_dir / "report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tracks.append({
            "track_id": track_id, "raw_text": raw_text,
            "normalized_text": report["normalized_text"],
            "corrected_text": report["corrected_text"],
            "format_family": report["format_family"],
            "formatted": result.formatted_text, "valid": result.valid,
            "status": result.status, "reason": result.reason,
            "report_path": str(report_path),
        })
    summary: dict[str, object] = {
        "source_fusion_summary": str(fusion_summary),
        "source_topk_json": str(topk_json),
        "tracks_total": len(tracks),
        "tracks_with_ocr": sum(item["raw_text"] is not None for item in tracks),
        "structurally_valid": sum(item["valid"] for item in tracks),
        "no_ocr": sum(item["status"] == "no_ocr" for item in tracks),
        "note": "Structural format only; not independent confirmation of the real plate.",
        "tracks": tracks,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = [
        "# Hậu xử lý biển số video phone",
        "",
        "Chỉ dùng Fusion và dữ liệu xe đã lưu; không chạy lại model. `valid` chỉ nghĩa là đúng cấu trúc theo rule hiện có, không xác nhận biển thật.",
        "",
        "| Track | Fusion raw | Normalize | Corrected | Family | Formatted | Valid | Lý do |",
        "| ---: | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in tracks:
        def display(value: object) -> str:
            return f"`{value}`" if value is not None else "—"

        lines.append(
            f"| {item['track_id']} | {display(item['raw_text'])} "
            f"| {display(item['normalized_text'])} | {display(item['corrected_text'])} "
            f"| {display(item['format_family'])} | {display(item['formatted'])} "
            f"| {item['valid']} | {item['reason'] or '—'} |"
        )
    lines += [
        "",
        "Giới hạn: chuỗi ô tô 7 ký tự có thể là biển 4 số thật hoặc OCR thiếu 1 số; rule hiện tại không phân biệt được. Video không dùng nhánh xóa ký tự dựa trên confidence từng ký tự.",
    ]
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-summary", type=Path, required=True)
    parser.add_argument("--topk-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_video_postprocess_accuracy(
        args.fusion_summary, args.topk_json, args.output_dir
    )
    for track in summary["tracks"]:
        print(f"Track {track['track_id']}: {track['raw_text']!r} -> {track['formatted']!r} ({track['status']})")
    print(f"Summary: {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
