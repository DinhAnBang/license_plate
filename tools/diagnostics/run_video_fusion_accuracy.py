"""Explain production OCR fusion over existing per-crop video OCR results."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ocr_fusion import (
    OCRFusionCandidate, OCRFusionConfig, candidate_weight, fuse_candidates,
)


def _crop_candidate(track_id: int, crop: dict) -> OCRFusionCandidate:
    result_path = Path(crop["result_path"])
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result["ocr"]["text"] != crop["ocr_text"]:
        raise ValueError(f"OCR summary and crop disagree: {result_path}")
    return OCRFusionCandidate(
        track_id=track_id,
        frame_index=int(crop["frame_index"]),
        rank=int(crop["rank"]),
        raw_text=result["ocr"]["text"],
        ocr_confidence=float(result["ocr"]["confidence"]),
        quality_score=float(crop["quality_score"]),
        char_confidences=tuple(float(item["confidence"]) for item in result["characters"]),
        plate_confidence=float(crop["plate_confidence"]),
        status=result["ocr"]["status"],
    )


def run_video_fusion_accuracy(
    ocr_summary: Path,
    output_dir: Path,
    *,
    config: OCRFusionConfig | None = None,
) -> dict[str, object]:
    """Adapt saved OCR evidence, call production fusion once, and serialize trace."""
    source = json.loads(ocr_summary.read_text(encoding="utf-8"))
    config = config or OCRFusionConfig()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[OCRFusionCandidate] = []
    track_ids: list[int] = []
    for track in source["tracks"]:
        track_id = int(track["track_id"])
        track_ids.append(track_id)
        for crop in track["crops"]:
            candidates.append(_crop_candidate(track_id, crop))
    fusion = fuse_candidates(candidates, track_ids=track_ids, config=config)
    by_track: dict[int, list[OCRFusionCandidate]] = {}
    for candidate in candidates:
        by_track.setdefault(candidate.track_id, []).append(candidate)
    tracks: list[dict[str, object]] = []
    for result in fusion.results:
        track_dir = output_dir / f"track_{result.track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        inputs = [
            {
                "rank": item.rank,
                "frame_index": item.frame_index,
                "text": item.raw_text,
                "normalized_for_fusion": item.raw_text,
                "ocr_confidence": item.ocr_confidence,
                "quality_score": item.quality_score,
                "plate_confidence": item.plate_confidence,
                "char_confidences": list(item.char_confidences or ()),
                "weight": candidate_weight(item, config.include_plate_confidence),
                "status": item.status,
            }
            for item in sorted(by_track.get(result.track_id, ()),
                               key=lambda item: (item.rank, item.frame_index))
        ]
        report: dict[str, object] = {
            "track_id": result.track_id,
            "status": "no_ocr_candidates" if result.valid_candidate_count == 0 else "ok",
            "inputs": inputs,
            "weight_formula": (
                "clamp(ocr_confidence * quality_score * plate_confidence, 0, 1)"
                if config.include_plate_confidence else
                "clamp(ocr_confidence * quality_score, 0, 1)"
            ),
            "normalization": "none; raw OCR text and case are unchanged",
            "exact_votes": [asdict(vote) for vote in result.exact_votes],
            "fusion": {
                "raw_text": result.raw_text if result.valid_candidate_count else None,
                "method": result.method,
                "confidence": result.confidence,
                "support_count": result.support_count,
                "valid_candidate_count": result.valid_candidate_count,
                "total_weight": result.total_weight,
                "winner_weight": result.winner_weight,
                "exact_winner_count": result.exact_votes[0].count if result.exact_votes else 0,
                "consensus_ratio": result.consensus_ratio,
                "exact_consensus_threshold": config.exact_consensus_threshold,
                "reference_text": result.reference_text,
                "supporting_frames": list(result.supporting_frames),
            },
            "alignment": (
                {
                    "reference_tokens": list(result.alignment_reference_row),
                    "rows": [
                        {"frame_index": frame, "text": text, "tokens": list(tokens)}
                        for frame, text, tokens in result.alignment_rows
                    ],
                    "column_votes": [asdict(vote) for vote in result.alignment_column_votes],
                }
                if result.method == "sequence_alignment" else None
            ),
        }
        (track_dir / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        tracks.append(report)
    summary: dict[str, object] = {
        "source_ocr_summary": str(ocr_summary),
        "config": asdict(config),
        "fusion_summary": {
            "tracks_total": fusion.tracks_total,
            "tracks_with_ocr": fusion.tracks_with_ocr,
            "tracks_with_exact_consensus": fusion.tracks_with_exact_consensus,
            "tracks_using_alignment": fusion.tracks_using_alignment,
            "single_candidate_tracks": fusion.single_candidate_tracks,
            "no_valid_ocr_tracks": fusion.no_valid_ocr_tracks,
        },
        "tracks": [
            {"track_id": item["track_id"], "status": item["status"],
             "method": item["fusion"]["method"], "raw_text": item["fusion"]["raw_text"],
             "confidence": item["fusion"]["confidence"],
             "report_path": str(output_dir / f"track_{item['track_id']}" / "report.json")}
            for item in tracks
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# Gộp OCR video (chưa chuẩn hóa biển Việt Nam)", "",
             "| Track | Số OCR | Phương pháp | Fused raw text | Confidence |",
             "| --- | ---: | --- | --- | ---: |"]
    for item in tracks:
        fused = item["fusion"]
        display_text = f"`{fused['raw_text']}`" if fused["raw_text"] is not None else "không có OCR"
        lines.append(
            f"| {item['track_id']} | {len(item['inputs'])} | {fused['method']} "
            f"| {display_text} | {fused['confidence']:.3f} |"
        )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ocr-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run_video_fusion_accuracy(args.ocr_summary, args.output_dir)
    for track in summary["tracks"]:
        print(f"Track {track['track_id']}: {track['method']} -> {track['raw_text']!r}")
    print(f"Summary: {args.output_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
