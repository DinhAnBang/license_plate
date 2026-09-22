"""Run V6 temporal OCR fusion from an existing V5 OCR JSON payload."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.ocr_fusion import OCRFusionConfig, fuse_candidates
from src.ocr_serialization import build_fused_json, format_fusion_debug


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="V5 *_ocr_candidates.json")
    parser.add_argument("--output", type=Path, default=None, help="V6 *_ocr_fused.json")
    parser.add_argument("--exact-consensus-threshold", type=float, default=0.65)
    parser.add_argument(
        "--include-plate-confidence",
        action="store_true",
        help="Optional experimental weight factor; default is OFF to avoid V4 double-counting",
    )
    parser.add_argument("--debug-ocr-fusion", action="store_true")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> Path:
    args = args or parse_args()
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    tracks = payload.get("tracks", ())
    candidates = []
    track_ids = []
    for track in tracks:
        if not isinstance(track, dict):
            continue
        track_id = int(track.get("track_id", 0))
        track_ids.append(track_id)
        for candidate in track.get("candidates", ()):
            if isinstance(candidate, dict):
                candidates.append({**candidate, "track_id": track_id})

    config = OCRFusionConfig(
        exact_consensus_threshold=args.exact_consensus_threshold,
        include_plate_confidence=args.include_plate_confidence,
    )
    report = fuse_candidates(candidates, track_ids=track_ids, config=config)
    output = args.output or args.input.with_name(
        args.input.name.replace("_ocr_candidates.json", "_ocr_fused.json")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(build_fused_json(payload, report), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if args.debug_ocr_fusion:
        print(format_fusion_debug(report, candidates))
    print(f"V6 tracks: {report.tracks_total}")
    print(f"V6 tracks with OCR: {report.tracks_with_ocr}")
    print(f"V6 exact consensus tracks: {report.tracks_with_exact_consensus}")
    print(f"V6 alignment tracks: {report.tracks_using_alignment}")
    print(f"V6 single-candidate tracks: {report.single_candidate_tracks}")
    print(f"V6 no-valid-OCR tracks: {report.no_valid_ocr_tracks}")
    print(f"V6 fusion time: {report.total_ms:.3f} ms")
    print(f"Saved V6 JSON: {output}")
    return output


if __name__ == "__main__":
    main()
