"""Regression tests for per-video BYTE benchmark track identity."""

from __future__ import annotations

from tests.benchmark_byte_t3 import qualified_track_key, qualify_existing_report_tracks


def main() -> int:
    assert qualified_track_key("test1_60fps.mp4", 17) != qualified_track_key(
        "test2_30fps.mp4", 17
    )
    report = {
        "videos": {
            "test1_60fps.mp4": {
                "byte_t3": {
                    "tracks": [{"track_id": 17}],
                    "suspicious_low_high_ratio_tracks": [{"track_id": 17}],
                    "rejected_unconfirmed_tracks": [],
                }
            },
            "test2_30fps.mp4": {
                "byte_t3": {
                    "tracks": [{"track_id": 17}],
                    "suspicious_low_high_ratio_tracks": [],
                    "rejected_unconfirmed_tracks": [],
                }
            },
        }
    }
    qualify_existing_report_tracks(report)
    first = report["videos"]["test1_60fps.mp4"]["byte_t3"]["tracks"][0]
    second = report["videos"]["test2_30fps.mp4"]["byte_t3"]["tracks"][0]
    suspicious = report["videos"]["test1_60fps.mp4"]["byte_t3"][
        "suspicious_low_high_ratio_tracks"
    ][0]
    assert first["video"] == suspicious["video"] == "test1_60fps.mp4"
    assert first["track_key"] == suspicious["track_key"]
    assert first["track_key"] != second["track_key"]
    print("BYTE benchmark per-video track identity: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
