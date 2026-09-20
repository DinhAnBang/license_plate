"""Unchanged domain defaults shared by runtime and benchmark entry points.

These constants centralize existing values only. They are intentionally not
user-tuning knobs and changing one is an algorithm/configuration change.
"""

from __future__ import annotations

# Detector defaults.
DETECTOR_CONF_THRESHOLD = 0.5
DETECTOR_NMS_IOU_THRESHOLD = 0.45

# Legacy tracker defaults retained for the historical baseline.
LEGACY_IOU_THRESHOLD = 0.3
LEGACY_MAX_MISSED = 10

# SORT defaults.
SORT_IOU_THRESHOLD = 0.25
SORT_MAX_AGE = 10
SORT_MIN_HITS = 1

# BYTE T1/T2/T3 defaults.
BYTE_HIGH_THRESHOLD = 0.70
BYTE_LOW_THRESHOLD = 0.10
BYTE_HIGH_MATCH_IOU_THRESHOLD = 0.25
BYTE_LOW_MATCH_IOU_THRESHOLD = 0.25
BYTE_UNCONFIRMED_MATCH_IOU_THRESHOLD = 0.25
BYTE_TRACK_BUFFER_FRAMES_AT_30FPS = 10
BYTE_REFERENCE_FPS = 30.0

# Recognition and quality defaults.
RECOGNITION_TOP_K = 3
QUALITY_CONFIDENCE_WEIGHT = 0.30
QUALITY_SHARPNESS_WEIGHT = 0.35
QUALITY_BRIGHTNESS_WEIGHT = 0.15
QUALITY_SIZE_WEIGHT = 0.20
QUALITY_SHARPNESS_REFERENCE = 500.0
QUALITY_BRIGHTNESS_TARGET = 127.5
QUALITY_REFERENCE_AREA = 12_000.0

# OCR character post-processing defaults.
OCR_CONF_THRESHOLD = 0.25
OCR_NMS_IOU_THRESHOLD = 0.70
OCR_VOTE_CONFIDENCE_WEIGHT = 0.70
OCR_VOTE_QUALITY_WEIGHT = 0.30

# T4 post-tracking tracklet stitching defaults. These values affect only the
# final video event/crop/result layer; frame-level association is unchanged.
STITCHING_ENABLED = True
STITCH_MAX_GAP_SEC = 0.75
STITCH_MAX_EDIT_DISTANCE = 1
STITCH_MIN_FUZZY_TEXT_LENGTH = 6
STITCH_MAX_CENTER_DISTANCE_RATIO = 3.0

# T5 Vietnam plate validation and final duplicate removal.  These switches
# are explicit so historical T1-T4 benchmark entry points can disable the
# rule layer without changing detector/tracker/OCR behavior.
VIETNAM_PLATE_VALIDATION_ENABLED = True
VIETNAM_PLATE_CORRECTION_ENABLED = True
FINAL_INVALID_FILTER_ENABLED = True
FINAL_DUPLICATE_MERGE_ENABLED = True
OVERLAP_DUPLICATE_MERGE_ENABLED = True
T5_MAX_CORRECTIONS = 2
DUPLICATE_MIN_SHARED_FRAMES = 3
DUPLICATE_MEAN_IOU_THRESHOLD = 0.65
DUPLICATE_CENTER_DISTANCE_THRESHOLD = 0.75
DUPLICATE_MAX_GAP_SEC = STITCH_MAX_GAP_SEC
DUPLICATE_MAX_CENTER_DISTANCE_RATIO = STITCH_MAX_CENTER_DISTANCE_RATIO

# Historical benchmark reference values. These are separate from the legacy
# tracker defaults because the benchmark intentionally compares legacy and
# SORT with the same 0.25 association threshold.
BENCHMARK_DETECTOR_CONF_THRESHOLD = BYTE_HIGH_THRESHOLD
BENCHMARK_LEGACY_IOU_THRESHOLD = SORT_IOU_THRESHOLD
