# R1B — Safe Architecture Audit

## Scope and baseline

This audit was completed before source refactoring. The current working
behavior is the source of truth. No model, input, user output, EXE, or frozen
runtime file was modified.

Inventory:

- Runtime entry points: `main.py`, `engine/ai_engine.py`,
  `engine/protocol.py`, `engine/output_manager.py`, and
  `engine/runtime_paths.py`.
- Detection/recognition: `core/detector.py`, `core/image_processor.py`,
  `core/video_processor.py`, `core/quality.py`, `core/ocr.py`,
  `core/ocr_voter.py`, and `core/plate_normalizer.py`.
- Tracking primitives: `core/tracker.py` (legacy), `core/sort_tracker.py`,
  `core/byte_tracker.py`, `core/kalman_box_tracker.py`, and
  `core/assignment.py`.
- Production result boundary: `core/result_serializer.py` and
  `core/result_writer.py`.
- Development benchmarks: `tests/benchmark_sort.py`,
  `tests/benchmark_byte.py`, and `tests/benchmark_byte_t3.py`.
- Tests: unit, integration, runtime-path, JSON-contract, persistent-engine,
  and benchmark-regression tests under `tests/`.
- Conversion/model utilities: `tools/convert_pt_to_onnx.py` and
  `tools/inspect_model.py`.
- Runtime dependencies in `requirements.txt`: `onnxruntime`,
  `opencv-python`, and `numpy`.

Baseline snapshot: `output/r1b_baseline/benchmark_report.json`.

- SORT: test1 = 31 final tracks; test2 = 15 final tracks.
- BYTE T3: test1 = 20 confirmed final tracks; test2 = 7 confirmed final
  tracks.
- `test2_30fps.mp4`, `59N304864`: track 17, frames 940–1052, 113 total
  hits, 25 HIGH, 88 LOW, high ratio `0.22123893805309736`.
- The current benchmark did not produce a CÒN PHÒNG candidate; this remains a
  characterization result, not a new expected detection.

## Findings

### A. SAFE TO CLEAN

| File / symbol | Evidence | Proposal | Risk | Test protection |
| --- | --- | --- | --- | --- |
| `core/detector.py`: `math` import | No call site in source/tests | Remove unused import | Low | Detector and integration tests |
| `core/image_processor.py`: `Detection` import | No call site; annotations use inferred detector output | Remove unused import | Low | Image processor tests |
| `core/video_processor.py`: `_is_better_candidate` | Definition only; no call site in source, tests, CLI, or benchmarks | Remove private dead helper | Low | Video/quality/OCR tests |
| `core/assignment.py` and `core/byte_tracker.py` gated matching | BYTE has a local wrapper; SORT has equivalent inline gating | Extract one pure gated-assignment helper and preserve ordering | Low/medium | Assignment, SORT, BYTE, lifecycle tests |

### B. POSSIBLE DUPLICATION

| Area | Files / symbols | Decision |
| --- | --- | --- |
| IoU | `core/assignment.py:iou_matrix` and `core/tracker.py:calculate_iou` | Keep both: scalar legacy helper has compatibility/error-handling semantics; vector helper is canonical for SORT/BYTE. |
| Box normalization | `core/tracker.py:_normalise_box` and `core/kalman_box_tracker.py:normalize_box` | Keep for now; legacy and Kalman paths have separate accepted input behavior. |
| Gated assignment | SORT inline and BYTE `_gated_assignment` | Safe extraction planned; numerical behavior must remain identical. |
| Path normalization | processors, `VideoResultWriter`, `RequestOutputManager`, and runtime paths | Keep boundaries separate; they protect different source/resource/output roots. |
| Track summaries | legacy/SORT/BYTE | Keep separate; fields and lifecycle semantics are intentionally different. |
| Benchmark setup | three benchmark scripts repeat model/config setup | Do not merge in R1B; preserve historical T1/T2/T3 entry points. |

### C. POSSIBLE DEAD CODE

| Item | Evidence | Decision |
| --- | --- | --- |
| `VideoProcessor._is_better_candidate` | Private definition has zero call sites | Safe deletion after baseline. |
| `tools/convert_pt_to_onnx.py` | Explicit CLI utility and documented conversion workflow | Keep. |
| `tools/inspect_model.py` | Explicit model inspection utility | Keep. |
| Benchmark scripts and traces | Referenced by benchmark tests and historical reports | Keep. |
| Legacy tracker | Imported by tests/benchmarks and selectable by `tracker_mode` | Keep. |

### D. ARCHITECTURE SMELLS

| Area | Finding | Decision |
| --- | --- | --- |
| Tracker selection | `VideoProcessor` contains one mode-selection branch; benchmarks intentionally construct explicit tracker variants | No factory in R1B; a factory would add churn without repeated production call sites. |
| BYTE inheritance | `ByteTracker` subclasses `SortTracker` but owns its own lifecycle collections/update/finalize logic | Keep compatibility inheritance; report coupling for a future tested refactor. |
| Candidate ownership | `VideoProcessor` owns quality, Top-K, crop metadata, OCR candidate collection, and voting orchestration | Keep in R1B; extraction risks crop/OCR/JSON behavior without a characterization suite for every candidate ordering. |
| Internal result shape | Processors retain rich diagnostic dictionaries and map them through the R1 serializer | Keep; the serializer is already the public boundary. |
| Detection representation | Detector exposes `Detection`, while trackers accept generic mappings and emit `TrackedDetection` | Keep in R1B; normalizing all internal shapes is a larger type-contract change. |
| Benchmark placement | Benchmarks live under `tests/` and are manually invoked | Keep paths for compatibility; do not mass-move files. |

### E. KEEP FOR COMPATIBILITY

- `core.tracker.PlateTracker`, `Track`, `TrackedDetection`, and
  `calculate_iou` are imported by tests and benchmarks.
- `finish = finalize` aliases are retained by legacy/SORT/BYTE trackers.
- `ByteTracker(SortTracker)` and its `active_tracks`/`finished_tracks` views
  are retained for mode and test compatibility.
- `VideoResultWriter` retains its historical constructor and width/height
  parameters while the public R1 DTO omits those fields.
- Existing benchmark filenames, report locations, trace formats, and output
  folder names are retained.

### F. UNCERTAIN — DO NOT TOUCH

- Do not move `core/` into new subpackages during this phase.
- Do not replace tracker inheritance with a protocol/base class yet.
- Do not merge scalar/vector IoU or legacy/Kalman box normalization.
- Do not introduce a candidate-manager object without a full candidate-level
  characterization snapshot.
- Do not delete benchmark helpers, compatibility exports, or conversion tools.
- Do not remove dependencies based only on package names; source and spec
  usage were checked first.

## Refactor guardrails

The only approved cleanup steps for R1B are unused-import removal, removal of
the proven-unused private helper, a pure gated-assignment extraction, and
centralization of unchanged domain defaults if tests prove output identity.
No threshold, model, tracker lifecycle, OCR, crop, filesystem, or production
JSON value may change.

After each step, run the relevant unit/integration tests and compare the
resulting benchmark snapshot against `output/r1b_baseline/`.

## R1B completed cleanup and validation

- Removed the proven-unused `math` import from `core/detector.py`.
- Removed the proven-unused `Detection` import from
  `core/image_processor.py`.
- Removed the private `VideoProcessor._is_better_candidate` helper; search
  found no call site.
- Added `core/config.py` and replaced duplicated unchanged defaults in the
  runtime and benchmark entry points.
- Added `core.assignment.gated_assignment()` and made SORT/BYTE use it. The
  helper preserves the existing Hungarian ordering and threshold gate.
- Kept scalar legacy IoU, legacy tracker, SORT, BYTE inheritance,
  compatibility exports, benchmark entry points, output folders, and the R1
  serializer unchanged.

Validation:

- Before/after normalized benchmark output matched for both reference videos;
  timing and generated-path fields were excluded from the comparison.
- 59-N3 remained track 17, frames 940–1052, 113/25/88 hits, and ratio
  `0.22123893805309736`.
- Non-report benchmark output file sets and SHA-256 hashes matched exactly.
- All source regression tests, including persistent engine and benchmark
  identity tests, passed.
- No frozen test or EXE build was run.
