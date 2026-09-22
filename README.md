# ALPR core engine

This project detects vehicles and license plates in images and videos, reads plate characters with ONNX OCR, combines video observations, and suggests a conservative Vietnamese plate display format.

## Models and installation

The production pipeline loads these files once per `ALPRPipeline` instance:

- `models/vehicle/yolo26n.onnx`
- `models/plate/best.onnx`
- `models/OCR/microcharnet.onnx`

Use Python 3.12 or a compatible version. From the project root:

```powershell
python -m pip install -r requirements.txt
```

Model export and diagnostic scripts additionally use:

```powershell
python -m pip install -r requirements-dev.txt
```

Production does not load `.pt` weights or import Ultralytics.

## Run

```powershell
python main.py --input input/images1.jpg
python main.py --input input/test1.jpg --save-annotated --debug
python main.py --input input/video.mp4 --save-annotated --save-topk-crops
python main.py --input input/video.mp4 --output output/my_result.json --device cpu
python main.py --release --input input/video.mp4 --device cpu
```

Use `python main.py --help` for all flags. `--device auto` is the default; `--device cuda` requires the CUDA ONNX Runtime provider. `--debug` prints OCR diagnostics and includes per-candidate evidence in JSON. The default source/dev mode is unchanged: JSON is saved as `output/<input-stem>_<YYYYMMDD_HHMMSS>.json`; annotated media is optional and uses the same timestamped prefix. Passing `--output` explicitly keeps the requested JSON path and uses the input stem for the related annotated/crop artifacts.

For customer/EXE mode, use `--release` while testing from source. A packaged executable enables this mode automatically. One input produces one timestamped folder next to the executable (next to the project when run from source), containing only the annotated media and compact customer JSON:

```text
<app-folder>/<input-stem>_<YYYYMMDD_HHMMSS>/
    <input-stem>_annotated.mp4
    <input-stem>.json
```

The release JSON contains only `status` and `vehicles`. Each vehicle has `track_id`, `vehicle_type`, `license`, final fused `confidence`, and `status`. Development diagnostics and Top-K details are not exposed in this JSON.

## Output

The JSON contains `status`, `input`, `summary`, `vehicles`, and `performance`. Each video vehicle has one `track_id`; each image vehicle has a `vehicle_index`. Every vehicle has vehicle metadata, a plate result, and compact OCR evidence. Plate fields include raw fused OCR, normalized text, corrected machine text, formatted display text when a supported family matches, confidence, status, layout, and best plate bbox/frame. Empty or uncertain OCR remains in the result with a status.

Only final results with `plate.confidence >= 0.50` are included in `vehicles`. Format validation is reported through `plate.format_valid` and `plate.format_reason`, but does not remove a result from the JSON. Detection, tracking, crop retention, OCR, and video processing still use the full internal evidence before this final confidence filter.

Video OCR runs on the retained Top-K crops after frame collection, at most K times per track. Image OCR uses one crop per resolved plate. The processing pipeline lives in `src/`. Older diagnostic CLIs and exporters live in `tools/diagnostics/` and run with `python -m tools.diagnostics.<module>` from the project root. The saved archives in `tools/diagnostics/archive/` are historical data.

## Code layout and regression checks

`ALPRPipeline` loads the three ONNX sessions and delegates image/video execution to `image_pipeline.py` and `video_pipeline.py`. Both runners use `vehicle_stage.py`, `plate_stage.py`, `ocr_stage.py`, and the result finalizer. `plate_detector.py` and `microcharnet_ocr.py` contain model preprocessing, inference, and decoding; `ocr_fusion.py` contains the fusion algorithm. Shared box math is in `geometry.py`. JSON formatting is in `result_serialization.py`, `ocr_serialization.py`, and `customer_output.py`.

The per-frame ownership resolver is the production implementation. Its older comparator lives in `tools/diagnostics/legacy_plate_ownership.py` because the diagnostic CLIs still offer legacy comparisons. The temporal resolver keeps its `_legacy_owner` calculation only to preserve the existing `changed_owner_groups` diagnostic counter.

Run regression tests and a compile check before changing a stage:

```powershell
python -m pytest tests -q
python -m compileall -q main.py src tools tests
```

The tests cover geometry, ownership, Top-K, OCR decode, fusion, postprocessing, tracking duplicate cleanup, and image/video JSON using deterministic model doubles. The image path does not use tracking. These boundaries prepare a later plate scheduler without changing detection frequency in this refactor.

## Known limitations

- OCR can emit duplicate characters on some crops. Format matching does not remove them arbitrarily.
- When a car/bus/truck OCR string fits only the common motorcycle family, the final result keeps the raw text, suppresses display formatting, and reports low confidence. Vehicle classification can itself be wrong, so this is a warning rather than a rejection.
- Common civilian car and motorcycle templates are suggestions; special plates can remain `unrecognized_format`.
- A single image has no temporal consensus. OCR confidence should be interpreted accordingly.
- An annotated video shows the plate detection available at each frame. The final fused text is in JSON after the video ends.
