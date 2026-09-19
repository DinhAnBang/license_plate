# License Plate AI Engine

Run the persistent ONNX engine from the project root:

```powershell
python -m pip install -r requirements.txt
python main.py
```

The engine loads `models/best.onnx` and `models/OCR/microcharnet.onnx` once, warms both sessions, then writes `{"event":"starting"}` and `{"event":"ready"}` as separate stdout lines. Send one JSON object per stdin line:

```json
{"id":"1","action":"ping"}
{"id":"2","action":"process","type":"image","path":"input/images1.jpg"}
{"id":"3","action":"process","type":"video","path":"input/video.mp4"}
{"id":"4","action":"shutdown"}
```

Each request gets one JSON response line in arrival order. Progress and diagnostics go to stderr; stdout contains only JSON Lines. Image/video results are returned directly from memory. Engine artifacts use `output/requests/<request_id>/`: `annotated.jpg` or `annotated.mp4`, `crops/plate_001.jpg` or `crops/track_0001.jpg`, and `result.json` unless `--no-json-files` is set. The result dict includes `request_id`, and generated artifact paths use project-relative forward slashes. Request IDs must be 1–80 ASCII letters, digits, hyphens, or underscores. Reusing an ID clears only that ID's output directory before processing; a failed request's partial directory is removed. Different IDs never share output paths, even when source filenames match.

To run the engine contract tests and real-model integration test:

```powershell
python -m tests.test_engine
python -m tests.test_output_namespace
python -m tests.test_engine_live
```

`tests/` contains development checks. `tools/inspect_model.py` and `tools/convert_pt_to_onnx.py` are development utilities; `.pt` conversion needs separate Torch/Ultralytics dependencies and is not used by the runtime.

## PyInstaller one-file build

Build the console engine (stdin/stdout must remain available) from the project root:

```powershell
python -m PyInstaller --clean --noconfirm LicensePlateEngine.spec
```

The spec bundles only the two ONNX model resources. In source mode, resources and output use the project root. In one-file mode, models are read from PyInstaller's temporary resource root while persistent output is written beside the executable at `dist/output/requests/`.

Run the real executable persistence test after rebuilding:

```powershell
python -B -m tests.test_frozen_executable
```
