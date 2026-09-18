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

Each request gets one JSON response line in arrival order. Progress and diagnostics go to stderr; stdout contains only JSON Lines. Image/video results are returned directly from memory. Annotated media and crops are written under `output/`; JSON copies are written under `output/json/` unless `--no-json-files` is set. Reprocessing the same source overwrites its output and removes stale crops/JSON for that source stem. Use distinct source stems when processing different files to avoid output name collisions.

To run the engine contract tests and real-model integration test:

```powershell
python -m tests.test_engine
python -m tests.test_engine_live
```

`tests/` contains development checks. `tools/inspect_model.py` and `tools/convert_pt_to_onnx.py` are development utilities; `.pt` conversion needs separate Torch/Ultralytics dependencies and is not used by the runtime.
