"""End-to-end persistence test for the PyInstaller one-file executable."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXE = ROOT / "dist" / "LicensePlateEngine.exe"
IMAGE = ROOT / "input" / "images1.jpg"
VIDEO = ROOT / "input" / "1788115615491-5334873298567571839-5334873298567571839_txDxN1HN.mp4"


def send(process: subprocess.Popen[str], request: dict) -> dict:
    assert process.stdin is not None and process.stdout is not None
    process.stdin.write(json.dumps(request) + "\n")
    process.stdin.flush()
    line = process.stdout.readline()
    assert line, f"EXE closed before responding to {request['id']}"
    return json.loads(line)


def check_result(response: dict, media_type: str) -> tuple[Path, dict[str, int]]:
    request_id = response["id"]
    result = response["result"]
    assert response["status"] == "ok"
    assert result["status"] == "ok" and result["type"] == media_type
    assert result["request_id"] == request_id
    request_dir = EXE.parent / "output" / "requests" / request_id
    annotated = request_dir / ("annotated.jpg" if media_type == "image" else "annotated.mp4")
    result_json = request_dir / "result.json"
    assert annotated.is_file() and result_json.is_file()
    assert json.loads(result_json.read_text(encoding="utf-8")) == result
    assert result["count"] == len(result["plates"])
    assert result["count"] > 0
    for plate in result["plates"]:
        assert plate["crop"].startswith(f"output/requests/{request_id}/crops/")
        assert "_MEI" not in plate["crop"]
        assert (EXE.parent / plate["crop"]).is_file()
    sizes = {str(path.relative_to(request_dir)): path.stat().st_size for path in request_dir.rglob("*") if path.is_file()}
    assert sizes and all(size > 0 for size in sizes.values())
    return request_dir, sizes


def main() -> int:
    assert EXE.is_file(), f"Executable not found: {EXE}"
    assert IMAGE.is_file(), f"Image not found: {IMAGE}"
    assert VIDEO.is_file(), f"Video not found: {VIDEO}"

    process = subprocess.Popen(
        [str(EXE)],
        cwd=EXE.parent,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        bufsize=1,
    )
    assert process.stdout is not None and process.stderr is not None
    try:
        assert json.loads(process.stdout.readline()) == {"event": "starting"}
        assert json.loads(process.stdout.readline()) == {"event": "ready"}

        image_response = send(process, {
            "id": "img-exe-path", "action": "process", "type": "image", "path": str(IMAGE)
        })
        image_dir, image_sizes_before = check_result(image_response, "image")

        video_response = send(process, {
            "id": "video-exe-path", "action": "process", "type": "video", "path": str(VIDEO)
        })
        video_dir, video_sizes_before = check_result(video_response, "video")

        assert send(process, {"id": "ping-exe-path", "action": "ping"}) == {
            "id": "ping-exe-path", "status": "ok", "state": "ready"
        }
        assert send(process, {"id": "stop-exe-path", "action": "shutdown"}) == {
            "id": "stop-exe-path", "status": "ok", "state": "shutting_down"
        }
        process.stdin.close()
        assert process.wait(timeout=30) == 0
        assert process.stdout.read() == ""
        stderr = process.stderr.read()
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=30)
        process.stdout.close()
        process.stderr.close()

    assert f"App root: {EXE.parent}" in stderr
    assert f"Output root: {EXE.parent / 'output' / 'requests'}" in stderr
    resource_line = next(line for line in stderr.splitlines() if "Resource root:" in line)
    assert "_MEI" in resource_line and str(EXE.parent) not in resource_line
    detector_line = next(line for line in stderr.splitlines() if "Detector model:" in line)
    ocr_line = next(line for line in stderr.splitlines() if "OCR model:" in line)
    assert "_MEI" in detector_line and detector_line.endswith(r"models\best.onnx")
    assert "_MEI" in ocr_line and ocr_line.endswith(r"models\OCR\microcharnet.onnx")

    for request_dir, expected_sizes in (
        (image_dir, image_sizes_before),
        (video_dir, video_sizes_before),
    ):
        after = {str(path.relative_to(request_dir)): path.stat().st_size for path in request_dir.rglob("*") if path.is_file()}
        assert after == expected_sizes
    assert not list((EXE.parent / "output").rglob("*.onnx"))

    print(f"EXE: {EXE}")
    print(f"Resource log: {resource_line}")
    print(f"App root: {EXE.parent}")
    print(f"Image output persisted: {image_dir}")
    print(f"Video output persisted: {video_dir}")
    print("Frozen stdout JSON Lines, bundled models, persistent output, ping, and shutdown: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
