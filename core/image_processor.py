"""Image-level processing built on top of :class:`PlateDetector`."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import Detection, DetectorError, PlateDetector


class ImageProcessingError(RuntimeError):
    """Raised when an image cannot be processed or an output cannot be saved."""


class ImageProcessor:
    """Process one image with an existing detector and save all plate outputs."""

    def __init__(
        self,
        detector: PlateDetector,
        output_dir: str | Path = "output",
        project_root: str | Path | None = None,
    ) -> None:
        self.detector = detector
        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = self.project_root / self.output_dir
        self.output_dir = self.output_dir.resolve()

        self.images_dir = self.output_dir / "images"
        self.crops_dir = self.output_dir / "crops"
        self.json_dir = self.output_dir / "json"

    def process(self, source_path: str | Path) -> dict[str, Any]:
        """Process an image and return the same data that is written to JSON."""

        source = Path(source_path)
        if not source.is_absolute():
            source = self.project_root / source
        source = source.resolve()
        if not source.is_file():
            raise ImageProcessingError(f"Input image does not exist: {source}")

        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            raise ImageProcessingError(f"OpenCV could not read image: {source}")

        height, width = image.shape[:2]
        stem = source.stem
        self._create_output_dirs()
        self._remove_old_crops(stem)

        try:
            detections = self.detector.detect(image)
        except DetectorError as exc:
            raise ImageProcessingError(f"Detection failed for '{source}': {exc}") from exc
        except Exception as exc:
            raise ImageProcessingError(f"Unexpected detection error for '{source}': {exc}") from exc

        annotated = image.copy()
        plates: list[dict[str, Any]] = []
        for detection in detections:
            box = self._valid_box(detection, width, height)
            if box is None:
                print(f"WARNING: skipping invalid detection: {detection!r}")
                continue

            x1, y1, x2, y2 = box
            crop = image[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] == 0 or crop.shape[1] == 0:
                print(f"WARNING: skipping empty crop for box: {box}")
                continue

            plate_id = len(plates) + 1
            crop_path = self.crops_dir / f"{stem}_{plate_id:03d}.jpg"
            if not cv2.imwrite(str(crop_path), crop):
                raise ImageProcessingError(f"Could not save crop: {crop_path}")

            confidence = round(float(detection["conf"]), 6)
            crop_relative = self._project_relative_path(crop_path)
            plates.append(
                {
                    "id": plate_id,
                    "conf": confidence,
                    "box": box,
                    "crop": crop_relative,
                }
            )
            self._draw_detection(annotated, plate_id, confidence, box)

        annotated_path = self.images_dir / f"{stem}_result.jpg"
        if not cv2.imwrite(str(annotated_path), annotated):
            raise ImageProcessingError(f"Could not save annotated image: {annotated_path}")

        result: dict[str, Any] = {
            "status": "ok",
            "type": "image",
            "source": self._project_relative_path(source),
            "size": {"w": width, "h": height},
            "count": len(plates),
            "plates": plates,
        }
        json_path = self.json_dir / f"{stem}.json"
        self._write_json(json_path, result)
        return result

    def _create_output_dirs(self) -> None:
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.crops_dir.mkdir(parents=True, exist_ok=True)
        self.json_dir.mkdir(parents=True, exist_ok=True)

    def _remove_old_crops(self, stem: str) -> None:
        # Restrict cleanup to output/crops and this exact input stem. Crops from
        # other input images are intentionally left untouched.
        for old_crop in self.crops_dir.glob(f"{stem}_*.jpg"):
            if old_crop.is_file():
                try:
                    old_crop.unlink()
                except OSError as exc:
                    raise ImageProcessingError(
                        f"Could not remove old crop '{old_crop}': {exc}"
                    ) from exc

    @staticmethod
    def _valid_box(
        detection: Mapping[str, Any],
        image_width: int,
        image_height: int,
    ) -> list[int] | None:
        box = detection.get("box")
        if not isinstance(box, Sequence) or isinstance(box, (str, bytes)) or len(box) != 4:
            return None
        try:
            values = [int(round(float(value))) for value in box]
        except (TypeError, ValueError, OverflowError):
            return None
        if not all(math.isfinite(float(value)) for value in values):
            return None

        x1, y1, x2, y2 = values
        x1 = max(0, min(image_width - 1, x1))
        y1 = max(0, min(image_height - 1, y1))
        x2 = max(0, min(image_width, x2))
        y2 = max(0, min(image_height, y2))
        if x1 >= x2 or y1 >= y2:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _draw_detection(
        annotated: np.ndarray,
        plate_id: int,
        confidence: float,
        box: list[int],
    ) -> None:
        x1, y1, x2, y2 = box
        label = f"Plate {plate_id} | {confidence:.2f}"
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        label_width = text_width + 6
        label_left = x1
        if label_left + label_width > annotated.shape[1]:
            label_left = max(0, x2 - label_width)
        label_top = y1 - text_height - baseline - 4
        if label_top < 0:
            label_top = min(annotated.shape[0] - text_height - baseline - 4, y2 + 2)
        label_top = max(0, label_top)
        label_right = min(annotated.shape[1], label_left + label_width)
        label_bottom = min(annotated.shape[0], label_top + text_height + baseline + 4)
        cv2.rectangle(
            annotated,
            (label_left, label_top),
            (label_right, label_bottom),
            (0, 255, 0),
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label,
            (label_left + 3, label_top + text_height + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )

    def _project_relative_path(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return path.resolve().as_posix()

    @staticmethod
    def _write_json(path: Path, result: dict[str, Any]) -> None:
        try:
            with path.open("w", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False)
                handle.write("\n")
        except OSError as exc:
            raise ImageProcessingError(f"Could not save JSON '{path}': {exc}") from exc
