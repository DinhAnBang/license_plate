from __future__ import annotations

import pytest

from core.detector import DetectorError, PlateDetector


def make_detector(shape: list[object], metadata: dict[str, str]) -> PlateDetector:
    detector = PlateDetector.__new__(PlateDetector)
    detector._input_type = "tensor(float)"
    detector._input_shape = shape
    detector._metadata = metadata
    return detector


def test_dynamic_spatial_shape_uses_ultralytics_imgsz_metadata() -> None:
    detector = make_detector(
        ["batch", 3, "height", "width"],
        {"imgsz": "[640, 640]"},
    )

    detector._input_height, detector._input_width, detector._channel_first = (
        detector._validate_input_metadata()
    )

    assert detector.inference_input_shape == [1, 3, 640, 640]


@pytest.mark.parametrize("imgsz", [None, "bad", "[640]", "[0, 640]"])
def test_dynamic_spatial_shape_rejects_missing_or_invalid_imgsz(
    imgsz: str | None,
) -> None:
    metadata = {} if imgsz is None else {"imgsz": imgsz}
    detector = make_detector(["batch", 3, "height", "width"], metadata)

    with pytest.raises(DetectorError, match="imgsz"):
        detector._validate_input_metadata()


def test_static_spatial_shape_does_not_require_metadata() -> None:
    detector = make_detector([1, 3, 320, 640], {})

    assert detector._validate_input_metadata() == (320, 640, True)
