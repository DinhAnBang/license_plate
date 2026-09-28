"""Tests for the first, model-independent video module."""

import cv2
import numpy as np
import pytest

from src.video.source import VideoReader


def _write_video(path, frame_count=3):
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (64, 48),
    )
    assert writer.isOpened()
    for frame_index in range(frame_count):
        writer.write(np.full((48, 64, 3), frame_index * 20, dtype=np.uint8))
    writer.release()


def test_video_reader_exposes_metadata_and_sequential_frames(tmp_path):
    source = tmp_path / "source.mp4"
    _write_video(source, frame_count=3)

    with VideoReader(source) as reader:
        assert reader.metadata.fps == pytest.approx(30.0, abs=0.5)
        assert reader.metadata.frame_count == 3
        assert reader.metadata.width == 64
        assert reader.metadata.height == 48
        frames = list(reader)

    assert [frame.frame_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_seconds for frame in frames] == pytest.approx(
        [0.0, 1 / 30, 2 / 30], abs=1e-6,
    )
    assert all(frame.image.shape == (48, 64, 3) for frame in frames)


def test_video_reader_requires_existing_source(tmp_path):
    with pytest.raises(FileNotFoundError):
        with VideoReader(tmp_path / "missing.mp4"):
            pass


def test_video_reader_requires_context_manager(tmp_path):
    source = tmp_path / "source.mp4"
    _write_video(source, frame_count=1)
    reader = VideoReader(source)

    with pytest.raises(RuntimeError, match="context manager"):
        list(reader)
