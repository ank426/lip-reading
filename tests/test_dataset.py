from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from lipreading.config import CONFIG
from lipreading.dataset import LipReadingDataset, build_transform, load_video_frames


def _write_video(path: Path, frames: int = 6, size: tuple[int, int] = (64, 64)) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, size)
    if not writer.isOpened():
        pytest.skip("OpenCV has no mp4v encoder available")
    for _ in range(frames):
        writer.write(np.full((size[1], size[0], 3), 128, dtype=np.uint8))
    writer.release()
    return path


def test_build_transform_output_shape():
    tensor = build_transform(image_size=64)(np.zeros((48, 48, 3), dtype=np.uint8))
    assert tensor.shape == (3, 64, 64)


def test_load_video_frames_pads_short_videos(tmp_path):
    video = _write_video(tmp_path / "clip.mp4", frames=3)
    frames = load_video_frames(video, max_frames=CONFIG.sequence_length)

    assert frames.shape == (CONFIG.sequence_length, 3, CONFIG.image_size, CONFIG.image_size)
    assert torch.isfinite(frames).all()


def test_load_video_frames_missing_file_returns_zeros(tmp_path):
    frames = load_video_frames(tmp_path / "nope.mp4")
    assert frames.shape == (CONFIG.sequence_length, 3, CONFIG.image_size, CONFIG.image_size)
    assert torch.count_nonzero(frames) == 0


def test_dataset_indexes_classes_and_samples(tmp_path):
    data_dir = tmp_path / "data"
    for word in ("ABOUT", "BANK"):
        for split in ("train", "val"):
            _write_video(data_dir / word / split / f"{word.lower()}_0.mp4", frames=4)
            _write_video(data_dir / word / split / f"{word.lower()}_1.mp4", frames=4)
    (data_dir / "EMPTY").mkdir(parents=True)

    train = LipReadingDataset("train", data_dir=data_dir)
    val = LipReadingDataset("val", selected_classes=train.classes, data_dir=data_dir)

    assert train.classes == ["ABOUT", "BANK", "EMPTY"]
    assert len(train) == 4
    assert val.class_to_idx == train.class_to_idx

    frames, label = train[0]
    assert frames.shape == (CONFIG.sequence_length, 3, CONFIG.image_size, CONFIG.image_size)
    assert label.dtype == torch.long
    assert 0 <= label.item() < len(train.classes)


def test_dataset_max_classes_is_deterministic(tmp_path):
    data_dir = tmp_path / "data"
    for index in range(5):
        _write_video(data_dir / f"W{index}" / "train" / "a.mp4", frames=2)

    first = LipReadingDataset("train", data_dir=data_dir, max_classes=3)
    second = LipReadingDataset("train", data_dir=data_dir, max_classes=3)

    assert len(first.classes) == 3
    assert first.classes == second.classes


def test_dataset_raises_without_videos(tmp_path):
    (tmp_path / "data" / "A" / "train").mkdir(parents=True)
    with pytest.raises(RuntimeError):
        LipReadingDataset("train", data_dir=tmp_path / "data")


def test_dataset_honours_sequence_length_and_image_size(tmp_path):
    data_dir = tmp_path / "data"
    _write_video(data_dir / "A" / "train" / "a.mp4", frames=4)

    dataset = LipReadingDataset("train", data_dir=data_dir, sequence_length=3, image_size=32)
    frames, _ = dataset[0]

    assert frames.shape == (3, 3, 32, 32)
