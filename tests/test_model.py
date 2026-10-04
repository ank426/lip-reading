from __future__ import annotations

import numpy as np
import pytest
import torch

from lipreading.config import Config
from lipreading.helpers import extract_mouth_region
from lipreading.model import LipReadingModel


def test_config_derives_paths(tmp_path):
    cfg = Config(project_root=tmp_path)

    assert cfg.data_dir == tmp_path / "data" / "raw" / "lipread_mp4"
    assert cfg.best_model_path == tmp_path / "models" / "current" / "best_lip_reading_model.pt"
    assert cfg.class_mapping_path == tmp_path / "word_mappings.json"


def test_ensure_dirs_creates_output_folders(tmp_path):
    cfg = Config(project_root=tmp_path)
    cfg.ensure_dirs()

    assert cfg.models_dir.is_dir()
    assert cfg.figures_dir.is_dir()


def test_config_honours_device_override(monkeypatch, tmp_path):
    monkeypatch.setenv("LIPREADING_DEVICE", "cpu")
    assert Config(project_root=tmp_path).device == torch.device("cpu")


@pytest.mark.parametrize("batch_size,seq_len", [(1, 4), (2, 3)])
def test_model_forward_shape(batch_size: int, seq_len: int):
    num_classes = 5
    model = LipReadingModel(num_classes=num_classes, hidden_size=32)
    frames = torch.randn(batch_size, seq_len, 3, 112, 112)

    with torch.inference_mode():
        logits = model(frames)

    assert logits.shape == (batch_size, num_classes)
    assert torch.isfinite(logits).all()


def test_backbone_is_frozen_by_default():
    model = LipReadingModel(num_classes=3, hidden_size=16)
    backbone_grads = [p.requires_grad for p in model.features.parameters()]

    assert not any(backbone_grads)
    assert all(p.requires_grad for p in model.classifier.parameters())


def test_extract_mouth_region_crops_frame():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    crop = extract_mouth_region(frame)

    assert crop.shape[0] == 50
    assert crop.shape[1] == 100
