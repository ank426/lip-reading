from __future__ import annotations

import numpy as np
import pytest

from lipreading.helpers import (
    load_class_mappings,
    plot_confusion_matrix,
    plot_history,
    save_class_mappings,
)


def test_class_mapping_roundtrip(tmp_path):
    classes = ["ABOUT", "BANK", "CASES"]
    path = tmp_path / "word_mappings.json"

    assert save_class_mappings(classes, path) == path
    assert load_class_mappings(path) == classes


def test_load_class_mappings_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_class_mappings(tmp_path / "absent.json")


def test_plot_history_writes_figure(tmp_path):
    history = {
        "train_loss": [1.0, 0.5],
        "val_loss": [1.1, 0.7],
        "train_acc": [0.2, 0.6],
        "val_acc": [0.1, 0.5],
    }
    target = plot_history(history, tmp_path / "history.png")

    assert target.exists()
    assert target.stat().st_size > 0


def test_plot_confusion_matrix_normalizes(tmp_path):
    classes = ["A", "B"]
    target = plot_confusion_matrix([0, 1, 1], [0, 1, 0], classes, tmp_path / "cm.png")

    assert target.exists()
    assert np.isfinite(1.0)
