from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe backend, must precede pyplot import

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

from lipreading.config import CONFIG

__all__ = [
    "extract_mouth_region",
    "load_class_mappings",
    "plot_confusion_matrix",
    "plot_history",
    "save_class_mappings",
    "seed_everything",
]


def seed_everything(seed: int = CONFIG.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_class_mappings(classes: Sequence[str], path: Path | None = None) -> Path:
    """Persist the ordered index-to-word mapping needed for inference."""
    target = Path(path) if path is not None else CONFIG.class_mapping_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(list(classes), indent=2))
    return target


def load_class_mappings(path: Path | None = None) -> list[str]:
    """Load the index-to-word mapping written during training."""
    source = Path(path) if path is not None else CONFIG.class_mapping_path
    if not source.exists():
        raise FileNotFoundError(f"Class mapping file not found at {source}. Train the model first.")
    return json.loads(source.read_text())


def extract_mouth_region(frame: np.ndarray) -> np.ndarray:
    """Crop the lower-central region of a frame.

    Heuristic crop; swap in a face-landmark detector for production use.
    """
    height, width = frame.shape[:2]
    mouth_region = frame[height // 2 :, width // 4 : 3 * width // 4]
    return mouth_region if mouth_region.size else frame


def plot_history(history: dict[str, list[float]], path: Path | None = None) -> Path:
    target = Path(path) if path is not None else CONFIG.figures_dir / "training_history.png"
    target.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="Validation")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss History")

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"], label="Validation")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy", title="Accuracy History")

    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.3)

    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target


def plot_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    classes: Sequence[str],
    path: Path | None = None,
) -> Path:
    target = Path(path) if path is not None else CONFIG.figures_dir / "confusion_matrix.png"
    target.parent.mkdir(parents=True, exist_ok=True)

    labels = list(range(len(classes)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")

    figure, axis = plt.subplots(figsize=(12, 10))
    ConfusionMatrixDisplay(matrix, display_labels=list(classes)).plot(
        ax=axis, cmap="Blues", colorbar=False, values_format=".2f"
    )
    axis.set(xlabel="Predicted", ylabel="True", title="Confusion Matrix (Normalized)")
    axis.tick_params(axis="x", rotation=90, labelsize=6)
    axis.tick_params(axis="y", rotation=0, labelsize=6)
    figure.tight_layout()
    figure.savefig(target, dpi=150)
    plt.close(figure)
    return target
