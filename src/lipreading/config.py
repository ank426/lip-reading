from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import torch

__all__ = ["CONFIG", "Config"]

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]


def _resolve_device() -> torch.device:
    override = os.environ.get("LIPREADING_DEVICE")
    if override:
        return torch.device(override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # type: ignore[attr-defined]
        return torch.device("mps")
    return torch.device("cpu")


def _default_num_workers() -> int:
    return min(4, os.cpu_count() or 1)


def _default_cuda() -> bool:
    return torch.cuda.is_available()


@dataclass(slots=True)
class Config:
    """Central configuration for paths, hardware and hyperparameters."""

    project_root: Path = PROJECT_ROOT

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    sequence_length: int = 20
    image_size: int = 112
    dropout: float = 0.5
    hidden_size: int = 512
    max_classes: int | None = None
    seed: int = 42

    device: torch.device = field(default_factory=_resolve_device)
    num_workers: int = field(default_factory=_default_num_workers)
    pin_memory: bool = field(default_factory=_default_cuda)
    use_amp: bool = field(default_factory=_default_cuda)

    data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    reports_dir: Path = field(init=False)
    figures_dir: Path = field(init=False)
    best_model_path: Path = field(init=False)
    class_mapping_path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data" / "raw" / "lipread_mp4"
        self.models_dir = self.project_root / "models" / "current"
        self.reports_dir = self.project_root / "reports"
        self.figures_dir = self.reports_dir / "figures"
        self.best_model_path = self.models_dir / "best_lip_reading_model.pt"
        self.class_mapping_path = self.project_root / "word_mappings.json"

    def ensure_dirs(self) -> None:
        for directory in (self.models_dir, self.figures_dir):
            directory.mkdir(parents=True, exist_ok=True)


CONFIG = Config()
