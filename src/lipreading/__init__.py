"""Visual speech (lip reading) pipeline: MobileNetV2 frame encoder + BiGRU classifier."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from lipreading.config import CONFIG, Config
from lipreading.dataset import LipReadingDataset, build_transform, load_video_frames
from lipreading.helpers import (
    extract_mouth_region,
    load_class_mappings,
    plot_confusion_matrix,
    plot_history,
    save_class_mappings,
    seed_everything,
)
from lipreading.model import LipReadingModel

try:
    __version__ = version("lipreading")
except PackageNotFoundError:  # pragma: no cover - running from a source checkout
    __version__ = "0.0.0+unknown"

__all__ = [
    "CONFIG",
    "Config",
    "LipReadingDataset",
    "LipReadingModel",
    "__version__",
    "build_transform",
    "extract_mouth_region",
    "load_class_mappings",
    "load_video_frames",
    "plot_confusion_matrix",
    "plot_history",
    "save_class_mappings",
    "seed_everything",
]
