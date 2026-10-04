from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from lipreading.config import CONFIG
from lipreading.helpers import extract_mouth_region

__all__ = ["LipReadingDataset", "build_transform", "load_video_frames"]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(image_size: int = CONFIG.image_size) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def load_video_frames(
    video_path: str | Path,
    transform: Callable[..., torch.Tensor] | None = None,
    max_frames: int = CONFIG.sequence_length,
    image_size: int = CONFIG.image_size,
) -> torch.Tensor:
    """Decode a video into a ``(max_frames, 3, image_size, image_size)`` tensor.

    Short videos are zero-padded, unreadable frames are skipped and the whole
    sequence falls back to zeros when nothing could be decoded.
    """
    to_tensor = transform if transform is not None else build_transform(image_size)
    capture = cv2.VideoCapture(str(video_path))
    frames: list[torch.Tensor] = []

    try:
        while capture.isOpened() and len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            try:
                frames.append(to_tensor(extract_mouth_region(rgb)))
            except Exception:  # noqa: BLE001 - skip corrupt frames, keep decoding
                continue
    finally:
        capture.release()

    return _stack_or_pad(frames, max_frames, image_size)


def _stack_or_pad(frames: list[torch.Tensor], max_frames: int, image_size: int) -> torch.Tensor:
    if not frames:
        return torch.zeros(max_frames, 3, image_size, image_size)

    tensor = torch.stack(frames[:max_frames])
    if tensor.shape[0] < max_frames:
        padding = tensor.new_zeros(max_frames - tensor.shape[0], *tensor.shape[1:])
        tensor = torch.cat((tensor, padding), dim=0)
    return tensor


class LipReadingDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Word-class video dataset laid out as ``<data_dir>/<WORD>/<split>/*.mp4``."""

    def __init__(
        self,
        split: str,
        *,
        selected_classes: Sequence[str] | None = None,
        data_dir: Path | None = None,
        max_classes: int | None = None,
        sequence_length: int = CONFIG.sequence_length,
        image_size: int = CONFIG.image_size,
    ) -> None:
        self.data_dir = Path(data_dir) if data_dir is not None else CONFIG.data_dir
        self.split = split
        self.sequence_length = sequence_length
        self.transform = build_transform(image_size)
        self.samples: list[tuple[Path, str]] = []

        self.classes = list(selected_classes) if selected_classes else self._discover_classes(max_classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = self._scan()
        if not self.samples:
            raise RuntimeError(f"No .mp4 samples found for split {split!r} under {self.data_dir}")

    def _discover_classes(self, max_classes: int | None) -> list[str]:
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        all_words = sorted(entry.name for entry in self.data_dir.iterdir() if entry.is_dir())
        if max_classes:
            rng = random.Random(CONFIG.seed)
            return sorted(rng.sample(all_words, min(max_classes, len(all_words))))
        return all_words

    def _scan(self) -> list[tuple[Path, str]]:
        samples: list[tuple[Path, str]] = []
        for word in self.classes:
            word_dir = self.data_dir / word / self.split
            if not word_dir.is_dir():
                continue
            samples.extend((path, word) for path in sorted(word_dir.glob("*.mp4")))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path, word = self.samples[idx]
        frames = load_video_frames(path, self.transform, self.sequence_length)
        label = torch.tensor(self.class_to_idx[word], dtype=torch.long)
        return frames, label

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(split={self.split!r}, samples={len(self)}, classes={len(self.classes)})"
        )
