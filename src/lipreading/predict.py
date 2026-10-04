from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import torch

from lipreading.config import CONFIG
from lipreading.dataset import load_video_frames
from lipreading.helpers import load_class_mappings
from lipreading.model import LipReadingModel

__all__ = ["Prediction", "load_model", "main", "predict", "predict_dir"]

LEGACY_KEY_RENAMES = {"fc.weight": "classifier.weight", "fc.bias": "classifier.bias"}


class Prediction(NamedTuple):
    """Predicted label with its confidence."""

    label: str
    confidence: float


def _remap_legacy_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate checkpoints written before the classifier layer was renamed."""
    return {LEGACY_KEY_RENAMES.get(key, key): value for key, value in state_dict.items()}


def load_model(
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
    classes: Sequence[str] | None = None,
) -> LipReadingModel:
    target_device = device or CONFIG.device
    labels = list(classes) if classes is not None else load_class_mappings()

    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else CONFIG.best_model_path
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {checkpoint_path}. Train the model first.")

    model = LipReadingModel(num_classes=len(labels), hidden_size=CONFIG.hidden_size, dropout=CONFIG.dropout)
    checkpoint = torch.load(checkpoint_path, map_location=target_device, weights_only=True)
    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(_remap_legacy_keys(state_dict))
    return model.to(target_device).eval()


@torch.inference_mode()
def predict(
    video_path: str | Path,
    model: LipReadingModel | None = None,
    device: torch.device | None = None,
    top_k: int = 3,
) -> list[Prediction]:
    target_device = device or CONFIG.device
    labels = load_class_mappings()
    net = model if model is not None else load_model(device=target_device, classes=labels)

    frames = load_video_frames(video_path).unsqueeze(0).to(target_device)
    probabilities = net(frames).softmax(dim=1)[0]
    scores, indices = probabilities.topk(min(top_k, probabilities.numel()))

    return [
        Prediction(labels[idx], score) for score, idx in zip(scores.tolist(), indices.tolist(), strict=True)
    ]


def predict_dir(directory: str | Path, top_k: int = 3) -> list[tuple[Path, list[Prediction]]]:
    model = load_model()
    return [
        (video, predict(video, model=model, top_k=top_k)) for video in sorted(Path(directory).glob("*.mp4"))
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lipreading-predict", description="Lip reading inference")
    parser.add_argument("videos", nargs="*", type=Path, help="Video file(s) to classify")
    parser.add_argument("--video", dest="video", type=Path, help="Single video file to classify")
    parser.add_argument("--dir", type=Path, help="Classify every .mp4 inside a directory")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override checkpoint path")
    parser.add_argument("--device", type=str, default=str(CONFIG.device))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--json", action="store_true", help="Emit results as JSON")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    targets: list[Path] = list(args.videos)
    if args.video is not None:
        targets.append(args.video)
    if args.dir is not None:
        targets.extend(sorted(args.dir.glob("*.mp4")))

    if not targets:
        print("Provide at least one video path or --dir.")
        return 2

    missing = [path for path in targets if not path.exists()]
    if missing:
        for path in missing:
            print(f"Video not found: {path}")
        return 1

    device = torch.device(args.device)
    model = load_model(checkpoint_path=args.checkpoint, device=device)

    for path in targets:
        results = predict(path, model=model, device=device, top_k=args.top_k)
        best = results[0]
        if args.json:
            print(json.dumps({"video": str(path), "top_k": [[r.label, r.confidence] for r in results]}))
        else:
            print(f"\n{path}")
            print(f"Prediction: {best.label} (confidence {best.confidence:.4f})")
            for rank, result in enumerate(results, start=1):
                print(f"  {rank}. {result.label}: {result.confidence:.4f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
