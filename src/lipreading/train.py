from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from lipreading.config import CONFIG, Config
from lipreading.dataset import LipReadingDataset
from lipreading.helpers import plot_history, save_class_mappings, seed_everything
from lipreading.model import LipReadingModel

__all__ = ["main", "parse_args", "run_epoch", "train"]

History = dict[str, list[float]]


def _epoch_stats(total_loss: float, correct: int, total: int) -> tuple[float, float]:
    return total_loss / max(total, 1), correct / max(total, 1)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    *,
    optimizer: AdamW | None = None,
    scaler: torch.amp.GradScaler | None = None,
    description: str = "Epoch",
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)

    total_loss, correct, total = 0.0, 0, 0
    context = torch.enable_grad if training else torch.inference_mode

    with context():
        for videos, labels in tqdm(loader, desc=description, leave=False):
            batch = videos.to(device, non_blocking=True)
            targets = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type=device.type, enabled=scaler is not None):
                outputs = model(batch)
                loss = criterion(outputs, targets)

            if training:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            total_loss += loss.item() * targets.size(0)
            correct += (outputs.argmax(dim=1) == targets).sum().item()
            total += targets.size(0)

    return _epoch_stats(total_loss, correct, total)


def _make_loader(dataset: LipReadingDataset, cfg: Config, *, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )


def train(cfg: Config = CONFIG) -> History:
    cfg.ensure_dirs()
    seed_everything(cfg.seed)

    print(f"Training on {cfg.device} (amp={'on' if cfg.use_amp else 'off'})")

    train_dataset = LipReadingDataset(
        "train",
        data_dir=cfg.data_dir,
        max_classes=cfg.max_classes,
        sequence_length=cfg.sequence_length,
        image_size=cfg.image_size,
    )
    val_dataset = LipReadingDataset(
        "val",
        selected_classes=train_dataset.classes,
        data_dir=cfg.data_dir,
        sequence_length=cfg.sequence_length,
        image_size=cfg.image_size,
    )

    save_class_mappings(train_dataset.classes, cfg.class_mapping_path)
    print(f"Train: {train_dataset!r}\nVal:   {val_dataset!r}")

    train_loader = _make_loader(train_dataset, cfg, shuffle=True)
    val_loader = _make_loader(val_dataset, cfg, shuffle=False)

    model = LipReadingModel(
        num_classes=len(train_dataset.classes),
        hidden_size=cfg.hidden_size,
        dropout=cfg.dropout,
    ).to(cfg.device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler: ReduceLROnPlateau = ReduceLROnPlateau(optimizer, "min", patience=3)
    amp_enabled = cfg.use_amp and cfg.device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    history: History = defaultdict(list)
    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        print(f"\nEpoch {epoch}/{cfg.epochs}")
        train_loss, train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            cfg.device,
            optimizer=optimizer,
            scaler=scaler,
            description="Training",
        )
        val_loss, val_acc = run_epoch(model, val_loader, criterion, cfg.device, description="Validating")
        scheduler.step(val_loss)

        for key, value in (
            ("train_loss", train_loss),
            ("train_acc", train_acc),
            ("val_loss", val_loss),
            ("val_acc", val_acc),
        ):
            history[key].append(value)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            _save_checkpoint(model, epoch, val_acc, cfg.best_model_path)
            print(f"New best model saved to {cfg.best_model_path}")

    print(f"Training complete. Best val acc: {best_acc:.4f}")
    print(f"History plot: {plot_history(history, cfg.figures_dir / 'training_history.png')}")
    return dict(history)


def _save_checkpoint(model: nn.Module, epoch: int, val_acc: float, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"epoch": epoch, "val_acc": val_acc, "state_dict": model.state_dict()},
        path,
    )


def parse_args(argv: Sequence[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(prog="lipreading-train", description="Train the lip reading model")
    defaults = CONFIG
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--hidden-size", type=int, default=defaults.hidden_size)
    parser.add_argument("--sequence-length", type=int, default=defaults.sequence_length)
    parser.add_argument("--image-size", type=int, default=defaults.image_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--max-classes", type=int, default=defaults.max_classes)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=str(defaults.device))
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--data-dir", type=Path, default=None, help="Override the dataset root")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Override the best-model output path")
    parser.add_argument("--class-mapping", type=Path, default=None, help="Override the class mapping path")
    args = parser.parse_args(argv)

    cfg = Config(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
        sequence_length=args.sequence_length,
        image_size=args.image_size,
        num_workers=args.num_workers,
        max_classes=args.max_classes,
        seed=args.seed,
        device=torch.device(args.device),
        use_amp=not args.no_amp,
    )
    if args.data_dir is not None:
        cfg.data_dir = args.data_dir
    if args.checkpoint is not None:
        cfg.best_model_path = args.checkpoint
    if args.class_mapping is not None:
        cfg.class_mapping_path = args.class_mapping
    return cfg


def main(argv: Sequence[str] | None = None) -> int:
    train(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
