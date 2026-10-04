# Lip Reading

Visual speech recognition: each video is encoded frame-by-frame with a pretrained MobileNetV2
backbone, the resulting feature sequences are aggregated by a bidirectional GRU, and the final
hidden state is classified into a word class.

## Quickstart

```bash
git clone https://github.com/ank426/lip-reading.git
cd lip-reading

uv sync --all-groups          # creates .venv from pyproject.toml + uv.lock
uv run lipreading-train --help
```

Requirements: Python >= 3.11 (`.python-version` pins 3.13) and, ideally, a CUDA GPU.

## Usage

### Training

```bash
uv run lipreading-train                                  # all hyperparameters from config defaults
uv run lipreading-train --epochs 20 --batch-size 16      # override anything from the CLI
uv run lipreading-train --max-classes 10 --no-amp        # quick CPU-friendly experiment
```

The best checkpoint lands in `models/current/best_lip_reading_model.pt`, the class order is written to
`word_mappings.json`, and plots are saved to `reports/figures/`.

Useful flags: `--epochs`, `--batch-size`, `--learning-rate`, `--weight-decay`, `--dropout`,
`--hidden-size`, `--sequence-length`, `--image-size`, `--max-classes`, `--num-workers`, `--device`,
`--no-amp`, `--seed`, `--data-dir`, `--checkpoint`, `--class-mapping`.

### Inference

```bash
uv run lipreading-predict path/to/video.mp4
uv run lipreading-predict clip_a.mp4 clip_b.mp4 --top-k 5
uv run lipreading-predict --dir data/raw/lipread_mp4/ABOUT/test --json
```

### Library use

```python
from lipreading import LipReadingDataset, LipReadingModel, load_video_frames, load_class_mappings

classes = load_class_mappings()
model = LipReadingModel(num_classes=len(classes))
frames = load_video_frames("clip.mp4")  # (20, 3, 112, 112)
logits = model(frames.unsqueeze(0))
```

## Project layout

```
pyproject.toml            # PEP 621 metadata, dependencies, ruff + pytest config
uv.lock                   # pinned resolution (committed)
src/lipreading/
├── __init__.py           # public API re-exports
├── config.py             # Config dataclass: paths, device, hyperparameters
├── dataset.py            # LipReadingDataset + video decoding
├── model.py              # MobileNetV2 + BiGRU network
├── helpers.py            # class-mapping IO, plotting, seeding
├── train.py              # lipreading-train entry point
└── predict.py            # lipreading-predict entry point
tests/                    # pytest suite (dataset, model, helpers, config)
data/raw/lipread_mp4/     # dataset root (git-ignored)
models/current/           # checkpoints (git-ignored)
reports/figures/          # generated plots (git-ignored)
```

## Data preparation

Videos are grouped per word, with one directory per split:

```
data/raw/lipread_mp4/
├── ABOUT/
│   ├── train/
│   ├── val/
│   └── test/
├── BANK/
│   └── ...
```

Each class may need at most `sequence_length` decoded frames; shorter clips are zero-padded and the
mouth region is cropped heuristically from the lower-central part of each frame.

## Configuration

Defaults live in the `Config` dataclass (`src/lipreading/config.py`) and every field can be
overridden from the CLI. Device selection is automatic (`cuda` → `mps` → `cpu`) and can be forced
with `--device` or the `LIPREADING_DEVICE` environment variable.

## Development

```bash
uv sync --all-groups      # install runtime + dev dependencies
uv run ruff check .       # lint
uv run ruff format .      # format
uv run pytest             # tests
uv run pytest --cov=lipreading
uv build                  # build a wheel/sdist
```

## Results

Generated reports in `reports/figures/`:

* **Training history** — loss and accuracy curves per epoch.
* **Confusion matrix** — row-normalized heatmap over word classes.