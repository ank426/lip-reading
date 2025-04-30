# Lip Reading with MobileNetV2 and GRU

This project implements a deep learning pipeline for automated lip reading (visual speech recognition). It uses a hybrid architecture that combines a Convolutional Neural Network (MobileNetV2) for spatial feature extraction and a Gated Recurrent Unit (GRU) for temporal sequence modeling.

## Project Structure

The codebase is organized as follows:

* **data/**: Contains raw video datasets and processed tensors.
* **models/**: Stores trained model artifacts (.pt files).
* **reports/**: Generated figures (confusion matrices, training history) and logs.
* **src/**: Source code package.

  * **config.py**: Central configuration for hyperparameters and file paths.
  * **data/**: Dataset classes and video processing logic.
  * **models/**: Neural network architecture definitions.
  * **utils/**: Helper functions for logging and visualization.
  * **train.py**: Main training loop.
  * **predict.py**: Inference script for single video prediction.

## Prerequisites

* Python 3.8 or higher
* CUDA enabled GPU (recommended)

## Installation

1. Clone the repository.

```bash
git clone https://github.com/ank426/lip-reading.git
```

2. Create and activate a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Ensure your dataset is structured as follows inside the `data/` directory. The data loader expects a root folder containing subfolders for each word class, each with `train`, `val`, and `test` directories.

```
data/raw/lipread_mp4/
├── ABOUT/
│   ├── train/
│   ├── val/
│   └── test/
├── BANK/
│   ├── ...
└── ...
```

## Configuration

Modify `src/config.py` to adjust training parameters such as:

* `BATCH_SIZE`
* `LEARNING_RATE`
* `EPOCHS`
* `MAX_CLASSES` (set to `None` to train on the full dataset)

## Usage

### Training

Run the training module from the project root. It handles data loading, model initialization, training, validation, and metric logging.

```bash
python -m src.train
```

The best model weights will be stored in `models/current/best_lip_reading_model.pt`, and training plots will be saved in `reports/figures/`.

### Inference

To perform lip reading on a specific video:

```bash
python -m src.predict --video path/to/video.mp4
```

## Architecture Details

The model processes video sequences in the following steps:

1. **Input:** A sequence of video frames converted to a tensor.
2. **Spatial Feature Extraction:** Each frame is passed through a pre trained MobileNetV2 with the classification head removed.
3. **Temporal Aggregation:** Extracted feature vectors are processed by a Bidirectional GRU to capture temporal speech patterns.
4. **Classification:** The final hidden state is fed into a fully connected layer followed by a Softmax activation to predict the word class.

## Results

After training, see the `reports/figures` directory for:

* **Training History:** Plots of loss and accuracy over epochs.
* **Confusion Matrix:** A heatmap showing classification performance across word classes.
