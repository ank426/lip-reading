# Lip Reading with MobileNetV2 and GRU

This project implements a deep learning pipeline for automated lip reading (visual speech recognition). It utilizes a hybrid architecture combining a Convolutional Neural Network (MobileNetV2) for spatial feature extraction and a Gated Recurrent Unit (GRU) for temporal sequence modeling.

## Project Structure

The codebase is organized as a modular Python package to ensure scalability and reproducibility.

- **data/**: Contains raw video datasets and processed tensors.
- **models/**: Stores trained model artifacts (.pt files).
- **reports/**: Generated figures (confusion matrices, training history) and logs.
- **src/**: Source code package.
  - **config.py**: Central configuration for hyperparameters and file paths.
  - **data/**: Dataset classes and video processing logic.
  - **models/**: Neural network architecture definitions.
  - **utils/**: Helper functions for logging and visualization.
  - **train.py**: Main training loop.
  - **predict.py**: Inference script for single-video prediction.

## Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (recommended for training)

## Installation

1. Clone the repository to your local machine.
2. Create and activate a virtual environment.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Data Preparation

Ensure your dataset is structured as follows inside the `data/` directory. The data loader expects a root folder containing subfolders for each word class, which in turn contain `train`, `val`, and `test` directories.

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
- `BATCH_SIZE`
- `LEARNING_RATE`
- `EPOCHS`
- `MAX_CLASSES` (Set to `None` to train on the full dataset)


## Usage
### Training

To train the model, execute the training module from the project root. This script handles data loading, model initialization, training, validation, and metric logging.

```Bash
python -m src.train
```

Upon completion, the best model weights will be saved to `models/current/best_lip_reading_model.pt`, and training plots will be generated in `reports/figures/`.

### Inference

To perform lip reading on a specific video file using the trained model:

```Bash
python -m src.predict --video path/to/video.mp4
```

### Architecture Details

The model processes video sequences as follows:

1. **Input:** A sequence of video frames (Video -> Tensor).
2. **Spatial Feature Extraction:** Each frame is processed independently by a pre-trained MobileNetV2 (ImageNet weights), with the classification head removed.
3. **Temporal Aggregation:** The sequence of feature vectors is passed through a Bidirectional GRU to capture time-dependent speech patterns.
4. **Classification:** The final hidden state is passed through a fully connected layer and Softmax activation to predict the word class.

### Results

After training, refer to the `reports/figures` directory for:

- **Training History:** Visualizations of loss and accuracy over epochs.
- **Confusion Matrix:** A heatmap displaying the classification performance across different word classes.
