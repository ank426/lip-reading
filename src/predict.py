import torch
import argparse
import os
from src import config
from src.models.model import LipReadingModel
from src.data.dataset import load_video_frames
from src.utils.helpers import load_class_mappings

def predict(video_path):
    try:
        class_labels = load_class_mappings()
    except Exception as e:
        print(f"Error: {e}")
        return

    num_classes = len(class_labels)
    print(f"Loaded {num_classes} classes.")

    model = LipReadingModel(num_classes=num_classes).to(config.DEVICE)
    if not os.path.exists(config.BEST_MODEL_PATH):
        print("Model file not found.")
        return

    model.load_state_dict(torch.load(config.BEST_MODEL_PATH, map_location=config.DEVICE))
    model.eval()

    if not os.path.exists(video_path):
        print(f"Video not found: {video_path}")
        return

    print(f"Processing {video_path}...")
    video_tensor = load_video_frames(video_path)
    video_tensor = video_tensor.unsqueeze(0).to(config.DEVICE) # Add batch dim

    with torch.no_grad():
        outputs = model(video_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()

    print(f"\nPrediction: {class_labels[pred_idx]}")
    print(f"Confidence: {probs[pred_idx]:.4f}")

    print("\nTop 3 Predictions:")
    top_idxs = torch.argsort(probs, descending=True)[:3]
    for idx in top_idxs:
        print(f"{class_labels[idx]}: {probs[idx]:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Lip Reading Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    args = parser.parse_args()

    predict(args.video)
