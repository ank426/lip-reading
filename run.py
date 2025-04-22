import os
import torch

# Import classes and functions from the original code
from src import (
    device,
    SimpleLipReadingModel,
    load_video,
    transform
)

# ======= CONFIGURATION VARIABLES (Edit these) =======
# Path to the trained model weights
MODEL_PATH = 'best_lip_reading_model.pt'

# Path to the video file for testing
VIDEO_PATH = 'data/lipread_mp4/AGAIN/test/AGAIN_00001.mp4'

# List of class labels in the same order as used during training
CLASS_LABELS = ['ACCESS', 'AGAIN', 'AMOUNT', 'ANSWER', 'ARRESTED']
# ===================================================

def predict_video(model, video_path, class_labels):
    """Predict class for a single video"""
    # Load and preprocess video
    print(f"Processing video: {video_path}")
    video_tensor = load_video(video_path, transform)

    # Add batch dimension and move to device
    video_tensor = video_tensor.unsqueeze(0).to(device)

    # Set model to evaluation mode
    model.eval()

    # Make prediction
    with torch.no_grad():
        outputs = model(video_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_idx = torch.argmax(outputs, dim=1).item()

    predicted_word = class_labels[predicted_idx]
    confidence = probabilities[predicted_idx].item()

    # Display results
    print(f"Predicted word: {predicted_word}")
    print(f"Confidence: {confidence:.4f}")

    # Show all predictions sorted by confidence
    print("\nAll predictions:")
    sorted_indices = torch.argsort(probabilities, descending=True)
    for i, idx in enumerate(sorted_indices.cpu().numpy()):
        print(f"{i+1}. {class_labels[idx]}: {probabilities[idx]:.4f}")

    return predicted_word, confidence

def main():
    # Check if files exist
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        return

    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    # Print class labels
    print(f"Using {len(CLASS_LABELS)} class labels: {CLASS_LABELS}")

    # Load model
    num_classes = len(CLASS_LABELS)
    model = SimpleLipReadingModel(num_classes=num_classes).to(device)

    try:
        # Load weights
        print(f"Loading model weights from {MODEL_PATH}")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run prediction
    predict_video(model, VIDEO_PATH, CLASS_LABELS)

if __name__ == "__main__":
    main()
