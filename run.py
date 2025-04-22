import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2

# Import classes and functions from the original code
from src import (
    device,
    SimpleLipReadingModel,
    load_video,
    transform,
    IMAGE_SIZE,
    extract_mouth_region
)

# ======= CONFIGURATION VARIABLES (Edit these) =======
# Path to the trained model weights
MODEL_PATH = 'best_lip_reading_model.pt'

# Path to the video file for testing
VIDEO_PATH = 'data/lipread_mp4/AGAIN/test/AGAIN_00001.mp4'

# List of class labels in the same order as used during training
CLASS_LABELS = ['ACCESS', 'AGAIN', 'AMOUNT', 'ANSWER', 'ARRESTED']

# Whether to visualize mouth extraction and predictions
VISUALIZE = False
# ===================================================

def predict_video(model, video_path, class_labels, visualize=False):
    """Predict class for a single video and optionally visualize results"""
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

    if visualize:
        visualize_prediction(video_path, class_labels, probabilities.cpu().numpy())

    return predicted_word, confidence

def visualize_prediction(video_path, class_labels, probabilities):
    """Visualize video frames with mouth region extraction and prediction probabilities"""
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    # Get top 5 predictions (or all if less than 5 classes)
    top_k = min(5, len(class_labels))
    top_indices = np.argsort(probabilities)[-top_k:][::-1]
    top_classes = [class_labels[idx] for idx in top_indices]
    top_probs = [probabilities[idx] for idx in top_indices]

    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    # Original frame
    original_img = ax1.imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    ax1.set_title('Original Frame')
    ax1.axis('off')

    # Extracted mouth region
    mouth_img = ax2.imshow(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8))
    ax2.set_title('Extracted Mouth Region')
    ax2.axis('off')

    # Prediction probabilities
    bars = ax3.barh(range(top_k), top_probs, align='center')
    ax3.set_yticks(range(top_k))
    ax3.set_yticklabels(top_classes)
    ax3.set_title('Top Predictions')
    ax3.set_xlim(0, 1)

    # Initialize frame counter
    frame_count = 0

    def update(i):
        nonlocal frame_count
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop back to beginning
            ret, frame = cap.read()
            frame_count = 0

        frame_count += 1

        # Convert from BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract mouth region
        try:
            mouth_region = extract_mouth_region(frame_rgb)

            # Update images
            original_img.set_array(cv2.resize(frame_rgb, (IMAGE_SIZE, IMAGE_SIZE)))
            mouth_img.set_array(cv2.resize(mouth_region, (IMAGE_SIZE, IMAGE_SIZE)))

            # Update title with frame counter
            ax1.set_title(f'Original Frame {frame_count}')
        except Exception as e:
            print(f"Error processing frame: {e}")

        return original_img, mouth_img, bars

    # Create animation
    ani = FuncAnimation(fig, update, interval=100, blit=False)
    plt.tight_layout()
    plt.show()

    # Release resources
    cap.release()

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
    predict_video(model, VIDEO_PATH, CLASS_LABELS, VISUALIZE)

if __name__ == "__main__":
    main()
