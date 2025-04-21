import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.0
SEQUENCE_LENGTH = 20  # Maximum number of frames to use
IMAGE_SIZE = 112  # Size of input images
NUM_CLASSES = None  # Will be determined from selected classes
MAX_CLASSES = 5  # Maximum number of classes to use

# Data preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_mouth_region(frame):
    """
    Simplified mouth region extraction that doesn't rely on face detection
    Just takes the lower half of the frame where the mouth is likely to be
    """
    h, w = frame.shape[:2]
    # Take the lower half of the frame where the mouth is usually located
    mouth_region = frame[h//2:, w//4:3*w//4]

    if mouth_region.size == 0:
        # Fallback if crop is empty
        return frame

    return mouth_region

def load_video(video_path, transform, max_frames=SEQUENCE_LENGTH):
    """Load and preprocess video frames"""
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to RGB (from BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract mouth region
        try:
            mouth_region = extract_mouth_region(frame)
            # Apply transformations
            if mouth_region is not None and mouth_region.size > 0:
                processed_frame = transform(mouth_region)
                frames.append(processed_frame)
        except Exception as e:
            print(f"Error processing frame in {video_path}: {e}")
            continue

    cap.release()

    # Pad sequence if needed
    if len(frames) == 0:
        # Handle empty videos by creating dummy frames
        dummy_frame = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        frames = [dummy_frame] * max_frames
    elif len(frames) < max_frames:
        # Pad with zeros
        padding = [torch.zeros_like(frames[0]) for _ in range(max_frames - len(frames))]
        frames.extend(padding)
    elif len(frames) > max_frames:
        # Select evenly spaced frames
        indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames = [frames[i] for i in indices]

    # Stack frames into a tensor [sequence_length, channels, height, width]
    return torch.stack(frames)

class SelectiveLipReadingDataset(Dataset):
    def __init__(self, data_dir, split, selected_classes=None, max_classes=MAX_CLASSES, transform=None, preload=False):
        """
        Args:
            data_dir: Root directory containing word folders
            split: 'train', 'test', or 'val'
            selected_classes: List of specific classes to use (if None, random selection is made)
            max_classes: Maximum number of classes to use (if selected_classes is None)
            transform: Image transformations
            preload: Whether to preload and cache all videos
        """
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.preload = preload

        # Find all available word classes
        all_words = [word for word in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, word))]
        all_words.sort()

        # Select classes
        if selected_classes is not None:
            # Use provided classes (ensuring they exist in the dataset)
            self.words = [word for word in selected_classes if word in all_words]
        else:
            # Random selection of classes
            if max_classes >= len(all_words):
                self.words = all_words  # Use all if max_classes exceeds available classes
            else:
                # Randomly select max_classes
                self.words = random.sample(all_words, max_classes)
                self.words.sort()  # Sort for consistency

        print(f"Using {len(self.words)} classes: {self.words}")

        # Create word to index mapping
        self.word_to_idx = {word: idx for idx, word in enumerate(self.words)}

        # Create samples list
        self.samples = []
        for word in self.words:
            split_dir = os.path.join(data_dir, word, split)
            if os.path.exists(split_dir):
                video_files = [f for f in os.listdir(split_dir) if f.endswith('.mp4')]
                for video_file in video_files:
                    video_path = os.path.join(split_dir, video_file)
                    self.samples.append((video_path, word))

        print(f"Total samples in {split} set: {len(self.samples)}")

        # Preload data if requested
        self.cached_data = {}
        if preload:
            print(f"Preloading {split} dataset...")
            for idx, (video_path, _) in enumerate(tqdm(self.samples)):
                self.cached_data[idx] = load_video(video_path, self.transform)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, word = self.samples[idx]

        # Load and preprocess video
        if self.preload and idx in self.cached_data:
            video_tensor = self.cached_data[idx]
        else:
            video_tensor = load_video(video_path, self.transform)

        # Convert label to tensor
        label = torch.tensor(self.word_to_idx[word], dtype=torch.long)

        return video_tensor, label

# Model architecture
class SimpleLipReadingModel(nn.Module):
    def __init__(self, num_classes, hidden_size=512, dropout=0.5):
        super(SimpleLipReadingModel, self).__init__()

        # Use MobileNetV2 for smaller memory footprint
        mobilenet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        self.features = mobilenet.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Freeze feature extractor
        for param in self.features.parameters():
            param.requires_grad = False

        # Feature size
        self.feature_size = 1280  # Output channels of MobileNetV2

        # GRU - simpler than LSTM
        self.gru = nn.GRU(
            input_size=self.feature_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.size()

        # Process each frame with the CNN
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(batch_size, seq_len, -1)

        # Process sequence with GRU
        output, _ = self.gru(x)

        # Use the last timestep's output
        output = output[:, -1, :]
        output = self.dropout(output)
        output = self.fc(output)

        return output

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    """Train the model and validate"""
    best_val_acc = 0.0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for videos, labels in tqdm(train_loader, desc="Training"):
            videos = videos.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(videos)
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            # Statistics
            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)

        print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for videos, labels in tqdm(val_loader, desc="Validation"):
                videos = videos.to(device)
                labels = labels.to(device)

                outputs = model(videos)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * videos.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_lip_reading_model.pt')
            print("Saved best model checkpoint.")

    return history

def evaluate_model(model, test_loader):
    """Evaluate the model on test data"""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for videos, labels in tqdm(test_loader, desc="Testing"):
            videos = videos.to(device)
            labels = labels.to(device)

            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")

    return accuracy, all_preds, all_labels

def plot_history(history):
    """Plot training and validation metrics"""
    plt.figure(figsize=(12, 5))

    # Plot training & validation loss
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train')
    plt.plot(history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    # Plot training & validation accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train')
    plt.plot(history['val_acc'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    """Plot confusion matrix"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()

def main():
    # Set data directory
    data_dir = "data/lipread_mp4"

    # Define specific classes to use (comment out to use random selection)
    # selected_classes = ["about", "book", "car", "hello", "house"]
    selected_classes = None  # Will randomly select MAX_CLASSES classes

    # Set random seed for reproducibility
    random.seed(42)

    # Create datasets with selected classes
    print("Creating datasets...")
    train_dataset = SelectiveLipReadingDataset(
        data_dir, 'train',
        selected_classes=selected_classes,
        max_classes=MAX_CLASSES,
        transform=transform,
        preload=False
    )

    # Use the same classes for validation and test sets
    val_dataset = SelectiveLipReadingDataset(
        data_dir, 'val',
        selected_classes=train_dataset.words,  # Use the same classes selected for training
        transform=transform,
        preload=False
    )

    test_dataset = SelectiveLipReadingDataset(
        data_dir, 'test',
        selected_classes=train_dataset.words,  # Use the same classes selected for training
        transform=transform,
        preload=False
    )

    # Set number of classes
    global NUM_CLASSES
    NUM_CLASSES = len(train_dataset.words)
    print(f"Number of classes: {NUM_CLASSES}")
    print(f"Classes: {train_dataset.words}")

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Initialize model
    model = SimpleLipReadingModel(NUM_CLASSES).to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # Train the model
    print("Starting training...")
    history = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, EPOCHS)

    # Load best model for evaluation
    model.load_state_dict(torch.load('best_lip_reading_model.pt'))

    # Evaluate on test set
    print("Evaluating on test set...")
    accuracy, all_preds, all_labels = evaluate_model(model, test_loader)

    # Plot results
    plot_history(history)
    plot_confusion_matrix(all_labels, all_preds, test_dataset.words)

    print(f"Final test accuracy: {accuracy:.4f}")
    print("Done!")

if __name__ == "__main__":
    main()
