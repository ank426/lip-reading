import os
import cv2
import torch
import random
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from src import config
from src.utils.helpers import extract_mouth_region

video_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_video_frames(video_path, transform=video_transform, max_frames=config.SEQUENCE_LENGTH):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened() and len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            mouth = extract_mouth_region(frame)
            if mouth is not None:
                frames.append(transform(mouth))
        except Exception:
            continue

    cap.release()

    # Handle padding or empty videos
    if not frames:
        return torch.zeros(max_frames, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)

    tensor = torch.stack(frames)

    # Pad if too short
    if len(frames) < max_frames:
        padding = torch.zeros(max_frames - len(frames), 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
        tensor = torch.cat((tensor, padding), dim=0)
    # Subsample if too long (though loop above handles this, this is a safety check)
    elif len(frames) > max_frames:
        tensor = tensor[:max_frames]

    return tensor

class LipReadingDataset(Dataset):
    def __init__(self, split, selected_classes=None):
        self.data_dir = config.DATA_DIR
        self.split = split
        self.samples = []

        all_words = sorted([d for d in os.listdir(self.data_dir)
                          if os.path.isdir(os.path.join(self.data_dir, d))])

        if selected_classes:
            self.classes = selected_classes
        elif config.MAX_CLASSES:
            random.seed(42)
            self.classes = sorted(random.sample(all_words, config.MAX_CLASSES))
        else:
            self.classes = all_words

        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        print(f"Scanning {split} data for {len(self.classes)} classes...")
        for word in self.classes:
            word_dir = os.path.join(self.data_dir, word, split)
            if not os.path.exists(word_dir):
                continue

            for file in os.listdir(word_dir):
                if file.endswith('.mp4'):
                    self.samples.append((os.path.join(word_dir, file), word))

        print(f"Found {len(self.samples)} samples for {split}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, word = self.samples[idx]
        video_tensor = load_video_frames(path)
        label = self.class_to_idx[word]
        return video_tensor, torch.tensor(label, dtype=torch.long)
