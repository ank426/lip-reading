import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict

from src import config
from src.data.dataset import LipReadingDataset
from src.models.model import LipReadingModel
from src.utils.helpers import plot_history, save_class_mappings

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for videos, labels in tqdm(loader, desc="Training"):
        videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

        optimizer.zero_grad()
        outputs = model(videos)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * videos.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for videos, labels in tqdm(loader, desc="Validating"):
            videos, labels = videos.to(config.DEVICE), labels.to(config.DEVICE)

            outputs = model(videos)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * videos.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def main():
    print(f"Initializing training on {config.DEVICE}...")

    train_dataset = LipReadingDataset('train')
    val_dataset = LipReadingDataset('val', selected_classes=train_dataset.classes)

    save_class_mappings(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                          shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY)

    model = LipReadingModel(num_classes=len(train_dataset.classes)).to(config.DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3)

    history = defaultdict(list)
    best_acc = 0.0

    for epoch in range(config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{config.EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"🌟 New best model saved to {config.BEST_MODEL_PATH}")

    plot_history(history)
    print("Training Complete.")

if __name__ == "__main__":
    main()
