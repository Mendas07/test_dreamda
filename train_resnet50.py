import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import random_split, DataLoader
from typing import Tuple
from pathlib import Path

# -----------------------------
# Constants and Configuration
# -----------------------------
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 1e-3
AUGMENT_FACTOR = 30
DATASET_NAME = "Pets"
DATASET_PATH = "datasets/pets"
SEEDS = [0, 1, 2]
USE_PRETRAINED = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Utility Functions
# -----------------------------
def set_seed(seed: int) -> None:
    """Set seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_dataloaders(data_dir: str, batch_size: int, augment_factor: int) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train and test data loaders for image classification.

    Args:
        data_dir (str): Path to dataset directory.
        batch_size (int): Number of samples per batch.
        augment_factor (int): Factor to multiply dataset size for augmentation.

    Returns:
        tuple: (train_loader, test_loader, num_classes)
    """
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    transform_train = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
    ])

    full_dataset = datasets.ImageFolder(root=data_dir, transform=transform_train)
    num_classes = len(full_dataset.classes)

    # Proper train-test split (80/20)
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])

    # Override transform for test set
    test_dataset.dataset.transform = transform_test

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, num_classes


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer, device: torch.device) -> None:
    """Train model for one epoch."""
    model.train()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    """Evaluate model and return top-1 accuracy."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    return correct / total


def run_experiment(seed: int) -> float:
    """Run training and evaluation pipeline with a given seed."""
    set_seed(seed)

    train_loader, test_loader, num_classes = get_dataloaders(DATASET_PATH, BATCH_SIZE, AUGMENT_FACTOR)

    weights = models.ResNet50_Weights.IMAGENET1K_V1 if USE_PRETRAINED else None
    model = models.resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)

    accuracy = evaluate(model, test_loader, DEVICE)
    return accuracy


# -----------------------------
# Main Entry Point
# -----------------------------
def main():
    print(f"=== Dataset: {DATASET_NAME} | Mode: Pretrained Only ===")
    for seed in SEEDS:
        print(f"--- Seed: {seed} ---")
        acc = run_experiment(seed)
        print(f"Seed {seed} | Accuracy: {acc:.4f}\n")


if __name__ == "__main__":
    main()
