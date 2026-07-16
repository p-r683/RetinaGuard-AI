from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config.config import (
    BATCH_SIZE,
    DATASET_DIR,
    IMAGE_SIZE,
    NUM_WORKERS,
)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def get_train_transform() -> transforms.Compose:
    """Training preprocessing and mild augmentation."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(
                IMAGE_SIZE,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.05,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def get_evaluation_transform() -> transforms.Compose:
    """Deterministic preprocessing for validation and testing."""

    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=IMAGENET_MEAN,
                std=IMAGENET_STD,
            ),
        ]
    )


def create_dataset(
    split: str,
    transform: Callable,
) -> datasets.ImageFolder:
    """Create an ImageFolder dataset for one split."""

    split_directory = DATASET_DIR / split

    if not split_directory.exists():
        raise FileNotFoundError(
            f"Dataset split does not exist: {split_directory}"
        )

    dataset = datasets.ImageFolder(
        root=split_directory,
        transform=transform,
    )

    expected_classes = ["0", "1", "2", "3", "4"]

    if dataset.classes != expected_classes:
        raise ValueError(
            "Unexpected class folders.\n"
            f"Expected: {expected_classes}\n"
            f"Found: {dataset.classes}"
        )

    return dataset


def create_dataloaders(
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation and test data loaders."""

    train_dataset = create_dataset(
        split="train",
        transform=get_train_transform(),
    )

    validation_dataset = create_dataset(
        split="val",
        transform=get_evaluation_transform(),
    )

    test_dataset = create_dataset(
        split="test",
        transform=get_evaluation_transform(),
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
        drop_last=False,
    )

    return train_loader, validation_loader, test_loader


def calculate_class_weights() -> torch.Tensor:
    """Calculate inverse-frequency weights from the training split."""

    train_dataset = create_dataset(
        split="train",
        transform=get_evaluation_transform(),
    )

    targets = torch.tensor(
        train_dataset.targets,
        dtype=torch.long,
    )

    class_counts = torch.bincount(
        targets,
        minlength=len(train_dataset.classes),
    ).float()

    if torch.any(class_counts == 0):
        raise ValueError(
            f"At least one class has no training samples: {class_counts}"
        )

    weights = class_counts.sum() / (
        len(class_counts) * class_counts
    )

    return weights