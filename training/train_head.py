from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config.config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    DEVICE,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP_NORM,
    HEAD_EPOCHS,
    HEAD_LEARNING_RATE,
    NUM_CLASSES,
    PRETRAINED_CHECKPOINT,
    RANDOM_SEED,
    RETF_FOUND_REPO,
    USE_AMP,
    WEIGHT_DECAY,
    create_directories,
    validate_paths,
)
from utils.dataset import calculate_class_weights, create_dataloaders
from models.retfound_classifier import RETFoundClassifier


def set_seed(seed: int) -> None:
    """Make training more reproducible."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_f1: float,
    destination: Path,
) -> None:
    """Save a training checkpoint."""

    destination.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_loss": validation_loss,
            "validation_f1": validation_f1,
            "class_names": CLASS_NAMES,
        },
        destination,
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> tuple[float, float, float]:
    """Train the classification head for one epoch."""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    progress_bar = tqdm(
        loader,
        desc="Training",
        leave=False,
    )

    for step, (images, targets) in enumerate(progress_bar):
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

            scaled_loss = (
                loss / GRADIENT_ACCUMULATION_STEPS
            )

        scaler.scale(scaled_loss).backward()

        should_update = (
            (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0
            or (step + 1) == len(loader)
        )

        if should_update:
            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=GRADIENT_CLIP_NORM,
            )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(
            predictions.detach().cpu().tolist()
        )

        all_targets.extend(
            targets.detach().cpu().tolist()
        )

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}"
        )

    epoch_loss = total_loss / len(loader.dataset)

    epoch_accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    epoch_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return epoch_loss, epoch_accuracy, epoch_f1


@torch.inference_mode()
def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float]:
    """Evaluate the model on the validation set."""

    model.eval()

    total_loss = 0.0
    all_predictions: list[int] = []
    all_targets: list[int] = []

    progress_bar = tqdm(
        loader,
        desc="Validation",
        leave=False,
    )

    for images, targets in progress_bar:
        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=USE_AMP,
        ):
            logits = model(images)
            loss = criterion(logits, targets)

        total_loss += loss.item() * images.size(0)

        predictions = torch.argmax(logits, dim=1)

        all_predictions.extend(
            predictions.cpu().tolist()
        )

        all_targets.extend(
            targets.cpu().tolist()
        )

    validation_loss = total_loss / len(loader.dataset)

    validation_accuracy = accuracy_score(
        all_targets,
        all_predictions,
    )

    validation_f1 = f1_score(
        all_targets,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return (
        validation_loss,
        validation_accuracy,
        validation_f1,
    )


def main() -> None:
    set_seed(RANDOM_SEED)
    create_directories()
    validate_paths()

    print("=" * 65)
    print("RETINAGUARD — HEAD-ONLY TRAINING")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"AMP enabled: {USE_AMP}")

    train_loader, validation_loader, _ = (
        create_dataloaders()
    )

    print(f"Training images: {len(train_loader.dataset)}")
    print(
        f"Validation images: "
        f"{len(validation_loader.dataset)}"
    )

    print("\nCreating RETFound classifier...")

    model = RETFoundClassifier(
        repository_path=RETF_FOUND_REPO,
        checkpoint_path=PRETRAINED_CHECKPOINT,
        num_classes=NUM_CLASSES,
    )

    model.freeze_encoder()
    model.to(DEVICE)

    print(
        f"Total parameters: "
        f"{model.total_parameter_count():,}"
    )

    print(
        f"Trainable parameters: "
        f"{model.trainable_parameter_count():,}"
    )

    class_weights = calculate_class_weights().to(DEVICE)

    print("\nClass weights:")
    for class_name, weight in zip(
        CLASS_NAMES,
        class_weights.tolist(),
    ):
        print(f"{class_name}: {weight:.4f}")

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=0.05,
    )

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    optimizer = AdamW(
        trainable_parameters,
        lr=HEAD_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=1,
    )

    scaler = torch.amp.GradScaler(
        device=DEVICE.type,
        enabled=USE_AMP,
    )

    best_validation_f1 = -1.0
    history: list[dict] = []

    training_start = time.time()

    for epoch in range(1, HEAD_EPOCHS + 1):
        print(
            f"\nEpoch {epoch}/{HEAD_EPOCHS}"
        )
        print("-" * 65)

        train_loss, train_accuracy, train_f1 = (
            train_one_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                scaler=scaler,
                device=DEVICE,
            )
        )

        validation_loss, validation_accuracy, validation_f1 = (
            validate(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=DEVICE,
            )
        )

        scheduler.step(validation_loss)

        current_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        epoch_result = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "train_macro_f1": train_f1,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "validation_macro_f1": validation_f1,
        }

        history.append(epoch_result)

        print(
            f"Train loss: {train_loss:.4f} | "
            f"Accuracy: {train_accuracy:.4f} | "
            f"Macro F1: {train_f1:.4f}"
        )

        print(
            f"Validation loss: {validation_loss:.4f} | "
            f"Accuracy: {validation_accuracy:.4f} | "
            f"Macro F1: {validation_f1:.4f}"
        )

        print(
            f"Learning rate: "
            f"{current_learning_rate:.8f}"
        )

        last_checkpoint = (
            CHECKPOINT_DIR / "retfound_head_last.pth"
        )

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
            validation_f1=validation_f1,
            destination=last_checkpoint,
        )

        if validation_f1 > best_validation_f1:
            best_validation_f1 = validation_f1

            best_checkpoint = (
                CHECKPOINT_DIR / "retfound_head_best.pth"
            )

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_f1=validation_f1,
                destination=best_checkpoint,
            )

            print(
                "Best checkpoint updated: "
                f"{best_checkpoint}"
            )

    elapsed_minutes = (
        time.time() - training_start
    ) / 60

    history_path = (
        CHECKPOINT_DIR / "head_training_history.json"
    )

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            history,
            file,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("HEAD TRAINING COMPLETE")
    print("=" * 65)
    print(
        f"Best validation macro F1: "
        f"{best_validation_f1:.4f}"
    )
    print(
        f"Training duration: "
        f"{elapsed_minutes:.2f} minutes"
    )
    print(
        f"Best model: "
        f"{CHECKPOINT_DIR / 'retfound_head_best.pth'}"
    )


if __name__ == "__main__":
    main()