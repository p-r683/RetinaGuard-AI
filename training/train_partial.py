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
PARTIAL_EPOCHS = 3
from config.config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    DEVICE,
    GRADIENT_ACCUMULATION_STEPS,
    GRADIENT_CLIP_NORM,
   
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
    print("RETINAGUARD — PARTIAL ENCODER FINE-TUNING")
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

    # Load the best classification-head checkpoint produced earlier.
    head_checkpoint_path = (
        CHECKPOINT_DIR / "retfound_head_best.pth"
    )

    if not head_checkpoint_path.exists():
        raise FileNotFoundError(
            f"Head checkpoint not found: {head_checkpoint_path}"
        )

    head_checkpoint = torch.load(
        head_checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        head_checkpoint["model_state_dict"]
    )

    print(
        f"Loaded trained classification head from: "
        f"{head_checkpoint_path}"
    )

    # Keep most of RETFound frozen and unfreeze only the final block.
    model.unfreeze_encoder(last_blocks=1)

    model.to(DEVICE)

    print(
        f"Total parameters: "
        f"{model.total_parameter_count():,}"
    )

    print(
        f"Trainable parameters: "
        f"{model.trainable_parameter_count():,}"
    )
    print("\nTrainable layers:")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(f"  {name}")

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

    encoder_parameters = []
    head_parameters = []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue

        if (
            "model.head" in name
            or "model.fc_norm" in name
        ):
            head_parameters.append(parameter)
        else:
            encoder_parameters.append(parameter)

    print(
        f"\nTrainable encoder tensors: "
        f"{len(encoder_parameters)}"
    )
    print(
        f"Trainable head tensors: "
        f"{len(head_parameters)}"
    )

    optimizer = AdamW(
        [
            {
                "params": encoder_parameters,
                "lr": 2e-6,
            },
            {
                "params": head_parameters,
                "lr": 2e-5,
            },
        ],
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

    for epoch in range(1, PARTIAL_EPOCHS + 1):
        print(
            f"\nEpoch {epoch}/{PARTIAL_EPOCHS}"
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

        encoder_learning_rate = (
            optimizer.param_groups[0]["lr"]
        )

        head_learning_rate = (
            optimizer.param_groups[1]["lr"]
        )

        epoch_result = {
            "epoch": epoch,
            "encoder_learning_rate": encoder_learning_rate,
            "head_learning_rate": head_learning_rate,
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
            f"Encoder learning rate: "
            f"{encoder_learning_rate:.8f}"
        )

        print(
            f"Head learning rate: "
            f"{head_learning_rate:.8f}"
        )

        last_checkpoint = (
            CHECKPOINT_DIR / "retfound_partial_low_lr_last.pth"
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
                CHECKPOINT_DIR / "retfound_partial_low_lr_best.pth"
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
        CHECKPOINT_DIR / "partial_training_lr_history.json"
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
    print("PARTIAL FINE-TUNING COMPLETE")
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
        f"{CHECKPOINT_DIR / 'retfound_partial_low_lr_best.pth'}"
    )


if __name__ == "__main__":
    main()