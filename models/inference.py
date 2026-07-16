from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.nn import functional as F

from config.config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    NUM_CLASSES,
    PRETRAINED_CHECKPOINT,
    RETF_FOUND_REPO,
)
from utils.dataset import get_evaluation_transform
from models.retfound_classifier import RETFoundClassifier


INFERENCE_CHECKPOINT = (
    CHECKPOINT_DIR / "retfound_inference.pth"
)

TRAINING_CHECKPOINT = (
    CHECKPOINT_DIR / "retfound_head_best.pth"
)


def get_model_device(
    prefer_gpu: bool = False,
) -> torch.device:
    """
    Select the inference device.

    CPU is the default because RETFound ViT-Large may exceed
    the available memory of a 4 GB GPU during explainability.
    """

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_inference_checkpoint(
    source_checkpoint: Path = TRAINING_CHECKPOINT,
    destination: Path = INFERENCE_CHECKPOINT,
) -> Path:
    """
    Convert the training checkpoint into a model-only checkpoint.

    The output excludes optimizer state and training metadata.
    """

    if not source_checkpoint.exists():
        raise FileNotFoundError(
            f"Training checkpoint not found: {source_checkpoint}"
        )

    checkpoint = torch.load(
        source_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "The training checkpoint does not contain "
            "'model_state_dict'."
        )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": checkpoint[
                "model_state_dict"
            ],
            "class_names": checkpoint.get(
                "class_names",
                CLASS_NAMES,
            ),
            "num_classes": NUM_CLASSES,
            "model_name": "RETFound CFP ViT-Large/16",
        },
        destination,
    )

    return destination


def load_inference_model(
    device: torch.device | None = None,
    checkpoint_path: Path = INFERENCE_CHECKPOINT,
) -> RETFoundClassifier:
    """
    Build RETFound and load the best inference weights.

    Falls back to the training checkpoint when the model-only
    checkpoint has not yet been created.
    """

    selected_device = (
        device
        if device is not None
        else get_model_device()
    )

    model = RETFoundClassifier(
        repository_path=RETF_FOUND_REPO,
        checkpoint_path=PRETRAINED_CHECKPOINT,
        num_classes=NUM_CLASSES,
    )

    selected_checkpoint = checkpoint_path

    if not selected_checkpoint.exists():
        selected_checkpoint = TRAINING_CHECKPOINT

    if not selected_checkpoint.exists():
        raise FileNotFoundError(
            "No inference or training checkpoint was found.\n"
            f"Inference checkpoint: {checkpoint_path}\n"
            f"Training checkpoint: {TRAINING_CHECKPOINT}"
        )

    checkpoint = torch.load(
        selected_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain model_state_dict: "
            f"{selected_checkpoint}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    del checkpoint

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.to(selected_device)
    model.eval()

    return model


@torch.inference_mode()
def predict_image(
    model: RETFoundClassifier,
    image: Image.Image | Path | str,
) -> dict[str, Any]:
    """
    Predict diabetic-retinopathy severity for one fundus image.

    Returns:
        predicted_index
        predicted_class
        confidence
        probabilities
        logits
    """

    if isinstance(image, (str, Path)):
        image_path = Path(image)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        pil_image = Image.open(
            image_path
        ).convert("RGB")
    else:
        pil_image = image.convert("RGB")

    transform = get_evaluation_transform()

    device = next(
        model.parameters()
    ).device

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    logits = model(input_tensor)

    probabilities_tensor = F.softmax(
        logits,
        dim=1,
    )[0]

    predicted_index = int(
        torch.argmax(
            probabilities_tensor
        ).item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probabilities_tensor[
            predicted_index
        ].item()
    )

    probabilities = {
        CLASS_NAMES[index]: float(
            probabilities_tensor[index].item()
        )
        for index in range(NUM_CLASSES)
    }

    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probabilities,
        "logits": logits.detach().cpu(),
    }


def model_information(
    model: RETFoundClassifier,
) -> dict[str, Any]:
    """Return deployment information about the loaded model."""

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    return {
        "device": str(
            next(model.parameters()).device
        ),
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "checkpoint": str(INFERENCE_CHECKPOINT),
        "classes": CLASS_NAMES,
    }