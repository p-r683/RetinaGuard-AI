from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.nn import functional as F

from config.config import (
    CLASS_NAMES,
    HF_MODEL_REPO,
    INFERENCE_CHECKPOINT,
    NUM_CLASSES,
    RETF_FOUND_REPO,
)
from models.retfound_classifier import RETFoundClassifier
from utils.dataset import get_evaluation_transform


def get_model_device(
    prefer_gpu: bool = False,
) -> torch.device:
    """Choose CPU by default for low-memory deployment."""

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def resolve_inference_checkpoint() -> Path:
    """
    Return the local checkpoint or download it from Hugging Face.
    """

    if INFERENCE_CHECKPOINT.exists():
        return INFERENCE_CHECKPOINT

    token = os.getenv("HF_TOKEN")

    downloaded_path = hf_hub_download(
        repo_id=HF_MODEL_REPO,
        filename="retfound_inference.pth",
        repo_type="model",
        token=token,
    )

    return Path(downloaded_path)


def load_inference_model(
    device: torch.device | None = None,
) -> RETFoundClassifier:
    """Build RETFound and load the final inference weights."""

    selected_device = (
        device
        if device is not None
        else get_model_device(prefer_gpu=False)
    )

    checkpoint_path = resolve_inference_checkpoint()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Inference checkpoint does not contain "
            "'model_state_dict'."
        )

    model = RETFoundClassifier(
        repository_path=RETF_FOUND_REPO,
        checkpoint_path=None,
        num_classes=NUM_CLASSES,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
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
    """Predict the DR grade for one retinal image."""

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

    device = next(model.parameters()).device

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(device)

    logits = model(input_tensor)

    probability_tensor = F.softmax(
        logits,
        dim=1,
    )[0]

    predicted_index = int(
        torch.argmax(probability_tensor).item()
    )

    probabilities = {
        CLASS_NAMES[index]: float(
            probability_tensor[index].item()
        )
        for index in range(NUM_CLASSES)
    }

    return {
        "predicted_index": predicted_index,
        "predicted_class": CLASS_NAMES[predicted_index],
        "confidence": float(
            probability_tensor[predicted_index].item()
        ),
        "probabilities": probabilities,
        "logits": logits.detach().cpu(),
    }
