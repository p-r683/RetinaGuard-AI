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
    """
    Select the model device.

    CPU is used by default because Streamlit Community Cloud
    does not normally provide a GPU.
    """

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def get_huggingface_token() -> str | None:
    """
    Read the Hugging Face access token.

    It first checks the HF_TOKEN environment variable.
    If running inside Streamlit, it also checks Streamlit Secrets.
    """

    token = os.getenv("HF_TOKEN")

    if token:
        return token

    try:
        import streamlit as st

        return st.secrets.get("HF_TOKEN")
    except Exception:
        return None


def resolve_inference_checkpoint() -> Path:
    """
    Return the local inference checkpoint.

    If the checkpoint is not stored locally, download it from
    Hugging Face Hub and use the Hugging Face cache location.
    """

    if INFERENCE_CHECKPOINT.exists():
        print(
            "Using local inference checkpoint:",
            INFERENCE_CHECKPOINT,
        )

        return INFERENCE_CHECKPOINT

    print(
        "Local inference checkpoint not found. "
        "Downloading from Hugging Face Hub..."
    )

    token = get_huggingface_token()

    try:
        downloaded_path = hf_hub_download(
            repo_id=HF_MODEL_REPO,
            filename="retfound_inference.pth",
            repo_type="model",
            token=token,
        )
    except Exception as error:
        raise RuntimeError(
            "Unable to download the RetinaGuard checkpoint from "
            f"Hugging Face repository '{HF_MODEL_REPO}'. "
            "Confirm that the repository exists, the checkpoint "
            "filename is 'retfound_inference.pth', and HF_TOKEN "
            "has read permission."
        ) from error

    checkpoint_path = Path(downloaded_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "The Hugging Face download completed, but the "
            f"checkpoint could not be found at: {checkpoint_path}"
        )

    print(
        "Inference checkpoint downloaded successfully:",
        checkpoint_path,
    )

    return checkpoint_path


def load_inference_model(
    device: torch.device | None = None,
) -> RETFoundClassifier:
    """
    Build the RETFound architecture and load final inference weights.

    The original 3.8 GB RETFound pretraining checkpoint is not loaded.
    The complete encoder and classification-head weights come from
    retfound_inference.pth.
    """

    selected_device = (
        device
        if device is not None
        else get_model_device(prefer_gpu=False)
    )

    checkpoint_path = resolve_inference_checkpoint()

    print(
        "Loading inference checkpoint on CPU:",
        checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if not isinstance(checkpoint, dict):
        raise TypeError(
            "The inference checkpoint must contain a dictionary."
        )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "Inference checkpoint does not contain "
            "'model_state_dict'."
        )

    print("Creating RETFound ViT-Large architecture...")

    model = RETFoundClassifier(
        repository_path=RETF_FOUND_REPO,
        checkpoint_path=None,
        num_classes=NUM_CLASSES,
    )

    load_result = model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    if load_result.missing_keys:
        raise RuntimeError(
            "Missing parameters while loading inference model: "
            f"{load_result.missing_keys}"
        )

    if load_result.unexpected_keys:
        raise RuntimeError(
            "Unexpected parameters while loading inference model: "
            f"{load_result.unexpected_keys}"
        )

    del checkpoint

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.to(selected_device)
    model.eval()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(
        "RETFound inference model loaded successfully."
    )
    print(
        "Model device:",
        next(model.parameters()).device,
    )

    return model


@torch.inference_mode()
def predict_image(
    model: RETFoundClassifier,
    image: Image.Image | Path | str,
) -> dict[str, Any]:
    """
    Predict diabetic-retinopathy severity for one fundus image.

    Returns the predicted class, confidence, class probabilities,
    predicted index and CPU logits.
    """

    if isinstance(image, (str, Path)):
        image_path = Path(image)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as opened_image:
            pil_image = opened_image.convert("RGB")

    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")

    else:
        raise TypeError(
            "image must be a PIL Image, pathlib.Path, "
            "or string file path."
        )

    transform = get_evaluation_transform()

    model_device = next(
        model.parameters()
    ).device

    input_tensor = transform(
        pil_image
    ).unsqueeze(0).to(model_device)

    logits = model(input_tensor)

    probability_tensor = F.softmax(
        logits,
        dim=1,
    )[0]

    predicted_index = int(
        torch.argmax(
            probability_tensor
        ).item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = float(
        probability_tensor[
            predicted_index
        ].item()
    )

    probabilities = {
        CLASS_NAMES[index]: float(
            probability_tensor[index].item()
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
    """Return information about the loaded deployment model."""

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
        "checkpoint_source": str(
            INFERENCE_CHECKPOINT
        ),
        "huggingface_repository": HF_MODEL_REPO,
        "classes": CLASS_NAMES,
    }
