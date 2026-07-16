from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from config.config import (
    CHECKPOINT_DIR,
    CLASS_NAMES,
    DEVICE,
    IMAGE_SIZE,
    NUM_CLASSES,
    PRETRAINED_CHECKPOINT,
    RETF_FOUND_REPO,
    create_directories,
    validate_paths,
)
from utils.dataset import get_evaluation_transform
from models.retfound_classifier import RETFoundClassifier


from config.config import EXPLAINABILITY_DIR

OUTPUT_DIR = EXPLAINABILITY_DIR

MODEL_CHECKPOINT = (
    CHECKPOINT_DIR / "retfound_head_best.pth"
)


class AttentionCollector:
    """
    Collect attention matrices from all RETFound transformer blocks.
    """

    def __init__(self, model: RETFoundClassifier) -> None:
        self.attention_maps: list[torch.Tensor] = []
        self.hooks = []

        for block in model.model.blocks:
            # Modern timm may use fused attention, which does not expose
            # the attention matrix through attn_drop.
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False

            hook = block.attn.attn_drop.register_forward_hook(
                self._save_attention
            )

            self.hooks.append(hook)

    def _save_attention(
        self,
        module,
        inputs,
        output,
    ) -> None:
        if isinstance(output, torch.Tensor):
            self.attention_maps.append(
                output.detach().cpu()
            )

    def clear(self) -> None:
        self.attention_maps.clear()

    def remove(self) -> None:
        for hook in self.hooks:
            hook.remove()

        self.hooks.clear()


def load_model() -> RETFoundClassifier:
    """Load the best head-only RETFound model."""

    if not MODEL_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {MODEL_CHECKPOINT}"
        )

    model = RETFoundClassifier(
        repository_path=RETF_FOUND_REPO,
        checkpoint_path=PRETRAINED_CHECKPOINT,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        MODEL_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    del checkpoint

    app_device = torch.device("cpu")

    model.to(app_device)
    model.eval()

    return model


def calculate_attention_rollout(
    attention_maps: list[torch.Tensor],
) -> np.ndarray:
    """
    Combine attention matrices from all transformer layers.

    Each layer contributes residual attention, and the matrices
    are multiplied to estimate how information flows from the
    class token to retinal image patches.
    """

    if not attention_maps:
        raise RuntimeError(
            "No attention matrices were captured. "
            "Make sure fused attention is disabled."
        )

    processed_layers = []

    for attention in attention_maps:
        # Expected shape:
        # [batch_size, heads, tokens, tokens]
        if attention.ndim != 4:
            raise RuntimeError(
                "Unexpected attention shape: "
                f"{tuple(attention.shape)}"
            )

        # Average across attention heads.
        attention = attention.mean(dim=1)

        number_of_tokens = attention.size(-1)

        identity = torch.eye(
            number_of_tokens,
            dtype=attention.dtype,
        ).unsqueeze(0)

        # Residual connection.
        attention = attention + identity

        # Normalize rows.
        attention = attention / attention.sum(
            dim=-1,
            keepdim=True,
        )

        processed_layers.append(attention)

    joint_attention = processed_layers[0]

    for attention in processed_layers[1:]:
        joint_attention = torch.bmm(
            attention,
            joint_attention,
        )

    # Attention from CLS token to all image patches.
    cls_attention = joint_attention[0, 0, 1:]

    number_of_patches = cls_attention.numel()
    grid_size = int(math.sqrt(number_of_patches))

    if grid_size * grid_size != number_of_patches:
        raise RuntimeError(
            "Patch count is not a square number: "
            f"{number_of_patches}"
        )

    attention_map = cls_attention.reshape(
        grid_size,
        grid_size,
    )

    attention_map = attention_map.numpy()

    attention_map -= attention_map.min()

    maximum = attention_map.max()

    if maximum > 0:
        attention_map /= maximum

    return attention_map


def create_overlay(
    original_image: Image.Image,
    attention_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a heatmap and overlay it on the retinal image."""

    original_rgb = np.array(
        original_image.convert("RGB")
    )

    height, width = original_rgb.shape[:2]

    resized_attention = cv2.resize(
        attention_map,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    resized_attention = np.clip(
        resized_attention,
        0,
        1,
    )

    heatmap_uint8 = np.uint8(
        255 * resized_attention
    )

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay = cv2.addWeighted(
        original_rgb,
        0.60,
        heatmap_rgb,
        0.40,
        0,
    )

    return heatmap_rgb, overlay


def save_visualization(
    original_image: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    predicted_class: str,
    confidence: float,
    probabilities: dict[str, float],
    output_path: Path,
) -> None:
    """
    Save a professional four-panel explainability report.

    The attention overlay indicates regions that influenced the model.
    It does not confirm the location of clinical lesions.
    """

    figure = plt.figure(
        figsize=(14, 10),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1, 1],
        height_ratios=[1, 1],
    )

    # ---------------------------------------------------------
    # Panel 1: Original retinal image
    # ---------------------------------------------------------
    axis_original = figure.add_subplot(grid[0, 0])

    axis_original.imshow(original_image)
    axis_original.set_title(
        "Original Fundus Image",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    axis_original.axis("off")

    # ---------------------------------------------------------
    # Panel 2: Prediction summary and probabilities
    # ---------------------------------------------------------
    axis_summary = figure.add_subplot(grid[0, 1])
    axis_summary.axis("off")

    probability_lines = []

    for class_name, probability in probabilities.items():
        probability_lines.append(
            f"{class_name:<22} {probability:>7.2%}"
        )

    probability_text = "\n".join(probability_lines)

    summary_text = (
        "RETFound Prediction\n\n"
        f"Predicted grade:\n{predicted_class}\n\n"
        f"Confidence:\n{confidence:.2%}\n\n"
        "Class probabilities\n"
        "────────────────────────────\n"
        f"{probability_text}\n\n"
        "Interpretation\n"
        "The attention map highlights image regions that\n"
        "contributed relatively more to the model output.\n"
        "It does not confirm the presence or location of\n"
        "a retinal lesion."
    )

    axis_summary.text(
        0.04,
        0.96,
        summary_text,
        transform=axis_summary.transAxes,
        fontsize=11,
        verticalalignment="top",
        horizontalalignment="left",
        family="monospace",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.95,
        },
    )

    # ---------------------------------------------------------
    # Panel 3: Pure attention heatmap
    # ---------------------------------------------------------
    axis_heatmap = figure.add_subplot(grid[1, 0])

    axis_heatmap.imshow(heatmap)
    axis_heatmap.set_title(
        "RETFound Attention Heatmap",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    axis_heatmap.axis("off")

    # ---------------------------------------------------------
    # Panel 4: Overlay
    # ---------------------------------------------------------
    axis_overlay = figure.add_subplot(grid[1, 1])

    axis_overlay.imshow(overlay)
    axis_overlay.set_title(
        "Attention Overlay",
        fontsize=15,
        fontweight="bold",
        pad=12,
    )
    axis_overlay.axis("off")

    figure.suptitle(
        "RetinaGuard AI — Explainable Diabetic Retinopathy Screening",
        fontsize=19,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.01,
        (
            "Research prototype only. This result is not a medical "
            "diagnosis and should be reviewed by a qualified ophthalmologist."
        ),
        ha="center",
        fontsize=10,
        fontstyle="italic",
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    figure.savefig(
        output_path,
        dpi=220,
        bbox_inches="tight",
    )

    plt.close(figure)

def explain_image(
    model: RETFoundClassifier,
    image_path: Path,
) -> dict:
    """Predict one image and generate its attention visualization."""

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    original_image = Image.open(
        image_path
    ).convert("RGB")

    transform = get_evaluation_transform()

    model_device = next(model.parameters()).device

    input_tensor = transform(
        original_image
    ).unsqueeze(0).to(model_device)

    collector = AttentionCollector(model)

    try:
        collector.clear()

        with torch.inference_mode():
            logits = model(input_tensor)

            probabilities = F.softmax(
                logits,
                dim=1,
            )[0]

        predicted_index = int(
            torch.argmax(probabilities).item()
        )

        predicted_class = CLASS_NAMES[
            predicted_index
        ]

        confidence = float(
            probabilities[predicted_index].item()
        )

        attention_map = calculate_attention_rollout(
            collector.attention_maps
        )
        attention_layer_count = len(
            collector.attention_maps
        )
    finally:
        collector.remove()

    heatmap, overlay = create_overlay(
        original_image=original_image,
        attention_map=attention_map,
    )

    probability_results = {
        CLASS_NAMES[index]: float(
            probabilities[index].item()
        )
        for index in range(NUM_CLASSES)
    }

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_clinical_explanation.png"
    )

    save_visualization(
        original_image=original_image,
        heatmap=heatmap,
        overlay=overlay,
        predicted_class=predicted_class,
        confidence=confidence,
        probabilities=probability_results,
        output_path=output_path,
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probabilities": probability_results,
        "attention_layers": attention_layer_count,
        "output_path": output_path,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate RETFound attention rollout "
            "for a retinal image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to a retinal image.",
    )

    return parser.parse_args()


def main() -> None:
    create_directories()
    validate_paths()
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    arguments = parse_arguments()

    print("=" * 65)
    print("RETINAGUARD — ATTENTION EXPLAINABILITY")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Image: {arguments.image}")

    model = load_model()

    result = explain_image(
        model=model,
        image_path=arguments.image,
    )

    print("\nPrediction:")
    print(result["predicted_class"])

    print("\nConfidence:")
    print(f"{result['confidence']:.2%}")

    print("\nClass probabilities:")

    for class_name, probability in (
        result["probabilities"].items()
    ):
        print(
            f"{class_name:<22} "
            f"{probability:.2%}"
        )

    print("\nAttention layers captured:")
    print(result["attention_layers"])

    print("\nVisualization saved to:")
    print(result["output_path"])


if __name__ == "__main__":
    main()