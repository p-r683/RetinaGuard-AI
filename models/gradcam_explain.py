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


class ViTGradCAM:
    """
    Grad-CAM adapted for RETFound's Vision Transformer.

    The final transformer block's norm1 output contains:
        [CLS token + 196 patch tokens]

    Grad-CAM is calculated from the 196 retinal-image patch tokens.
    """

    def __init__(
        self,
        model: RETFoundClassifier,
    ) -> None:
        self.model = model

        # Final transformer block.
        self.target_layer = (
            self.model.model.blocks[-1].norm1
        )

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None

        self.forward_hook = (
            self.target_layer.register_forward_hook(
                self._forward_hook
            )
        )

    def _forward_hook(
        self,
        module,
        inputs,
        output,
    ) -> torch.Tensor:
        """
        Capture target-layer activations.

        Detaching here prevents PyTorch from storing gradients for all
        previous transformer blocks, which greatly reduces GPU memory.
        """

        detached_output = (
            output.detach().requires_grad_(True)
        )

        self.activations = detached_output

        detached_output.register_hook(
            self._save_gradients
        )

        # Replace the original layer output with the detached tensor.
        return detached_output

    def _save_gradients(
        self,
        gradients: torch.Tensor,
    ) -> None:
        self.gradients = gradients.detach()

    def clear(self) -> None:
        self.activations = None
        self.gradients = None

    def remove(self) -> None:
        self.forward_hook.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int | None = None,
    ) -> tuple[np.ndarray, torch.Tensor, int]:
        """
        Generate class-specific Grad-CAM.

        Returns:
            cam: normalized 2D patch heatmap
            probabilities: class probabilities
            target_class: explained class index
        """

        self.clear()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(input_tensor)

        probabilities = F.softmax(
            logits,
            dim=1,
        )[0]

        if target_class is None:
            target_class = int(
                torch.argmax(probabilities).item()
            )

        target_score = logits[
            0,
            target_class,
        ]

        target_score.backward()

        if self.activations is None:
            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        # Shape: [batch, tokens, channels]
        activations = self.activations[0]
        gradients = self.gradients[0]

        if activations.ndim != 2:
            raise RuntimeError(
                "Unexpected activation shape: "
                f"{tuple(activations.shape)}"
            )

        # Remove CLS token.
        patch_activations = activations[1:, :]
        patch_gradients = gradients[1:, :]

        number_of_patches = patch_activations.shape[0]

        grid_size = int(
            math.sqrt(number_of_patches)
        )

        if grid_size * grid_size != number_of_patches:
            raise RuntimeError(
                "Patch count is not square: "
                f"{number_of_patches}"
            )

        # Standard Grad-CAM:
        # average gradient over spatial patches for each channel.
        channel_weights = patch_gradients.mean(
            dim=0
        )

        cam = torch.sum(
            patch_activations
            * channel_weights.unsqueeze(0),
            dim=1,
        )

        cam = F.relu(cam)

        cam = cam.reshape(
            grid_size,
            grid_size,
        )

        cam = cam.detach().cpu().numpy()

        cam -= cam.min()

        maximum = cam.max()

        if maximum > 0:
            cam /= maximum

        return (
            cam,
            probabilities.detach().cpu(),
            target_class,
        )


def load_model() -> RETFoundClassifier:
    """Load the best head-only RETFound checkpoint."""

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

    # Parameter gradients are unnecessary. Gradients are collected
    # only from the selected transformer activation.
    for parameter in model.parameters():
        parameter.requires_grad = False

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    app_device = torch.device("cpu")

    model.to(app_device)
    model.eval()

    return model


def create_heatmap_and_overlay(
    original_image: Image.Image,
    cam: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Upscale the patch Grad-CAM and overlay it on the image."""

    original_rgb = np.array(
        original_image.convert("RGB")
    )

    height, width = original_rgb.shape[:2]

    resized_cam = cv2.resize(
        cam,
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )

    resized_cam = np.clip(
        resized_cam,
        0,
        1,
    )

    cam_uint8 = np.uint8(
        resized_cam * 255
    )

    heatmap_bgr = cv2.applyColorMap(
        cam_uint8,
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


def save_gradcam_report(
    original_image: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    predicted_class: str,
    explained_class: str,
    confidence: float,
    probabilities: dict[str, float],
    output_path: Path,
) -> None:
    """Save a professional four-panel Grad-CAM report."""

    figure = plt.figure(
        figsize=(14, 10),
        constrained_layout=True,
    )

    grid = figure.add_gridspec(
        nrows=2,
        ncols=2,
    )

    # Original image
    axis_original = figure.add_subplot(
        grid[0, 0]
    )

    axis_original.imshow(original_image)
    axis_original.set_title(
        "Original Fundus Image",
        fontsize=15,
        fontweight="bold",
    )
    axis_original.axis("off")

    # Prediction summary
    axis_summary = figure.add_subplot(
        grid[0, 1]
    )
    axis_summary.axis("off")

    probability_lines = [
        f"{name:<22} {value:>7.2%}"
        for name, value in probabilities.items()
    ]

    summary_text = (
        "RETFound Prediction\n\n"
        f"Predicted grade:\n{predicted_class}\n\n"
        f"Confidence:\n{confidence:.2%}\n\n"
        f"Grad-CAM target:\n{explained_class}\n\n"
        "Class probabilities\n"
        "────────────────────────────\n"
        + "\n".join(probability_lines)
        + "\n\nInterpretation\n"
        "Grad-CAM highlights patches that had a stronger\n"
        "gradient-based influence on the selected class.\n"
        "Highlighted areas are not confirmed lesions."
    )

    axis_summary.text(
        0.04,
        0.96,
        summary_text,
        transform=axis_summary.transAxes,
        fontsize=11,
        verticalalignment="top",
        family="monospace",
        linespacing=1.35,
        bbox={
            "boxstyle": "round,pad=0.8",
            "facecolor": "white",
            "edgecolor": "black",
            "alpha": 0.95,
        },
    )

    # Grad-CAM heatmap
    axis_heatmap = figure.add_subplot(
        grid[1, 0]
    )

    axis_heatmap.imshow(heatmap)
    axis_heatmap.set_title(
        "ViT Grad-CAM Heatmap",
        fontsize=15,
        fontweight="bold",
    )
    axis_heatmap.axis("off")

    # Overlay
    axis_overlay = figure.add_subplot(
        grid[1, 1]
    )

    axis_overlay.imshow(overlay)
    axis_overlay.set_title(
        "Grad-CAM Overlay",
        fontsize=15,
        fontweight="bold",
    )
    axis_overlay.axis("off")

    figure.suptitle(
        "RetinaGuard AI — ViT Grad-CAM Explanation",
        fontsize=19,
        fontweight="bold",
    )

    figure.text(
        0.5,
        0.01,
        (
            "Research prototype only. Grad-CAM indicates relative "
            "model influence and is not a medical diagnosis."
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
    target_class: int | None = None,
) -> dict:
    """Predict an image and generate ViT Grad-CAM."""

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

    gradcam = ViTGradCAM(model)

    try:
        cam, probabilities, explained_index = (
            gradcam.generate(
                input_tensor=input_tensor,
                target_class=target_class,
            )
        )
    finally:
        gradcam.remove()

    predicted_index = int(
        torch.argmax(probabilities).item()
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    explained_class = CLASS_NAMES[
        explained_index
    ]

    confidence = float(
        probabilities[predicted_index].item()
    )

    probability_results = {
        CLASS_NAMES[index]: float(
            probabilities[index].item()
        )
        for index in range(NUM_CLASSES)
    }

    heatmap, overlay = create_heatmap_and_overlay(
        original_image=original_image,
        cam=cam,
    )

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}_gradcam.png"
    )

    save_gradcam_report(
        original_image=original_image,
        heatmap=heatmap,
        overlay=overlay,
        predicted_class=predicted_class,
        explained_class=explained_class,
        confidence=confidence,
        probabilities=probability_results,
        output_path=output_path,
    )

    return {
        "predicted_class": predicted_class,
        "explained_class": explained_class,
        "confidence": confidence,
        "probabilities": probability_results,
        "output_path": output_path,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ViT Grad-CAM for a retinal image."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        type=Path,
        help="Path to the retinal image.",
    )

    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        choices=range(NUM_CLASSES),
        help=(
            "Optional class index to explain. "
            "Defaults to the predicted class."
        ),
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
    print("RETINAGUARD — VIT GRAD-CAM")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Image: {arguments.image}")

    model = load_model()

    result = explain_image(
        model=model,
        image_path=arguments.image,
        target_class=arguments.target_class,
    )

    print("\nPrediction:")
    print(result["predicted_class"])

    print("\nGrad-CAM explained class:")
    print(result["explained_class"])

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

    print("\nGrad-CAM visualization saved to:")
    print(result["output_path"])


if __name__ == "__main__":
    main()