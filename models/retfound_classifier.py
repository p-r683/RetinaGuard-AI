from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn


def add_retfound_repository_to_path(
    repository_path: Path,
) -> None:
    """Allow importing the official RETFound model files."""

    repository_string = str(repository_path)

    if repository_string not in sys.path:
        sys.path.insert(0, repository_string)


class RETFoundClassifier(nn.Module):
    """RETFound CFP encoder adapted for five-class classification."""

    def __init__(
        self,
        repository_path: Path,
        checkpoint_path: Path,
        num_classes: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        add_retfound_repository_to_path(repository_path)

        import models_vit
        from util.pos_embed import interpolate_pos_embed

        self.model = models_vit.vit_large_patch16(
            num_classes=num_classes,
            drop_path_rate=0.0,
            global_pool=True,
        )

        self._load_pretrained_checkpoint(
            checkpoint_path=checkpoint_path,
            interpolate_pos_embed=interpolate_pos_embed,
        )

        feature_dimension = self.model.head.in_features

        self.model.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(
                feature_dimension,
                num_classes,
            ),
        )

        self._initialize_classification_head()

    def _load_pretrained_checkpoint(
        self,
        checkpoint_path: Path,
        interpolate_pos_embed,
    ) -> None:
        """Load encoder weights and discard MAE-only parameters."""

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )

        checkpoint_model = checkpoint.get(
            "model",
            checkpoint,
        )

        # Copy so the original checkpoint dictionary is not changed.
        checkpoint_model = checkpoint_model.copy()

        model_state = self.model.state_dict()

        for key in ("head.weight", "head.bias"):
            if (
                key in checkpoint_model
                and key in model_state
                and checkpoint_model[key].shape
                != model_state[key].shape
            ):
                del checkpoint_model[key]

        # MAE decoder parameters are unnecessary for classification.
        mae_only_prefixes = (
            "mask_token",
            "decoder_",
        )

        for key in list(checkpoint_model.keys()):
            if key.startswith(mae_only_prefixes):
                del checkpoint_model[key]

        interpolate_pos_embed(
            self.model,
            checkpoint_model,
        )

        load_result = self.model.load_state_dict(
            checkpoint_model,
            strict=False,
        )

        allowed_missing = {
            "fc_norm.weight",
            "fc_norm.bias",
            "head.weight",
            "head.bias",
        }

        unexpected_missing = (
            set(load_result.missing_keys) - allowed_missing
        )

        if unexpected_missing:
            raise RuntimeError(
                "Unexpected missing checkpoint parameters: "
                f"{sorted(unexpected_missing)}"
            )

        if load_result.unexpected_keys:
            print(
                "Ignored unexpected checkpoint parameters:",
                load_result.unexpected_keys,
            )

        print("RETFound encoder weights loaded successfully.")

    def _initialize_classification_head(self) -> None:
        """Initialize the custom classifier."""

        linear_layer = self.model.head[1]

        nn.init.trunc_normal_(
            linear_layer.weight,
            std=2e-5,
        )

        nn.init.zeros_(linear_layer.bias)

    def freeze_encoder(self) -> None:
        """Freeze the transformer encoder."""

        for parameter in self.model.parameters():
            parameter.requires_grad = False

        for parameter in self.model.head.parameters():
            parameter.requires_grad = True

        if hasattr(self.model, "fc_norm"):
            for parameter in self.model.fc_norm.parameters():
                parameter.requires_grad = True

    def unfreeze_encoder(
        self,
        last_blocks: int | None = None,
    ) -> None:
        """
        Unfreeze all encoder layers or only the final transformer blocks.

        Partial unfreezing is recommended on low-memory hardware.
        """

        if last_blocks is None:
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            return

        if last_blocks <= 0:
            raise ValueError(
                "last_blocks must be a positive integer."
            )

        self.freeze_encoder()

        transformer_blocks = self.model.blocks
        number_of_blocks = len(transformer_blocks)

        start_index = max(
            0,
            number_of_blocks - last_blocks,
        )

        for block in transformer_blocks[start_index:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        if hasattr(self.model, "norm"):
            for parameter in self.model.norm.parameters():
                parameter.requires_grad = True

        if hasattr(self.model, "fc_norm"):
            for parameter in self.model.fc_norm.parameters():
                parameter.requires_grad = True

    def trainable_parameter_count(self) -> int:
        """Return the number of parameters currently trainable."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def total_parameter_count(self) -> int:
        """Return the total number of model parameters."""

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )

    def extract_features(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the 1024-dimensional RETFound retinal representation.

        We call forward_features directly because RETFound's custom
        implementation already performs global pooling and fc_norm.
        """
        features = self.model.forward_features(images)

        if features.ndim != 2:
            raise RuntimeError(
                "Unexpected RETFound feature shape. "
                f"Expected [batch_size, 1024], got {tuple(features.shape)}"
            )

        return features


    def forward(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract RETFound features and pass them through our custom head.
        """
        features = self.extract_features(images)
        logits = self.model.head(features)

        return logits