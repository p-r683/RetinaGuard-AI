from __future__ import annotations

import os
from pathlib import Path

import torch


# ---------------------------------------------------------
# Portable project paths
# ---------------------------------------------------------
PROJECT_DIR = Path(
    os.getenv(
        "RETINAGUARD_PROJECT_DIR",
        Path(__file__).resolve().parents[1],
    )
).resolve()

DATASET_DIR = Path(
    os.getenv(
        "RETINAGUARD_DATASET_DIR",
        PROJECT_DIR / "datasets" / "aptos_retfound",
    )
).resolve()

RETF_FOUND_REPO = Path(
    os.getenv(
        "RETFOUND_REPO_DIR",
        PROJECT_DIR / "external" / "RETFound_MAE",
    )
).resolve()
PRETRAINED_CHECKPOINT = Path(
    os.getenv(
        "RETFOUND_PRETRAINED_CHECKPOINT",
        PROJECT_DIR
        / "checkpoints"
        / "RETFound_cfp_weights.pth",
    )
).resolve()
INFERENCE_CHECKPOINT = Path(
    os.getenv(
        "RETINAGUARD_INFERENCE_CHECKPOINT",
        PROJECT_DIR
        / "checkpoints"
        / "retfound_inference.pth",
    )
).resolve()

CHECKPOINT_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = PROJECT_DIR / "logs"
UPLOAD_DIR = PROJECT_DIR / "uploads"
REPORT_DIR = PROJECT_DIR / "reports"
HISTORY_DIR = PROJECT_DIR / "history"
EXPLAINABILITY_DIR = (
    PROJECT_DIR / "explainability_outputs"
)


CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

NUM_CLASSES = len(CLASS_NAMES)
IMAGE_SIZE = 224

BATCH_SIZE = 1
NUM_WORKERS = 0
RANDOM_SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

USE_AMP = DEVICE.type == "cuda"


def create_directories() -> None:
    """Create runtime directories."""

    for directory in (
        CHECKPOINT_DIR,
        OUTPUT_DIR,
        LOG_DIR,
        UPLOAD_DIR,
        REPORT_DIR,
        HISTORY_DIR,
        EXPLAINABILITY_DIR,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

def validate_paths() -> None:
    """Validate files required by the deployed inference app."""

    required = {
        "RETFound runtime source": RETF_FOUND_REPO,
        "Inference checkpoint": INFERENCE_CHECKPOINT,
    }

    missing = [
        f"{name}: {path}"
        for name, path in required.items()
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Required files are missing:\n"
            + "\n".join(missing)
        )
