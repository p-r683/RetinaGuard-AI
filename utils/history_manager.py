from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd


from config.config import HISTORY_DIR

HISTORY_FILE = HISTORY_DIR / "analysis_history.csv"


HISTORY_COLUMNS = [
    "analysis_id",
    "timestamp",
    "filename",
    "image_path",
    "predicted_class",
    "confidence",
    "no_dr_probability",
    "mild_dr_probability",
    "moderate_dr_probability",
    "severe_dr_probability",
    "proliferative_dr_probability",
    "attention_path",
    "gradcam_path",
]


def initialize_history() -> None:
    """Create the history folder and CSV when missing."""

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not HISTORY_FILE.exists():
        dataframe = pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

        dataframe.to_csv(
            HISTORY_FILE,
            index=False,
        )


def save_analysis(result: dict) -> str:
    """Append one completed RetinaGuard analysis."""

    initialize_history()

    probabilities = result["probabilities"]
    image_path = Path(result["image_path"])

    analysis_id = uuid4().hex[:12].upper()

    new_record = {
        "analysis_id": analysis_id,
        "timestamp": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "filename": image_path.name,
        "image_path": str(image_path),
        "predicted_class": result["predicted_class"],
        "confidence": float(result["confidence"]),
        "no_dr_probability": float(
            probabilities.get("No DR", 0.0)
        ),
        "mild_dr_probability": float(
            probabilities.get("Mild DR", 0.0)
        ),
        "moderate_dr_probability": float(
            probabilities.get("Moderate DR", 0.0)
        ),
        "severe_dr_probability": float(
            probabilities.get("Severe DR", 0.0)
        ),
        "proliferative_dr_probability": float(
            probabilities.get("Proliferative DR", 0.0)
        ),
        "attention_path": str(
            result["attention_path"]
        ),
        "gradcam_path": str(
            result["gradcam_path"]
        ),
    }

    history = load_history()

    history = pd.concat(
        [
            history,
            pd.DataFrame([new_record]),
        ],
        ignore_index=True,
    )

    history.to_csv(
        HISTORY_FILE,
        index=False,
    )

    return analysis_id


def load_history() -> pd.DataFrame:
    """Load all previous screening records."""

    initialize_history()

    try:
        dataframe = pd.read_csv(
            HISTORY_FILE
        )
    except pd.errors.EmptyDataError:
        dataframe = pd.DataFrame(
            columns=HISTORY_COLUMNS
        )

    return dataframe


def clear_history() -> None:
    """Remove all saved history records."""

    dataframe = pd.DataFrame(
        columns=HISTORY_COLUMNS
    )

    dataframe.to_csv(
        HISTORY_FILE,
        index=False,
    )