from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


from config.config import CHECKPOINT_DIR


EXPERIMENTS = {
    "RETFound Head-only": {
        "metrics": CHECKPOINT_DIR / "head_test_metrics.json",
        "confusion_matrix": (
            CHECKPOINT_DIR / "head_test_confusion_matrix.png"
        ),
        "description": (
            "Frozen RETFound encoder with trainable normalization "
            "and five-class classification head."
        ),
    },
    "Partial FT — Higher LR": {
        "metrics": CHECKPOINT_DIR / "partial_test_metrics.json",
        "confusion_matrix": (
            CHECKPOINT_DIR / "partial_test_confusion_matrix.png"
        ),
        "description": (
            "Final transformer block unfrozen with encoder LR 1e-5 "
            "and head LR 1e-4."
        ),
    },
    "Partial FT — Lower LR": {
        "metrics": CHECKPOINT_DIR / "partial_low_lr_metrics.json",
        "confusion_matrix": (
            CHECKPOINT_DIR / "partial_low_lr_confusion_matrix.png"
        ),
        "description": (
            "Final transformer block unfrozen with encoder LR 2e-6 "
            "and head LR 2e-5."
        ),
    },
}


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "macro_precision": "Macro Precision",
    "macro_recall": "Macro Recall",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
    "quadratic_weighted_kappa": "Quadratic Kappa",
    "matthews_correlation_coefficient": "MCC",
    "macro_auc_ovr": "Macro AUROC",
}


def load_experiment_metrics() -> dict[str, dict]:
    """Load available experiment metric files."""

    loaded = {}

    for experiment_name, details in EXPERIMENTS.items():
        metrics_path = details["metrics"]

        if not metrics_path.exists():
            continue

        with metrics_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            metrics = json.load(file)

        loaded[experiment_name] = {
            "metrics": metrics,
            "confusion_matrix": details[
                "confusion_matrix"
            ],
            "description": details["description"],
        }

    return loaded


def build_summary_dataframe(
    experiments: dict[str, dict],
) -> pd.DataFrame:
    """Convert experiment results into one comparison table."""

    rows = []

    for name, experiment in experiments.items():
        metrics = experiment["metrics"]

        row = {
            "Experiment": name,
        }

        for metric_key, metric_label in (
            METRIC_LABELS.items()
        ):
            value = metrics.get(metric_key)

            row[metric_label] = (
                float(value)
                if value is not None
                else None
            )

        rows.append(row)

    return pd.DataFrame(rows)


def get_best_experiment(
    summary: pd.DataFrame,
    metric: str = "Macro F1",
) -> tuple[str, float]:
    """Return the highest-performing experiment for a metric."""

    if summary.empty:
        raise ValueError(
            "No experiment metrics are available."
        )

    valid_rows = summary.dropna(
        subset=[metric]
    )

    if valid_rows.empty:
        raise ValueError(
            f"No values are available for {metric}."
        )

    best_index = valid_rows[metric].idxmax()

    best_name = str(
        valid_rows.loc[
            best_index,
            "Experiment",
        ]
    )

    best_value = float(
        valid_rows.loc[
            best_index,
            metric,
        ]
    )

    return best_name, best_value


def build_per_class_recall_dataframe(
    experiments: dict[str, dict],
) -> pd.DataFrame:
    """Build a table comparing recall for every DR class."""

    class_names = [
        "No DR",
        "Mild DR",
        "Moderate DR",
        "Severe DR",
        "Proliferative DR",
    ]

    rows = []

    for experiment_name, experiment in (
        experiments.items()
    ):
        report = experiment[
            "metrics"
        ].get(
            "classification_report",
            {},
        )

        for class_name in class_names:
            class_metrics = report.get(
                class_name,
                {},
            )

            rows.append(
                {
                    "Experiment": experiment_name,
                    "DR grade": class_name,
                    "Recall": float(
                        class_metrics.get(
                            "recall",
                            0.0,
                        )
                    ),
                    "Precision": float(
                        class_metrics.get(
                            "precision",
                            0.0,
                        )
                    ),
                    "F1": float(
                        class_metrics.get(
                            "f1-score",
                            0.0,
                        )
                    ),
                }
            )

    return pd.DataFrame(rows)