from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.nn import functional as F
from tqdm import tqdm

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
from utils.dataset import create_dataloaders
from models.retfound_classifier import RETFoundClassifier


@torch.inference_mode()
def collect_predictions(
    model,
    loader,
    device,
):
    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, targets in tqdm(loader, desc="Testing"):
        images = images.to(device, non_blocking=True)

        logits = model(images)
        probabilities = F.softmax(logits, dim=1)
        predictions = torch.argmax(probabilities, dim=1)

        all_targets.extend(targets.numpy().tolist())
        all_predictions.extend(predictions.cpu().numpy().tolist())
        all_probabilities.extend(probabilities.cpu().numpy().tolist())

    return (
        np.array(all_targets),
        np.array(all_predictions),
        np.array(all_probabilities),
    )


def save_confusion_matrix(
    matrix: np.ndarray,
) -> None:
    plt.figure(figsize=(9, 7))
    plt.imshow(matrix)
    plt.title("RETFound Test Confusion Matrix")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.xticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
        rotation=45,
        ha="right",
    )
    plt.yticks(
        range(NUM_CLASSES),
        CLASS_NAMES,
    )
    plt.colorbar()

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(
                column,
                row,
                str(matrix[row, column]),
                ha="center",
                va="center",
            )

    plt.tight_layout()
    output_path = CHECKPOINT_DIR / "partial_low_lr_test_confusion_matrix.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print(f"Confusion matrix saved to: {output_path}")


def main() -> None:
    create_directories()
    validate_paths()

    checkpoint_path = (
        CHECKPOINT_DIR / "retfound_partial_low_lr_best.pth"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Best checkpoint not found: {checkpoint_path}"
        )

    _, _, test_loader = create_dataloaders()

    model = RETFoundClassifier(
        repository_path=RETF_FOUND_REPO,
        checkpoint_path=PRETRAINED_CHECKPOINT,
        num_classes=NUM_CLASSES,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    del checkpoint

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    model.to(DEVICE)

    y_true, y_pred, y_prob = collect_predictions(
        model=model,
        loader=test_loader,
        device=DEVICE,
    )

    accuracy = accuracy_score(y_true, y_pred)
    macro_precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0,
    )
    quadratic_kappa = cohen_kappa_score(
        y_true,
        y_pred,
        weights="quadratic",
    )
    mcc = matthews_corrcoef(
        y_true,
        y_pred,
    )

    try:
        macro_auc = roc_auc_score(
            y_true,
            y_prob,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = None

    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    results = {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "quadratic_weighted_kappa": quadratic_kappa,
        "matthews_correlation_coefficient": mcc,
        "macro_auc_ovr": macro_auc,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    results_path = (
        CHECKPOINT_DIR / "partial_low_lr_metrics.json"
    )

    with results_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    print("\n" + "=" * 65)
    print("RETFOUND TEST RESULTS")
    print("=" * 65)
    print(f"Accuracy:                 {accuracy:.4f}")
    print(f"Macro precision:          {macro_precision:.4f}")
    print(f"Macro recall:             {macro_recall:.4f}")
    print(f"Macro F1:                 {macro_f1:.4f}")
    print(f"Weighted F1:              {weighted_f1:.4f}")
    print(f"Quadratic weighted kappa: {quadratic_kappa:.4f}")
    print(f"MCC:                      {mcc:.4f}")

    if macro_auc is not None:
        print(f"Macro AUROC:              {macro_auc:.4f}")

    print("\nPer-class report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    save_confusion_matrix(matrix)

    print(f"Metrics saved to: {results_path}")


if __name__ == "__main__":
    main()