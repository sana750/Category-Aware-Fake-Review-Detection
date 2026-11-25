import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from config import FAKE_LABEL


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()[:, FAKE_LABEL]
    predictions = (probabilities >= 0.5).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
    }


def logits_to_predictions(logits):
    probabilities = torch.softmax(
        torch.tensor(logits),
        dim=1,
    ).numpy()[:, FAKE_LABEL]
    predictions = np.where(probabilities >= 0.5, FAKE_LABEL, 0)
    return predictions, probabilities


def evaluate_transformer_test(y_true, predictions, probabilities):
    metrics = {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(
            y_true,
            predictions,
            pos_label=FAKE_LABEL,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            pos_label=FAKE_LABEL,
            zero_division=0,
        ),
        "F1 Score": f1_score(
            y_true,
            predictions,
            pos_label=FAKE_LABEL,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
    }
    print("DISTILBERT - FINAL TEST RESULTS")
    print("--------------------------------")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    print()
    print(
        classification_report(
            y_true,
            predictions,
            labels=[0, FAKE_LABEL],
            target_names=["Genuine", "Fake"],
            zero_division=0,
        )
    )
    return metrics


def plot_transformer_confusion_matrix(y_true, predictions):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        predictions,
        labels=[0, FAKE_LABEL],
        display_labels=["Genuine", "Fake"],
        cmap="Blues",
        ax=ax,
    )
    plt.title("DistilBERT - Test Confusion Matrix")
    plt.tight_layout()
    plt.show()
