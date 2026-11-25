import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config import FAKE_LABEL, GENUINE_LABEL


def evaluate_binary(y_true, y_pred, probabilities, title):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, pos_label=FAKE_LABEL),
        "Recall": recall_score(y_true, y_pred, pos_label=FAKE_LABEL),
        "F1 Score": f1_score(y_true, y_pred, pos_label=FAKE_LABEL),
        "ROC-AUC": roc_auc_score(
            (y_true == FAKE_LABEL).astype(int),
            probabilities,
        ),
    }

    print(title)
    print("-" * len(title))
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    print()
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[GENUINE_LABEL, FAKE_LABEL],
            target_names=["Genuine", "Fake"],
            zero_division=0,
        )
    )
    return metrics


def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=[GENUINE_LABEL, FAKE_LABEL],
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fake"],
        yticklabels=["Genuine", "Fake"],
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

