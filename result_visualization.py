import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from config import PROJECT_DIR


RESULTS_DIR = PROJECT_DIR / "saved_results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODEL_TITLES = {
    "baseline": "Text-only baseline",
    "category-aware": "Category-aware",
    "transformer": "DistilBERT",
}
LABELS = ["Genuine", "Fake"]


def generate_result_figures(
    model_names,
    comparison=None,
    include_category_metrics=False,
):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    created_figures = []

    for model_name in model_names:
        predictions_path = (
            RESULTS_DIR / model_name / "test_predictions.csv"
        )
        predictions = pd.read_csv(predictions_path)
        if "true_label" not in predictions.columns:
            if "original_label" not in predictions.columns:
                raise ValueError(
                    f"{predictions_path} must contain 'true_label' or "
                    "'original_label'."
                )
            predictions = predictions.rename(
                columns={"original_label": "true_label"}
            )
        _plot_confusion_matrix(model_name, predictions)
        created_figures.append(
            FIGURES_DIR / f"{model_name}_confusion_matrix.png"
        )

        if "fake_probability" in predictions.columns:
            _plot_roc_curve(model_name, predictions)
            created_figures.append(
                FIGURES_DIR / f"{model_name}_roc_curve.png"
            )

    if comparison is not None:
        _plot_model_comparison(comparison)
        created_figures.append(FIGURES_DIR / "model_comparison.png")

    if include_category_metrics:
        _plot_category_metrics()
        created_figures.append(FIGURES_DIR / "category_metrics.png")

    return created_figures


def _plot_confusion_matrix(model_name, predictions):
    matrix = confusion_matrix(
        predictions["true_label"],
        predictions["predicted_label"],
        labels=LABELS,
    )
    figure, axis = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=LABELS,
        yticklabels=LABELS,
        ax=axis,
    )
    axis.set(
        title=f"{MODEL_TITLES[model_name]} - Test Confusion Matrix",
        xlabel="Predicted label",
        ylabel="True label",
    )
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / f"{model_name}_confusion_matrix.png",
        dpi=200,
    )
    plt.close(figure)


def _plot_roc_curve(model_name, predictions):
    true_labels = (predictions["true_label"] == "Fake").astype(int)
    probabilities = pd.to_numeric(
        predictions["fake_probability"],
        errors="raise",
    )
    false_positive_rate, true_positive_rate, _ = roc_curve(
        true_labels,
        probabilities,
    )
    auc = roc_auc_score(true_labels, probabilities)

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label=f"AUC = {auc:.3f}",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
    axis.set(
        title=f"{MODEL_TITLES[model_name]} - Test ROC Curve",
        xlabel="False positive rate",
        ylabel="True positive rate",
        xlim=(0, 1),
        ylim=(0, 1.02),
    )
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(
        FIGURES_DIR / f"{model_name}_roc_curve.png",
        dpi=200,
    )
    plt.close(figure)


def _plot_model_comparison(comparison):
    metrics = comparison.astype(float)
    figure, axis = plt.subplots(figsize=(10, 6))
    metrics.plot(kind="bar", ax=axis)
    axis.set(
        title="Test Performance by Model",
        xlabel="Metric",
        ylabel="Score",
        ylim=(0, 1.05),
    )
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left")
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "model_comparison.png", dpi=200)
    plt.close(figure)


def _plot_category_metrics():
    metrics_path = RESULTS_DIR / "category" / "metrics.json"
    with metrics_path.open("r", encoding="utf-8") as metrics_file:
        metrics = json.load(metrics_file)

    values = {
        "Validation accuracy": metrics["validation"]["Accuracy"],
        "Validation macro F1": metrics["validation"]["Macro F1"],
        "Train OOF accuracy": metrics["train_oof"]["Accuracy"],
    }
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(values.keys(), values.values(), color="slateblue")
    axis.set(
        title="Category Classifier Metrics",
        ylabel="Score",
        ylim=(0, 1.05),
    )
    axis.tick_params(axis="x", rotation=12)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "category_metrics.png", dpi=200)
    plt.close(figure)
