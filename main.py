import argparse
import json

import joblib
import pandas as pd

from baseline_model import run_baseline_model
from category_aware_model import (
    combine_text_and_category_features,
    run_category_aware_model,
)
from category_model import run_category_model
from config import FAKE_LABEL, GENUINE_LABEL, PROJECT_DIR
from data_loader import create_tfidf_features, load_datasets
from evaluation import evaluate_binary, plot_confusion_matrix
from result_cache import (
    create_cache_signature,
    dataset_fingerprint,
    load_cached_result,
    save_cached_result,
    save_dataframe,
    save_result_artifacts,
    write_json,
)
from result_visualization import generate_result_figures
from threshold_tuning import (
    apply_threshold,
    predict_transformer_probabilities,
    select_f1_threshold,
)


TFIDF_SETTINGS = {
    "lowercase": True,
    "strip_accents": "unicode",
    "ngram_range": (1, 2),
    "max_features": 100000,
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,
}
BASELINE_SETTINGS = {
    "model": "LogisticRegression",
    "max_iter": 1000,
    "random_state": 42,
    "threshold": 0.5,
    "tfidf": TFIDF_SETTINGS,
}
CATEGORY_SETTINGS = {
    "model": "LogisticRegression",
    "max_iter": 2000,
    "random_state": 42,
    "cross_validation": {
        "n_splits": 5,
        "shuffle": True,
        "random_state": 42,
    },
    "tfidf": TFIDF_SETTINGS,
}
CATEGORY_AWARE_SETTINGS = {
    "model": "LogisticRegression",
    "max_iter": 1000,
    "random_state": 42,
    "threshold": 0.5,
    "tfidf": TFIDF_SETTINGS,
    "category_model": CATEGORY_SETTINGS,
}


def load_project_data():
    try:
        en_train, en_val, en_test = load_datasets()
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc

    print("Training data:", en_train.shape)
    print("Validation data:", en_val.shape)
    print("Testing data:", en_test.shape)

    y_train = en_train["label"]
    y_val = en_val["label"]
    y_test = en_test["label"]
    y_train_category = en_train["category"]
    y_val_category = en_val["category"]

    return {
        "en_train": en_train,
        "en_val": en_val,
        "en_test": en_test,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "y_train_category": y_train_category,
        "y_val_category": y_val_category,
    }


def ensure_tfidf_features(data):
    if "X_train_tfidf" not in data:
        vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf = (
            create_tfidf_features(
                data["en_train"],
                data["en_val"],
                data["en_test"],
            )
        )
        joblib.dump(vectorizer, PROJECT_DIR / "english_tfidf_vectorizer.pkl")
        data.update({
            "X_train_tfidf": X_train_tfidf,
            "X_val_tfidf": X_val_tfidf,
            "X_test_tfidf": X_test_tfidf,
        })
    return data


def run_cached(
    name,
    data_hash,
    settings,
    source_files,
    compute,
    force_retrain,
):
    signature = create_cache_signature(data_hash, settings, source_files)
    if force_retrain:
        print(f"Force retraining {name}.")
    else:
        cached_result = load_cached_result(name, signature)
        if cached_result is not None:
            return cached_result
        print(f"Training {name} and saving results.")

    result = compute()
    save_cached_result(name, signature, result)
    return result


def prediction_frame(data, predictions, probabilities=None):
    frame = pd.DataFrame({
        "review_text": data["en_test"]["review_text"].fillna("").astype(str),
        "true_label": data["y_test"].map({
            GENUINE_LABEL: "Genuine",
            FAKE_LABEL: "Fake",
        }).values,
        "predicted_label": pd.Series(predictions).map({
            GENUINE_LABEL: "Genuine",
            FAKE_LABEL: "Fake",
        }).values,
    })
    if probabilities is not None:
        frame["fake_probability"] = probabilities
    return frame


def tune_binary_result(
    result,
    y_val,
    y_test,
    val_prediction_index,
    val_probability_index,
    test_prediction_index,
    test_probability_index,
    val_metrics_index,
    test_metrics_index,
    title,
):
    tuned_result = list(result)
    threshold = select_f1_threshold(
        y_val,
        tuned_result[val_probability_index],
    )
    val_predictions = apply_threshold(
        tuned_result[val_probability_index],
        threshold,
    )
    test_predictions = apply_threshold(
        tuned_result[test_probability_index],
        threshold,
    )

    tuned_result[val_prediction_index] = val_predictions
    tuned_result[test_prediction_index] = test_predictions
    tuned_result[val_metrics_index] = evaluate_binary(
        y_val,
        val_predictions,
        tuned_result[val_probability_index],
        f"{title} - VALIDATION RESULTS (F1-TUNED THRESHOLD)",
    )
    tuned_result[test_metrics_index] = evaluate_binary(
        y_test,
        test_predictions,
        tuned_result[test_probability_index],
        f"{title} - FINAL TEST RESULTS (F1-TUNED THRESHOLD)",
    )
    print(f"F1-tuned decision threshold: {threshold:.4f}")
    return tuple(tuned_result), threshold


def load_or_predict_transformer_validation_probabilities(
    transformer_result,
    data,
    max_length,
    batch_size,
):
    output_dir = PROJECT_DIR / "saved_results" / "transformer"
    probabilities_path = output_dir / "validation_probabilities.csv"
    metadata_path = output_dir / "validation_probabilities.json"
    validation_hash = dataset_fingerprint([
        ("validation", data["en_val"]),
    ])

    if (
        transformer_result["trainer"] is None
        and probabilities_path.is_file()
        and metadata_path.is_file()
    ):
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        if metadata.get("data_hash") == validation_hash:
            saved_probabilities = pd.read_csv(probabilities_path)[
                "fake_probability"
            ].to_numpy()
            if len(saved_probabilities) == len(data["en_val"]):
                return saved_probabilities

    probabilities = predict_transformer_probabilities(
        transformer_result["model"],
        transformer_result["tokenizer"],
        data["en_val"]["review_text"].fillna("").astype(str).tolist(),
        max_length=max_length,
        batch_size=batch_size,
    )
    save_dataframe(
        probabilities_path,
        pd.DataFrame({"fake_probability": probabilities}),
    )
    write_json(
        metadata_path,
        {"data_hash": validation_hash},
    )
    return probabilities


def run_baseline(data):
    ensure_tfidf_features(data)
    return run_baseline_model(
        data["X_train_tfidf"],
        data["y_train"],
        data["X_val_tfidf"],
        data["y_val"],
        data["X_test_tfidf"],
        data["y_test"],
    )


def run_category(data):
    ensure_tfidf_features(data)
    return run_category_model(
        data["X_train_tfidf"],
        data["y_train_category"],
        data["X_val_tfidf"],
        data["y_val_category"],
        data["X_test_tfidf"],
    )


def run_category_aware(data, category_result=None):
    ensure_tfidf_features(data)
    if category_result is None:
        category_result = run_category(data)
    train_category_predictions = category_result[1]
    val_category_predictions = category_result[2]
    test_category_predictions = category_result[3]

    encoder, X_train_combined, X_val_combined, X_test_combined = (
        combine_text_and_category_features(
            data["X_train_tfidf"],
            data["X_val_tfidf"],
            data["X_test_tfidf"],
            train_category_predictions,
            val_category_predictions,
            test_category_predictions,
            category_result[5],
            category_result[6],
            category_result[7],
            category_result[0].classes_,
        )
    )
    joblib.dump(encoder, PROJECT_DIR / "english_category_encoder.pkl")

    aware_result = run_category_aware_model(
        X_train_combined,
        data["y_train"],
        X_val_combined,
        data["y_val"],
        X_test_combined,
        data["y_test"],
    )

    return aware_result + (encoder,)


def save_category_aware_outputs(
    data,
    category_result,
    aware_result,
    decision_threshold,
):
    test_category_predictions = category_result[3]
    plot_confusion_matrix(
        data["y_test"],
        aware_result[3],
        "Category-Aware Fake Review Detection",
    )
    results = pd.DataFrame({
        "review_text": data["en_test"]["review_text"].fillna("").astype(str),
        "original_category": data["en_test"]["category"].values,
        "predicted_category": test_category_predictions,
        "original_label": data["y_test"].map({
            GENUINE_LABEL: "Genuine",
            FAKE_LABEL: "Fake",
        }).values,
        "predicted_label": pd.Series(aware_result[3]).map({
            GENUINE_LABEL: "Genuine",
            FAKE_LABEL: "Fake",
        }).values,
        "fake_probability": aware_result[4],
    })
    output_path = PROJECT_DIR / "english_category_aware_results.csv"
    results.to_csv(output_path, index=False)
    print("Results saved to:", output_path)
    save_result_artifacts(
        "category-aware",
        {
            "validation": aware_result[6],
            "test": aware_result[5],
            "decision_threshold": decision_threshold,
            "threshold_objective": "F1",
        },
        results,
    )


def print_comparison(baseline_result, aware_result):
    baseline_metrics = baseline_result[6]
    aware_metrics = aware_result[5]
    comparison = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"],
        "Text Only": [
            baseline_metrics["Accuracy"],
            baseline_metrics["Precision"],
            baseline_metrics["Recall"],
            baseline_metrics["F1 Score"],
            baseline_metrics["ROC-AUC"],
        ],
        "Text + Category": [
            aware_metrics["Accuracy"],
            aware_metrics["Precision"],
            aware_metrics["Recall"],
            aware_metrics["F1 Score"],
            aware_metrics["ROC-AUC"],
        ],
    })
    print(comparison.to_string(index=False))
    improvement = (
        aware_metrics["F1 Score"] - baseline_metrics["F1 Score"]
    ) * 100
    print(f"F1 improvement: {improvement:+.2f} percentage points")


def build_cached_comparison(
    baseline_result,
    aware_result,
    transformer_result,
):
    return pd.DataFrame({
        "Model 1 - Text Only": baseline_result[6],
        "Model 3 - Category Aware": aware_result[5],
        "Model 4 - DistilBERT": transformer_result["test_metrics"],
    })


def main():
    parser = argparse.ArgumentParser(
        description="Run one or all fake-review detection models."
    )
    parser.add_argument(
        "--model",
        choices=["baseline", "category", "category-aware", "transformer", "all"],
        default="all",
        help="Model to run (default: all)",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--reuse",
        dest="force_retrain",
        action="store_false",
        default=False,
        help="Reuse saved results when data and settings match (default)",
    )
    cache_group.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore saved results and retrain the selected model(s)",
    )
    args = parser.parse_args()

    data = load_project_data()
    data_hash = dataset_fingerprint([
        ("train", data["en_train"]),
        ("validation", data["en_val"]),
        ("test", data["en_test"]),
    ])
    baseline_result = None
    aware_result = None
    comparison = None

    if args.model in ("baseline", "all"):
        baseline_result = run_cached(
            "baseline",
            data_hash,
            BASELINE_SETTINGS,
            [
                PROJECT_DIR / "baseline_model.py",
                PROJECT_DIR / "data_loader.py",
                PROJECT_DIR / "evaluation.py",
                PROJECT_DIR / "config.py",
            ],
            lambda: run_baseline(data),
            args.force_retrain,
        )
        baseline_result, baseline_threshold = tune_binary_result(
            baseline_result,
            data["y_val"],
            data["y_test"],
            1,
            2,
            3,
            4,
            5,
            6,
            "BASELINE MODEL",
        )
        joblib.dump(
            baseline_result[0],
            PROJECT_DIR / "english_logistic_regression.pkl",
        )
        save_result_artifacts(
            "baseline",
            {
                "validation": baseline_result[5],
                "test": baseline_result[6],
                "decision_threshold": baseline_threshold,
                "threshold_objective": "F1",
            },
            prediction_frame(data, baseline_result[3], baseline_result[4]),
        )

    category_result = None
    if args.model in ("category", "category-aware", "all"):
        category_result = run_cached(
            "category",
            data_hash,
            CATEGORY_SETTINGS,
            [
                PROJECT_DIR / "category_model.py",
                PROJECT_DIR / "data_loader.py",
                PROJECT_DIR / "config.py",
            ],
            lambda: run_category(data),
            args.force_retrain,
        )
        joblib.dump(
            category_result[0],
            PROJECT_DIR / "english_category_model.pkl",
        )
        category_predictions = pd.DataFrame({
            "review_text": data["en_test"]["review_text"].fillna("").astype(str),
            "true_category": data["en_test"]["category"].values,
            "predicted_category": category_result[3],
        })
        save_result_artifacts(
            "category",
            category_result[4],
            category_predictions,
        )

    if args.model in ("category-aware", "all"):
        aware_result = run_cached(
            "category-aware",
            data_hash,
            CATEGORY_AWARE_SETTINGS,
            [
                PROJECT_DIR / "category_aware_model.py",
                PROJECT_DIR / "category_model.py",
                PROJECT_DIR / "data_loader.py",
                PROJECT_DIR / "evaluation.py",
                PROJECT_DIR / "config.py",
            ],
            lambda: run_category_aware(data, category_result),
            args.force_retrain,
        )
        aware_result, aware_threshold = tune_binary_result(
            aware_result,
            data["y_val"],
            data["y_test"],
            1,
            2,
            3,
            4,
            6,
            5,
            "CATEGORY-AWARE MODEL",
        )
        joblib.dump(
            aware_result[0],
            PROJECT_DIR / "english_category_aware_model.pkl",
        )
        joblib.dump(
            aware_result[7],
            PROJECT_DIR / "english_category_encoder.pkl",
        )
        save_category_aware_outputs(
            data,
            category_result,
            aware_result,
            aware_threshold,
        )

    if args.model in ("transformer", "all"):
        from transformer_model import (
            EVAL_BATCH_SIZE,
            MAX_LENGTH,
            train_transformer_model,
        )

        transformer_result = train_transformer_model(
            force_retrain=args.force_retrain,
        )
        val_probabilities = (
            load_or_predict_transformer_validation_probabilities(
                transformer_result,
                data,
                max_length=MAX_LENGTH,
                batch_size=EVAL_BATCH_SIZE,
            )
        )
        transformer_threshold = select_f1_threshold(
            data["y_val"],
            val_probabilities,
        )
        val_predictions = apply_threshold(
            val_probabilities,
            transformer_threshold,
        )
        test_predictions = apply_threshold(
            transformer_result["test_probabilities"],
            transformer_threshold,
        )
        print(
            "DistilBERT F1-tuned decision threshold: "
            f"{transformer_threshold:.4f}"
        )
        val_metrics = evaluate_binary(
            data["y_val"],
            val_predictions,
            val_probabilities,
            "DISTILBERT - VALIDATION RESULTS (F1-TUNED THRESHOLD)",
        )
        test_metrics = evaluate_binary(
            data["y_test"],
            test_predictions,
            transformer_result["test_probabilities"],
            "DISTILBERT - FINAL TEST RESULTS (F1-TUNED THRESHOLD)",
        )
        transformer_result["test_predictions"] = test_predictions
        transformer_result["test_metrics"] = test_metrics
        save_result_artifacts(
            "transformer",
            {
                "validation": val_metrics,
                "test": test_metrics,
                "decision_threshold": transformer_threshold,
                "threshold_objective": "F1",
            },
            prediction_frame(
                data,
                test_predictions,
                transformer_result["test_probabilities"],
            ),
        )
        if args.model == "all":
            comparison = build_cached_comparison(
                baseline_result,
                aware_result,
                transformer_result,
            )
            print(comparison.round(4).to_string())
            save_dataframe(
                PROJECT_DIR / "saved_results" / "model_comparison.csv",
                comparison.rename_axis("Metric").reset_index(),
            )

    if args.model == "all":
        print_comparison(baseline_result, aware_result)

    model_names = {
        "baseline": ["baseline"],
        "category": [],
        "category-aware": ["category-aware"],
        "transformer": ["transformer"],
        "all": ["baseline", "category-aware", "transformer"],
    }[args.model]
    figures = generate_result_figures(
        model_names,
        comparison=comparison,
        include_category_metrics=args.model in (
            "category",
            "category-aware",
            "all",
        ),
    )
    for figure_path in figures:
        print("Figure saved to:", figure_path)


if __name__ == "__main__":
    main()
