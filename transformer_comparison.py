import joblib
import pandas as pd

from category_aware_model import combine_text_and_category_split
from config import FAKE_LABEL, PROJECT_DIR


def evaluate_model(y_true, predictions, probabilities):
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    return {
        "Accuracy": accuracy_score(y_true, predictions),
        "Precision": precision_score(y_true, predictions, zero_division=0),
        "Recall": recall_score(y_true, predictions, zero_division=0),
        "F1 Score": f1_score(y_true, predictions, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
    }


def build_model_comparison(en_test, transformer_predictions, transformer_probabilities):
    y_test = en_test["label"].astype(int).values

    tfidf_vectorizer = joblib.load(
        PROJECT_DIR / "english_tfidf_vectorizer.pkl"
    )
    baseline_model = joblib.load(
        PROJECT_DIR / "english_logistic_regression.pkl"
    )
    category_model = joblib.load(
        PROJECT_DIR / "english_category_model.pkl"
    )
    category_encoder = joblib.load(
        PROJECT_DIR / "english_category_encoder.pkl"
    )
    category_aware_model = joblib.load(
        PROJECT_DIR / "english_category_aware_model.pkl"
    )

    X_test_tfidf = tfidf_vectorizer.transform(
        en_test["review_text"].fillna("").astype(str)
    )

    baseline_prob = baseline_model.predict_proba(X_test_tfidf)[
        :, list(baseline_model.classes_).index(FAKE_LABEL)
    ]
    baseline_pred = (baseline_prob >= 0.5).astype(int)

    test_categories = category_model.predict(X_test_tfidf)
    test_category_probabilities = category_model.predict_proba(X_test_tfidf)
    X_test_combined = combine_text_and_category_split(
        X_test_tfidf,
        test_categories,
        test_category_probabilities,
        category_encoder,
    )

    aware_prob = category_aware_model.predict_proba(X_test_combined)[
        :, list(category_aware_model.classes_).index(FAKE_LABEL)
    ]
    aware_pred = (aware_prob >= 0.5).astype(int)

    comparison = pd.DataFrame({
        "Model 1 - Text Only": evaluate_model(
            y_test, baseline_pred, baseline_prob
        ),
        "Model 3 - Category Aware": evaluate_model(
            y_test, aware_pred, aware_prob
        ),
        "Model 4 - DistilBERT": evaluate_model(
            y_test,
            transformer_predictions,
            transformer_probabilities,
        ),
    })
    print(comparison.round(4).to_string())
    return comparison

