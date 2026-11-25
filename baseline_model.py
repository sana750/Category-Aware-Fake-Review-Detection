import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression

from config import (
    CLASSIFICATION_THRESHOLD,
    FAKE_LABEL,
    GENUINE_LABEL,
    PROJECT_DIR,
)
from evaluation import evaluate_binary


def train_baseline_model(X_train_tfidf, y_train):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    return model


def predict_fake_probabilities(model, features):
    fake_index = list(model.classes_).index(FAKE_LABEL)
    probabilities = model.predict_proba(features)[:, fake_index]
    predictions = np.where(
        probabilities >= CLASSIFICATION_THRESHOLD,
        FAKE_LABEL,
        GENUINE_LABEL,
    )
    return predictions, probabilities


def run_baseline_model(X_train_tfidf, y_train, X_val_tfidf, y_val,
                       X_test_tfidf, y_test):
    model = train_baseline_model(X_train_tfidf, y_train)

    val_predictions, val_probabilities = predict_fake_probabilities(
        model, X_val_tfidf
    )
    val_metrics = evaluate_binary(
        y_val,
        val_predictions,
        val_probabilities,
        "BASELINE MODEL - VALIDATION RESULTS",
    )

    test_predictions, test_probabilities = predict_fake_probabilities(
        model, X_test_tfidf
    )
    test_metrics = evaluate_binary(
        y_test,
        test_predictions,
        test_probabilities,
        "BASELINE MODEL - FINAL TEST RESULTS",
    )

    joblib.dump(model, PROJECT_DIR / "english_logistic_regression.pkl")
    return model, val_predictions, val_probabilities, test_predictions, test_probabilities, val_metrics, test_metrics

