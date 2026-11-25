import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from config import (
    CLASSIFICATION_THRESHOLD,
    FAKE_LABEL,
    GENUINE_LABEL,
    PROJECT_DIR,
)
from evaluation import evaluate_binary


def combine_text_and_category_split(
    text_features,
    category_predictions,
    category_probabilities,
    encoder,
):
    category_features = encoder.transform(
        np.asarray(category_predictions).reshape(-1, 1)
    )
    probability_features = csr_matrix(category_probabilities)
    interactions = []
    category_values = np.asarray(category_predictions)
    for class_index, category in enumerate(encoder.categories_[0]):
        weights = (
            (category_values == category)
            * category_probabilities[:, class_index]
        )
        interactions.append(
            text_features.multiply(weights[:, np.newaxis])
        )
    category_text_features = hstack(interactions).tocsr()
    return hstack([
        text_features,
        category_features,
        probability_features,
        category_text_features,
    ]).tocsr()


def combine_text_and_category_features(
    X_train_tfidf,
    X_val_tfidf,
    X_test_tfidf,
    train_category_predictions,
    val_category_predictions,
    test_category_predictions,
    train_category_probabilities,
    val_category_probabilities,
    test_category_probabilities,
    category_classes,
):
    encoder = OneHotEncoder(
        categories=[category_classes],
        handle_unknown="ignore",
    )
    encoder.fit(np.asarray(category_classes).reshape(-1, 1))

    X_train_combined = combine_text_and_category_split(
        X_train_tfidf,
        train_category_predictions,
        train_category_probabilities,
        encoder,
    )
    X_val_combined = combine_text_and_category_split(
        X_val_tfidf,
        val_category_predictions,
        val_category_probabilities,
        encoder,
    )
    X_test_combined = combine_text_and_category_split(
        X_test_tfidf,
        test_category_predictions,
        test_category_probabilities,
        encoder,
    )
    return (
        encoder,
        X_train_combined,
        X_val_combined,
        X_test_combined,
    )


def predict_category_aware(model, features):
    fake_index = list(model.classes_).index(FAKE_LABEL)
    probabilities = model.predict_proba(features)[:, fake_index]
    predictions = np.where(
        probabilities >= CLASSIFICATION_THRESHOLD,
        FAKE_LABEL,
        GENUINE_LABEL,
    )
    return predictions, probabilities


def run_category_aware_model(
    X_train_combined,
    y_train,
    X_val_combined,
    y_val,
    X_test_combined,
    y_test,
):
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_combined, y_train)

    val_predictions, val_probabilities = predict_category_aware(
        model, X_val_combined
    )
    val_metrics = evaluate_binary(
        y_val,
        val_predictions,
        val_probabilities,
        "CATEGORY-AWARE MODEL - VALIDATION RESULTS",
    )

    test_predictions, test_probabilities = predict_category_aware(
        model, X_test_combined
    )
    test_metrics = evaluate_binary(
        y_test,
        test_predictions,
        test_probabilities,
        "CATEGORY-AWARE MODEL - FINAL TEST RESULTS",
    )

    joblib.dump(model, PROJECT_DIR / "english_category_aware_model.pkl")
    return (
        model,
        val_predictions,
        val_probabilities,
        test_predictions,
        test_probabilities,
        test_metrics,
        val_metrics,
    )

