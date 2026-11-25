import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from config import PROJECT_DIR


def train_category_model(X_train_tfidf, y_train_category):
    model = LogisticRegression(max_iter=2000, random_state=42)
    model.fit(X_train_tfidf, y_train_category)
    return model


def evaluate_category_model(model, X_val_tfidf, y_val_category):
    predictions = model.predict(X_val_tfidf)
    metrics = {
        "Accuracy": accuracy_score(y_val_category, predictions),
        "Macro F1": f1_score(
            y_val_category,
            predictions,
            average="macro",
        ),
    }
    print("CATEGORY MODEL - VALIDATION RESULTS")
    print("-----------------------------------")
    print(f"Accuracy: {metrics['Accuracy']:.4f}")
    print(f"Macro F1: {metrics['Macro F1']:.4f}")
    return predictions, metrics


def generate_out_of_fold_probabilities(
    model,
    X_train_tfidf,
    y_train_category,
):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    probabilities = cross_val_predict(
        model,
        X_train_tfidf,
        y_train_category,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )
    return probabilities


def run_category_model(
    X_train_tfidf,
    y_train_category,
    X_val_tfidf,
    y_val_category,
    X_test_tfidf,
):
    model = train_category_model(X_train_tfidf, y_train_category)
    val_predictions, val_metrics = evaluate_category_model(
        model, X_val_tfidf, y_val_category
    )
    test_predictions = model.predict(X_test_tfidf)
    train_probabilities = generate_out_of_fold_probabilities(
        model, X_train_tfidf, y_train_category
    )
    train_predictions = model.classes_[train_probabilities.argmax(axis=1)]
    oof_accuracy = accuracy_score(y_train_category, train_predictions)
    print("OOF Category Accuracy:", f"{oof_accuracy:.4f}")
    val_probabilities = model.predict_proba(X_val_tfidf)
    test_probabilities = model.predict_proba(X_test_tfidf)

    joblib.dump(model, PROJECT_DIR / "english_category_model.pkl")
    metrics = {
        "validation": val_metrics,
        "train_oof": {"Accuracy": oof_accuracy},
    }
    return (
        model,
        train_predictions,
        val_predictions,
        test_predictions,
        metrics,
        train_probabilities,
        val_probabilities,
        test_probabilities,
    )

