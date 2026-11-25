from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from config import COLAB_DATA_DIR, DATA_DIR


def resolve_data_dir() -> Path:
    if (DATA_DIR / "en_train.csv").exists():
        return DATA_DIR
    if (COLAB_DATA_DIR / "en_train.csv").exists():
        return COLAB_DATA_DIR
    raise FileNotFoundError(
        "Could not find en_train.csv, en_val.csv, or en_test.csv. "
        f"Place them under {DATA_DIR}."
    )


def load_datasets(data_dir: Path | None = None):
    data_dir = data_dir or resolve_data_dir()
    en_train = pd.read_csv(data_dir / "en_train.csv")
    en_val = pd.read_csv(data_dir / "en_val.csv")
    en_test = pd.read_csv(data_dir / "en_test.csv")
    return en_train, en_val, en_test


def create_tfidf_features(en_train, en_val, en_test):
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        max_features=100000,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )

    X_train = en_train["review_text"]
    X_val = en_val["review_text"]
    X_test = en_test["review_text"]

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)

    return vectorizer, X_train_tfidf, X_val_tfidf, X_test_tfidf

