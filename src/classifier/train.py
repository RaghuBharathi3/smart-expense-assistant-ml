"""Train and evaluate TF-IDF + Logistic Regression expense classifier."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.classifier.preprocessing import load_and_clean_dataset, split_dataset
from src.config import CLASSIFIER_MODEL_PATH, RANDOM_STATE, SEED_DATA_PATH


def train_classifier(
    data_path: Path = SEED_DATA_PATH,
    model_path: Path = CLASSIFIER_MODEL_PATH,
) -> dict[str, float]:
    """Train TF-IDF + Logistic Regression pipeline, evaluate on test split, and save .pkl."""
    df = load_and_clean_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(df, random_state=RANDOM_STATE)

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    return metrics


if __name__ == "__main__":
    results = train_classifier()
    print(f"Saved classifier to {CLASSIFIER_MODEL_PATH}")
    print("Evaluation metrics:", results)
