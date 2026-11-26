"""Train, evaluate, compare, and export expense classification models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.preprocessing import RANDOM_STATE, load_and_clean_dataset, split_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_DIR / "data" / "expenses.csv"
MODELS_DIR = ROOT_DIR / "models"
FIGURES_DIR = ROOT_DIR / "docs" / "figures"


def get_candidate_models(random_state: int = RANDOM_STATE) -> dict[str, Any]:
    """Return the candidate classifiers required for comparison."""
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=random_state),
        "Linear SVM": LinearSVC(random_state=random_state),
    }


def evaluate_model(
    model: Any,
    X_test_vec: Any,
    y_test: Any,
    labels: list[str],
) -> dict[str, Any]:
    """Compute evaluation metrics, confusion matrix, and classification report."""
    y_pred = model.predict(X_test_vec)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "precision_weighted": round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "recall_weighted": round(float(recall_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(y_test, y_pred, average="macro", zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_test, y_pred, labels=labels, digits=4),
    }


def _save_confusion_matrix_plot(
    cm: list[list[int]],
    labels: list[str],
    model_name: str,
    output_path: Path,
) -> None:
    """Save a heatmap visualization of the confusion matrix."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_title(f"Confusion Matrix - {model_name}")
    ax.set_xlabel("Predicted Category")
    ax.set_ylabel("True Category")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def train_and_select_model(
    data_path: Path = DATA_PATH,
    models_dir: Path = MODELS_DIR,
    figures_dir: Path = FIGURES_DIR,
) -> dict[str, Any]:
    """Run full training pipeline, compare models, and export the best artifacts."""
    df = load_and_clean_dataset(data_path)
    X_train, X_test, y_train, y_test = split_dataset(df)

    # Fit TF-IDF strictly on training data to prevent data leakage
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    labels = sorted(y_train.unique().tolist())
    candidates = get_candidate_models()

    results: dict[str, dict[str, Any]] = {}
    trained_models: dict[str, Any] = {}

    for name, model in candidates.items():
        model.fit(X_train_vec, y_train)
        trained_models[name] = model
        results[name] = evaluate_model(model, X_test_vec, y_test, labels=labels)

    # Select best model based on test weighted F1, accuracy, and probability calibration support
    best_name = max(
        results.keys(),
        key=lambda m: (results[m]["f1_weighted"], results[m]["accuracy"], m == "Logistic Regression"),
    )
    best_model = trained_models[best_name]
    best_metrics = results[best_name]

    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, models_dir / "expense_model.pkl")
    joblib.dump(vectorizer, models_dir / "vectorizer.pkl")

    _save_confusion_matrix_plot(
        cm=best_metrics["confusion_matrix"],
        labels=labels,
        model_name=best_name,
        output_path=figures_dir / "confusion_matrix.png",
    )

    summary = {
        "dataset_records": int(len(df)),
        "train_records": int(len(X_train)),
        "test_records": int(len(X_test)),
        "classes": labels,
        "selected_model": best_name,
        "models": {
            name: {k: v for k, v in metrics.items() if k != "classification_report"}
            for name, metrics in results.items()
        },
    }

    with open(models_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {"summary": summary, "reports": {k: v["classification_report"] for k, v in results.items()}}


if __name__ == "__main__":
    output = train_and_select_model()
    print("=== MODEL COMPARISON SUMMARY ===")
    print(json.dumps(output["summary"], indent=2))
    best = output["summary"]["selected_model"]
    print(f"\n=== CLASSIFICATION REPORT ({best}) ===")
    print(output["reports"][best])
