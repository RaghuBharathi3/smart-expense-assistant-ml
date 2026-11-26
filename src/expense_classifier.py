"""Prediction module for Smart Expense Assistant ML.

Provides a clean, reusable interface for loading the trained expense
classification artifacts and categorizing natural-language expense
descriptions into Food, Travel, Shopping, or Utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np

from src.preprocessing import clean_text

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = ROOT_DIR / "models" / "expense_model.pkl"
DEFAULT_VECTORIZER_PATH = ROOT_DIR / "models" / "vectorizer.pkl"

_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}


def load_model(
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
    force_reload: bool = False,
) -> tuple[Any, Any]:
    """Load the trained classifier and TF-IDF vectorizer from disk.

    Results are cached in memory per artifact path pair unless
    ``force_reload=True`` is specified.
    """
    m_path = Path(model_path).resolve()
    v_path = Path(vectorizer_path).resolve()
    cache_key = (str(m_path), str(v_path))

    if not force_reload and cache_key in _CACHE:
        return _CACHE[cache_key]

    if not m_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {m_path}")
    if not v_path.exists():
        raise FileNotFoundError(f"Vectorizer artifact not found at: {v_path}")

    model = joblib.load(m_path)
    vectorizer = joblib.load(v_path)
    _CACHE[cache_key] = (model, vectorizer)
    return model, vectorizer


def _validate_description(description: Any) -> str:
    """Validate and normalize a single expense description."""
    if not isinstance(description, str):
        raise TypeError(
            f"Expense description must be a string, got {type(description).__name__}."
        )
    stripped = description.strip()
    if not stripped:
        raise ValueError("Expense description cannot be empty or whitespace-only.")
    cleaned = clean_text(stripped)
    if not cleaned:
        raise ValueError(
            "Expense description must contain at least one alphanumeric character."
        )
    return cleaned


def _compute_confidence(model: Any, features: Any) -> np.ndarray:
    """Compute per-sample confidence scores from actual model outputs.

    Uses ``predict_proba`` when available (e.g., Logistic Regression,
    Multinomial Naive Bayes). For margin-based models such as LinearSVC that
    expose ``decision_function`` instead, applies softmax over the decision
    margins. Note that margin-based softmax scores reflect relative margin
    strength rather than calibrated probabilities.
    """
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(features)
        return np.max(probs, axis=1)

    if hasattr(model, "decision_function"):
        scores = np.asarray(model.decision_function(features), dtype=float)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        shifted = scores - np.max(scores, axis=1, keepdims=True)
        exp_scores = np.exp(shifted)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return np.max(probs, axis=1)

    return np.ones(features.shape[0], dtype=float)


def predict_with_confidence(
    description: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
) -> dict[str, Any]:
    """Predict the expense category and confidence score for one description."""
    cleaned = _validate_description(description)
    model, vectorizer = load_model(model_path=model_path, vectorizer_path=vectorizer_path)
    features = vectorizer.transform([cleaned])
    predicted_category = str(model.predict(features)[0])
    confidence = float(_compute_confidence(model, features)[0])

    return {
        "description": description.strip(),
        "category": predicted_category,
        "confidence": round(confidence, 4),
    }


def categorize_expense(
    description: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
) -> dict[str, Any]:
    """Categorize a single natural-language expense description.

    Primary integration entrypoint for downstream modules (Tasks 2, 3, and 4).
    """
    return predict_with_confidence(
        description=description,
        model_path=model_path,
        vectorizer_path=vectorizer_path,
    )


def batch_categorize_expenses(
    descriptions: Iterable[str],
    model_path: str | Path = DEFAULT_MODEL_PATH,
    vectorizer_path: str | Path = DEFAULT_VECTORIZER_PATH,
) -> list[dict[str, Any]]:
    """Categorize a batch of expense descriptions in a single vectorized pass."""
    if isinstance(descriptions, (str, bytes)) or not isinstance(descriptions, Iterable):
        raise TypeError("descriptions must be an iterable of strings (e.g., list or tuple).")

    raw_list = list(descriptions)
    if not raw_list:
        return []

    cleaned_list = [_validate_description(item) for item in raw_list]
    model, vectorizer = load_model(model_path=model_path, vectorizer_path=vectorizer_path)
    features = vectorizer.transform(cleaned_list)
    predictions = model.predict(features)
    confidences = _compute_confidence(model, features)

    return [
        {
            "description": orig.strip(),
            "category": str(pred),
            "confidence": round(float(conf), 4),
        }
        for orig, pred, conf in zip(raw_list, predictions, confidences)
    ]


if __name__ == "__main__":
    samples = [
        "Uber ride to office",
        "Ordered pizza online",
        "Purchased headphones",
        "Monthly electricity bill",
        "Swiggy food delivery",
        "Chennai Metro ticket",
    ]
    for item in batch_categorize_expenses(samples):
        print(item)
