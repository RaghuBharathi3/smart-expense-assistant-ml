"""Prediction entrypoint for the trained expense classifier pipeline."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib

from src.classifier.preprocessing import clean_text
from src.config import CLASSIFIER_MODEL_PATH


@lru_cache(maxsize=2)
def _load_pipeline(model_path: str) -> Any:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Classifier model not found at {path}")
    return joblib.load(path)


def predict(
    description: str,
    model_path: Path = CLASSIFIER_MODEL_PATH,
) -> dict[str, Any]:
    """Predict expense category and calibrated probability for a description."""
    if not isinstance(description, str):
        raise TypeError(f"description must be str, got {type(description).__name__}")
    cleaned = clean_text(description)
    if not cleaned:
        raise ValueError("description must contain at least one alphanumeric character")

    pipeline = _load_pipeline(str(model_path.resolve()))
    category = str(pipeline.predict([cleaned])[0])
    confidence = float(pipeline.predict_proba([cleaned])[0].max())
    return {
        "description": description.strip(),
        "category": category,
        "confidence": round(confidence, 4),
    }
