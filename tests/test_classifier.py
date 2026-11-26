"""Automated pytest suite for Smart Expense Assistant ML classifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.expense_classifier import (
    DEFAULT_MODEL_PATH,
    DEFAULT_VECTORIZER_PATH,
    batch_categorize_expenses,
    categorize_expense,
    load_model,
    predict_with_confidence,
)


def test_model_artifacts_exist() -> None:
    """Verify exported model and vectorizer files exist on disk."""
    assert DEFAULT_MODEL_PATH.exists(), "expense_model.pkl must exist in models/"
    assert DEFAULT_VECTORIZER_PATH.exists(), "vectorizer.pkl must exist in models/"


def test_load_model_success_and_reload() -> None:
    """Verify model and vectorizer load independently without retraining."""
    model, vectorizer = load_model(force_reload=True)
    assert hasattr(model, "predict")
    assert hasattr(vectorizer, "transform")

    # Cached call returns the same objects
    cached_model, cached_vec = load_model(force_reload=False)
    assert cached_model is model
    assert cached_vec is vectorizer


def test_load_model_missing_file_raises(tmp_path: Path) -> None:
    """Verify FileNotFoundError when artifact path does not exist."""
    with pytest.raises(FileNotFoundError):
        load_model(
            model_path=tmp_path / "missing_model.pkl",
            vectorizer_path=DEFAULT_VECTORIZER_PATH,
            force_reload=True,
        )


@pytest.mark.parametrize(
    ("description", "expected_category"),
    [
        ("Uber ride to office", "Travel"),
        ("Ordered pizza online", "Food"),
        ("Purchased headphones", "Shopping"),
        ("Monthly electricity bill", "Utilities"),
        ("Swiggy food delivery", "Food"),
        ("Chennai Metro ticket", "Travel"),
        ("Bought running shoes on Myntra", "Shopping"),
        ("Jio prepaid mobile recharge", "Utilities"),
    ],
)
def test_categorize_valid_inputs_all_categories(
    description: str, expected_category: str
) -> None:
    """Verify accurate classification across all four expense categories."""
    result = categorize_expense(description)
    assert result["description"] == description
    assert result["category"] == expected_category
    assert isinstance(result["confidence"], float)
    assert 0.0 < result["confidence"] <= 1.0


def test_predict_with_confidence_schema() -> None:
    """Verify predict_with_confidence returns the expected dictionary schema."""
    result = predict_with_confidence("  Uber ride to office  ")
    assert set(result.keys()) == {"description", "category", "confidence"}
    assert result["description"] == "Uber ride to office"
    assert result["category"] == "Travel"
    assert isinstance(result["confidence"], float)
    assert 0.0 <= result["confidence"] <= 1.0


@pytest.mark.parametrize("empty_input", ["", "   ", "\t\n", "!!!"])
def test_empty_or_punctuation_inputs_raise_value_error(empty_input: str) -> None:
    """Verify empty, whitespace-only, or punctuation-only inputs raise ValueError."""
    with pytest.raises(ValueError):
        categorize_expense(empty_input)


@pytest.mark.parametrize("invalid_input", [None, 123, 45.67, ["Uber"], {"text": "food"}])
def test_invalid_input_types_raise_type_error(invalid_input: object) -> None:
    """Verify non-string inputs raise TypeError."""
    with pytest.raises(TypeError):
        categorize_expense(invalid_input)  # type: ignore[arg-type]


def test_batch_categorize_expenses() -> None:
    """Verify batch prediction returns aligned results for multiple expenses."""
    inputs = [
        "Swiggy food delivery",
        "Chennai Metro ticket",
        "Purchased headphones",
        "Monthly electricity bill",
    ]
    results = batch_categorize_expenses(inputs)
    assert len(results) == 4
    assert [r["category"] for r in results] == ["Food", "Travel", "Shopping", "Utilities"]
    for item in results:
        assert set(item.keys()) == {"description", "category", "confidence"}
        assert 0.0 < item["confidence"] <= 1.0


def test_batch_categorize_empty_list() -> None:
    """Verify empty batch returns an empty list."""
    assert batch_categorize_expenses([]) == []


def test_batch_categorize_invalid_container_raises() -> None:
    """Verify passing a single string or non-iterable to batch API raises TypeError."""
    with pytest.raises(TypeError):
        batch_categorize_expenses("Uber ride to office")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        batch_categorize_expenses(42)  # type: ignore[arg-type]
