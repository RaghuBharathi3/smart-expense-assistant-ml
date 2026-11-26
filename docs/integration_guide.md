# Integration Guide (Tasks 2, 3, and 4)

This guide explains how teammates working on **Task 2 (Expense Retrieval & Calculation)**, **Task 3 (LLM Integration & Tool Calling)**, and **Task 4 (Chatbot Interface)** can use the trained classifier.

## Quick Start

```python
from src.expense_classifier import categorize_expense, batch_categorize_expenses

# Single expense
result = categorize_expense("Uber ride to office")
# {
#     "description": "Uber ride to office",
#     "category": "Travel",
#     "confidence": 0.7258
# }

# Batch of expenses
results = batch_categorize_expenses([
    "Swiggy food delivery",
    "Monthly electricity bill",
])
```

## Available Functions (`src/expense_classifier.py`)

- `load_model(model_path=..., vectorizer_path=..., force_reload=False)` -> `(model, vectorizer)`
- `categorize_expense(description: str)` -> `dict` with `description`, `category`, `confidence`
- `predict_with_confidence(description: str)` -> `dict` with `description`, `category`, `confidence`
- `batch_categorize_expenses(descriptions: Iterable[str])` -> `list[dict]`

## Error Handling

- Raises `TypeError` if `description` is not a `str` (or if `descriptions` is a `str` instead of a list/iterable).
- Raises `ValueError` if `description` is empty, whitespace-only, or has no alphanumeric characters.
