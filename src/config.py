"""Centralized paths and settings for Smart Expense Assistant."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent

SEED_DATA_PATH = Path(os.getenv("EXPENSE_SEED_DATA_PATH", str(ROOT_DIR / "data" / "expenses.csv")))
RUNTIME_DATA_PATH = Path(os.getenv("EXPENSE_RUNTIME_DATA_PATH", str(ROOT_DIR / "data" / "runtime" / "expenses.csv")))
CLASSIFIER_MODEL_PATH = Path(
    os.getenv("EXPENSE_CLASSIFIER_PATH", str(ROOT_DIR / "models" / "classifier" / "expense_classifier.pkl"))
)

ALLOWED_CATEGORIES = ("Food", "Travel", "Shopping", "Utilities")
CSV_COLUMNS = ("expense_id", "description", "amount", "date", "category")
RANDOM_STATE = 42

LLM_API_URL = os.getenv("LLM_API_URL", "").strip()
LLM_API_KEY = os.getenv("LLM_API_KEY", "").strip()
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "15.0"))
