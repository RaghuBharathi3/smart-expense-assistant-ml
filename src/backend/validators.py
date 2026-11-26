"""Input and CSV schema validators for backend expense tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from src.config import ALLOWED_CATEGORIES, CSV_COLUMNS


def validate_category(category: str | None) -> str | None:
    """Validate optional category against allowed expense categories."""
    if category is None:
        return None
    if not isinstance(category, str) or not category.strip():
        raise ValueError("category must be a non-empty string")
    normalized = category.strip().capitalize()
    if normalized not in ALLOWED_CATEGORIES:
        raise ValueError(f"category must be one of {ALLOWED_CATEGORIES}, got '{category}'")
    return normalized


def validate_month_year(month: int | None, year: int | None) -> tuple[int | None, int | None]:
    """Validate optional month (1-12) and year (2000-2100)."""
    if month is not None:
        if not isinstance(month, int) or isinstance(month, bool) or not (1 <= month <= 12):
            raise ValueError(f"month must be an integer between 1 and 12, got {month!r}")
    if year is not None:
        if not isinstance(year, int) or isinstance(year, bool) or not (2000 <= year <= 2100):
            raise ValueError(f"year must be an integer between 2000 and 2100, got {year!r}")
    return month, year


def validate_amount(amount: Any) -> float:
    """Validate positive numeric expense amount."""
    if isinstance(amount, bool) or not isinstance(amount, (int, float)):
        raise ValueError(f"amount must be a number, got {type(amount).__name__}")
    val = float(amount)
    if val <= 0:
        raise ValueError("amount must be greater than 0")
    return round(val, 2)


def validate_date(date_str: str) -> str:
    """Validate ISO-8601 date string (YYYY-MM-DD)."""
    if not isinstance(date_str, str):
        raise ValueError("date must be a YYYY-MM-DD string")
    cleaned = date_str.strip()
    try:
        datetime.strptime(cleaned, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(f"date must match YYYY-MM-DD, got '{date_str}'") from exc
    return cleaned


def validate_expense_id(expense_id: str, existing_df: pd.DataFrame) -> str:
    """Validate non-empty expense_id and block duplicates."""
    if not isinstance(expense_id, str) or not expense_id.strip():
        raise ValueError("expense_id must be a non-empty string")
    eid = expense_id.strip()
    if not existing_df.empty and eid in set(existing_df["expense_id"].astype(str)):
        raise ValueError(f"Duplicate expense_id '{eid}' is not allowed")
    return eid


def validate_csv_schema(df: pd.DataFrame) -> None:
    """Ensure DataFrame preserves the canonical expense CSV columns."""
    missing = set(CSV_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")
