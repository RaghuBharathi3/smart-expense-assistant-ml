"""Predefined backend expense tools (read-only seed + runtime CSV writes)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from src.backend.validators import (
    validate_amount,
    validate_category,
    validate_csv_schema,
    validate_date,
    validate_expense_id,
    validate_month_year,
)
from src.classifier.predict import predict
from src.config import CSV_COLUMNS, RUNTIME_DATA_PATH, SEED_DATA_PATH


def _ensure_runtime_csv(
    seed_path: Path = SEED_DATA_PATH,
    runtime_path: Path = RUNTIME_DATA_PATH,
) -> Path:
    """Copy seed CSV to runtime directory if runtime CSV does not yet exist."""
    if not runtime_path.exists():
        runtime_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(seed_path, runtime_path)
    return runtime_path


def _load_df(runtime_path: Path = RUNTIME_DATA_PATH) -> pd.DataFrame:
    path = _ensure_runtime_csv(runtime_path=runtime_path)
    df = pd.read_csv(path)
    validate_csv_schema(df)
    df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def _filter_df(
    df: pd.DataFrame,
    category: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> pd.DataFrame:
    cat = validate_category(category)
    m, y = validate_month_year(month, year)
    out = df
    if cat is not None:
        out = out[out["category"] == cat]
    if m is not None:
        out = out[out["date_dt"].dt.month == m]
    if y is not None:
        out = out[out["date_dt"].dt.year == y]
    return out


def get_expenses(
    category: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> list[dict[str, Any]]:
    """Retrieve expense records optionally filtered by category, month, and year."""
    filtered = _filter_df(_load_df(), category=category, month=month, year=year)
    filtered = filtered.sort_values("date", ascending=False)
    return filtered[list(CSV_COLUMNS)].to_dict(orient="records")


def calculate_spending(
    category: str | None = None,
    month: int | None = None,
    year: int | None = None,
) -> float:
    """Calculate total spending in INR for optional category, month, and year."""
    filtered = _filter_df(_load_df(), category=category, month=month, year=year)
    return round(float(filtered["amount"].sum()), 2)


def compare_months(
    month_a: int,
    month_b: int,
    year: int | None = 2025,
) -> dict[str, Any]:
    """Compare total and per-category spending between two months."""
    validate_month_year(month_a, year)
    validate_month_year(month_b, year)
    df = _load_df()
    df_a = _filter_df(df, month=month_a, year=year)
    df_b = _filter_df(df, month=month_b, year=year)

    total_a = round(float(df_a["amount"].sum()), 2)
    total_b = round(float(df_b["amount"].sum()), 2)
    diff = round(total_b - total_a, 2)
    pct = round((diff / total_a) * 100, 2) if total_a > 0 else 0.0

    return {
        "month_a": month_a,
        "month_b": month_b,
        "year": year,
        "total_a": total_a,
        "total_b": total_b,
        "difference": diff,
        "percent_change": pct,
        "by_category_a": {str(k): round(float(v), 2) for k, v in df_a.groupby("category")["amount"].sum().items()},
        "by_category_b": {str(k): round(float(v), 2) for k, v in df_b.groupby("category")["amount"].sum().items()},
    }


def get_highest_category(
    month: int | None = None,
    year: int | None = None,
) -> dict[str, Any]:
    """Return the category with the highest total spending."""
    filtered = _filter_df(_load_df(), month=month, year=year)
    if filtered.empty:
        return {"category": None, "amount": 0.0}
    grouped = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)
    return {"category": str(grouped.index[0]), "amount": round(float(grouped.iloc[0]), 2)}


def get_category_breakdown(
    month: int | None = None,
    year: int | None = None,
) -> dict[str, float]:
    """Return total spending per category."""
    filtered = _filter_df(_load_df(), month=month, year=year)
    grouped = filtered.groupby("category")["amount"].sum().sort_values(ascending=False)
    return {str(k): round(float(v), 2) for k, v in grouped.items()}


def add_expense(
    description: str,
    amount: float,
    expense_date: str,
    category: str | None = None,
    expense_id: str | None = None,
) -> dict[str, Any]:
    """Validate and append a new expense record to the runtime CSV."""
    if not isinstance(description, str) or not description.strip():
        raise ValueError("description must be a non-empty string")
    df = _load_df()
    clean_desc = description.strip()
    valid_amount = validate_amount(amount)
    valid_date = validate_date(expense_date)
    valid_cat = validate_category(category) if category else predict(clean_desc)["category"]

    if expense_id is None:
        next_num = len(df) + 1
        expense_id = f"EXP-{next_num:04d}"
        while expense_id in set(df["expense_id"].astype(str)):
            next_num += 1
            expense_id = f"EXP-{next_num:04d}"
    else:
        expense_id = validate_expense_id(expense_id, df)

    new_row = {
        "expense_id": expense_id,
        "description": clean_desc,
        "amount": valid_amount,
        "date": valid_date,
        "category": valid_cat,
    }
    updated = pd.concat([df[list(CSV_COLUMNS)], pd.DataFrame([new_row])], ignore_index=True)
    updated.to_csv(_ensure_runtime_csv(), index=False)
    return new_row
