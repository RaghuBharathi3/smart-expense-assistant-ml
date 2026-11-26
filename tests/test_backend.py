"""Unit tests for backend tools and input/schema validators."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.backend import tools, validators


@pytest.fixture()
def isolated_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect runtime CSV writes to a temporary path for isolated testing."""
    runtime_csv = tmp_path / "expenses.csv"
    monkeypatch.setattr(tools, "RUNTIME_DATA_PATH", runtime_csv)
    monkeypatch.setattr(tools, "_ensure_runtime_csv", lambda seed_path=tools.SEED_DATA_PATH, runtime_path=runtime_csv: (
        runtime_csv if runtime_csv.exists() else (__import__("shutil").copy2(seed_path, runtime_csv) or runtime_csv)
    ))
    return runtime_csv


def test_get_expenses(isolated_runtime: Path) -> None:
    """Verify get_expenses filters by category and month."""
    records = tools.get_expenses(category="Food", month=11, year=2025)
    assert isinstance(records, list)
    assert all(r["category"] == "Food" for r in records)


def test_calculate_spending(isolated_runtime: Path) -> None:
    """Verify calculate_spending returns a non-negative float."""
    total = tools.calculate_spending(category="Shopping", month=10, year=2025)
    assert isinstance(total, float)
    assert total >= 0.0


def test_compare_months(isolated_runtime: Path) -> None:
    """Verify compare_months returns expected comparison keys."""
    comp = tools.compare_months(10, 11, year=2025)
    assert comp["month_a"] == 10
    assert comp["month_b"] == 11
    assert "difference" in comp and "percent_change" in comp


def test_get_highest_category(isolated_runtime: Path) -> None:
    """Verify get_highest_category returns valid category and amount."""
    top = tools.get_highest_category()
    assert top["category"] in {"Food", "Travel", "Shopping", "Utilities"}
    assert top["amount"] > 0


def test_get_category_breakdown(isolated_runtime: Path) -> None:
    """Verify get_category_breakdown returns all four categories."""
    breakdown = tools.get_category_breakdown()
    assert set(breakdown.keys()) == {"Food", "Travel", "Shopping", "Utilities"}


def test_add_expense_and_duplicate_id(isolated_runtime: Path) -> None:
    """Verify add_expense writes to runtime CSV and blocks duplicate IDs."""
    created = tools.add_expense("Uber ride to office", 250.0, "2025-11-20", expense_id="EXP-9999")
    assert created["expense_id"] == "EXP-9999"
    assert created["category"] == "Travel"

    with pytest.raises(ValueError, match="Duplicate"):
        tools.add_expense("Duplicate test", 100.0, "2025-11-21", expense_id="EXP-9999")


@pytest.mark.parametrize("bad_date", ["20-11-2025", "not-a-date", "2025/11/20", 123])
def test_validator_bad_date(bad_date: object) -> None:
    """Verify invalid dates raise ValueError."""
    with pytest.raises(ValueError):
        validators.validate_date(bad_date)  # type: ignore[arg-type]


@pytest.mark.parametrize("bad_amount", [-50, 0, "100", True])
def test_validator_bad_amount(bad_amount: object) -> None:
    """Verify negative, zero, or non-numeric amounts raise ValueError."""
    with pytest.raises(ValueError):
        validators.validate_amount(bad_amount)


def test_validator_duplicate_id() -> None:
    """Verify duplicate expense_id check raises ValueError."""
    df = pd.DataFrame([{"expense_id": "EXP-0001"}])
    with pytest.raises(ValueError, match="Duplicate"):
        validators.validate_expense_id("EXP-0001", df)
