"""Centralized configuration for the Smart Expense Assistant frontend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = ROOT_DIR / "data" / "expenses.csv"


@dataclass(frozen=True)
class AppConfig:
    """Application-level configuration loaded from environment with safe defaults."""

    app_title: str = "Smart Expense Assistant"
    currency_symbol: str = "₹"
    data_path: Path = Path(os.getenv("EXPENSE_DATA_PATH", str(DEFAULT_DATA_PATH)))
    api_base_url: str | None = os.getenv("EXPENSE_API_URL") or None
    api_timeout_seconds: float = float(os.getenv("EXPENSE_API_TIMEOUT", "10.0"))
    force_mock_mode: bool = os.getenv("EXPENSE_MOCK_MODE", "false").lower() == "true"


CONFIG = AppConfig()
