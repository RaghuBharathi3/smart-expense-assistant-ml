"""Typed integration contracts between Task 4 (UI) and Tasks 1, 2, and 3."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol


class ResponseType(str, Enum):
    """Supported structured assistant response types."""

    TEXT = "text"
    SUMMARY = "summary"
    CATEGORY_BREAKDOWN = "category_breakdown"
    MONTHLY_COMPARISON = "monthly_comparison"
    TRANSACTION_LIST = "transaction_list"
    CATEGORIZATION = "categorization"
    VISUALIZATION = "visualization"
    ERROR = "error"


@dataclass
class ExpenseRecord:
    """Canonical expense transaction record."""

    expense_id: str
    description: str
    amount: float
    date: str
    category: str
    merchant: str | None = None


@dataclass
class VisualizationSpec:
    """Metadata describing a chart to render in the chat response."""

    chart_type: str  # "donut" | "bar" | "line" | "comparison"
    title: str
    data: dict[str, float] | list[dict[str, Any]]
    x_label: str = ""
    y_label: str = "Amount (₹)"


@dataclass
class AssistantResponse:
    """Stable response contract consumed by Task 4 from Task 3."""

    response_type: ResponseType
    message: str
    structured_data: dict[str, Any] = field(default_factory=dict)
    visualization: VisualizationSpec | None = None
    source: str = "backend"  # "backend" | "ml_classifier" | "mock"
    error: str | None = None


@dataclass
class DashboardMetrics:
    """Aggregated financial KPIs displayed on the dashboard."""

    total_expenses: float
    current_month_spending: float
    current_month_label: str
    highest_category: str
    highest_category_amount: float
    average_daily_spending: float
    record_count: int


class Task2ToolsProtocol(Protocol):
    """Expected interface from Task 2 (Expense Retrieval & Calculation Tools)."""

    def get_expenses(
        self,
        category: str | None = None,
        month: int | None = None,
        year: int | None = None,
    ) -> list[dict[str, Any]]: ...

    def calculate_total(
        self,
        category: str | None = None,
        month: int | None = None,
        year: int | None = None,
    ) -> float: ...

    def compare_months(
        self,
        month_a: int,
        month_b: int,
        year: int | None = None,
    ) -> dict[str, Any]: ...


class Task3AssistantProtocol(Protocol):
    """Expected interface from Task 3 (LLM Integration & Tool Calling)."""

    def handle_query(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]: ...
