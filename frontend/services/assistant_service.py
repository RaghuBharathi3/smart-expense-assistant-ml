"""Unified frontend service layer for dashboard data and chat queries."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from frontend.config import CONFIG
from frontend.contracts import (
    AssistantResponse,
    DashboardMetrics,
    ResponseType,
    VisualizationSpec,
)
from frontend.services.mock_adapter import MockAssistantAdapter


def load_expense_dataframe(data_path: Path = CONFIG.data_path) -> pd.DataFrame:
    """Load expense records from CSV for dashboard KPIs and charts."""
    if not data_path.exists():
        return pd.DataFrame(columns=["expense_id", "description", "amount", "date", "category"])
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "amount", "category"]).copy()
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0.0)
    return df


def compute_dashboard_metrics(df: pd.DataFrame) -> DashboardMetrics:
    """Compute summary KPI cards strictly from actual dataset records."""
    if df.empty:
        return DashboardMetrics(
            total_expenses=0.0,
            current_month_spending=0.0,
            current_month_label="N/A",
            highest_category="N/A",
            highest_category_amount=0.0,
            average_daily_spending=0.0,
            record_count=0,
        )

    total = round(float(df["amount"].sum()), 2)
    latest_period = df["date"].dt.to_period("M").max()
    current_month_df = df[df["date"].dt.to_period("M") == latest_period]
    current_month_spending = round(float(current_month_df["amount"].sum()), 2)

    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    highest_cat = str(cat_totals.index[0])
    highest_amt = round(float(cat_totals.iloc[0]), 2)

    unique_days = max(int(df["date"].dt.date.nunique()), 1)
    avg_daily = round(total / unique_days, 2)

    return DashboardMetrics(
        total_expenses=total,
        current_month_spending=current_month_spending,
        current_month_label=latest_period.strftime("%b %Y"),
        highest_category=highest_cat,
        highest_category_amount=highest_amt,
        average_daily_spending=avg_daily,
        record_count=int(len(df)),
    )


class AssistantService:
    """Facade communicating with Task 3 HTTP API, Python module, or mock fallback."""

    def __init__(self) -> None:
        self._mock = MockAssistantAdapter()

    def get_backend_mode(self) -> str:
        """Return active backend mode label for transparent UI status display."""
        if CONFIG.force_mock_mode:
            return "Development Fallback (Forced Mock Mode)"
        if CONFIG.api_base_url:
            return f"HTTP Backend ({CONFIG.api_base_url})"
        if importlib.util.find_spec("src.assistant") is not None:
            return "Local Task 3 Module (src.assistant)"
        return "Development Fallback (Task 2/3 Pending — Real CSV + Task 1 ML)"

    def send_message(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> AssistantResponse:
        """Send user message to the active backend with graceful error handling."""
        if not isinstance(user_message, str) or not user_message.strip():
            return AssistantResponse(
                response_type=ResponseType.ERROR,
                message="Please enter a non-empty message.",
                source="frontend",
                error="Empty message",
            )

        if not CONFIG.force_mock_mode and CONFIG.api_base_url:
            return self._call_http_backend(user_message, history or [])

        if not CONFIG.force_mock_mode and importlib.util.find_spec("src.assistant") is not None:
            return self._call_python_backend(user_message, history or [])

        return self._mock.handle_query(user_message)

    def _call_http_backend(
        self,
        user_message: str,
        history: list[dict[str, Any]],
    ) -> AssistantResponse:
        try:
            resp = requests.post(
                f"{CONFIG.api_base_url.rstrip('/')}/chat",
                json={"message": user_message, "history": history},
                timeout=CONFIG.api_timeout_seconds,
            )
            resp.raise_for_status()
            payload = resp.json()
            return self._parse_backend_payload(payload)
        except requests.Timeout:
            return AssistantResponse(
                response_type=ResponseType.ERROR,
                message="The backend took too long to respond. Please retry.",
                source="backend",
                error="Request timeout",
            )
        except requests.RequestException as exc:
            return AssistantResponse(
                response_type=ResponseType.ERROR,
                message="Backend service is currently unavailable. Please check the server or retry.",
                source="backend",
                error=str(exc),
            )

    def _call_python_backend(
        self,
        user_message: str,
        history: list[dict[str, Any]],
    ) -> AssistantResponse:
        try:
            mod = importlib.import_module("src.assistant")
            payload = mod.handle_query(user_message, history=history)
            return self._parse_backend_payload(payload)
        except Exception as exc:  # noqa: BLE001
            return AssistantResponse(
                response_type=ResponseType.ERROR,
                message="An error occurred in the assistant module.",
                source="backend",
                error=str(exc),
            )

    @staticmethod
    def _parse_backend_payload(payload: dict[str, Any]) -> AssistantResponse:
        raw_type = payload.get("response_type", "text")
        try:
            rtype = ResponseType(raw_type)
        except ValueError:
            rtype = ResponseType.TEXT

        viz_raw = payload.get("visualization")
        viz = (
            VisualizationSpec(
                chart_type=viz_raw.get("chart_type", "bar"),
                title=viz_raw.get("title", "Expense Visualization"),
                data=viz_raw.get("data", {}),
                x_label=viz_raw.get("x_label", ""),
                y_label=viz_raw.get("y_label", "Amount (₹)"),
            )
            if isinstance(viz_raw, dict)
            else None
        )
        return AssistantResponse(
            response_type=rtype,
            message=str(payload.get("message", "")),
            structured_data=payload.get("structured_data", {}),
            visualization=viz,
            source="backend",
            error=payload.get("error"),
        )
