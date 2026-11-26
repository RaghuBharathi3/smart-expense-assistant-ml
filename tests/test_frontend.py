"""Automated tests for Task 4 frontend services, contracts, formatting, and Streamlit app."""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest
import requests

from frontend.contracts import ResponseType
from frontend.services.assistant_service import (
    AssistantService,
    compute_dashboard_metrics,
    load_expense_dataframe,
)
from frontend.theme import format_indian_date, format_inr


def test_format_inr() -> None:
    """Verify Indian Rupee formatting."""
    assert format_inr(250) == "₹250.00"
    assert format_inr(12450.5) == "₹12,450.50"
    assert format_inr(1234567.89) == "₹12,34,567.89"


def test_format_indian_date() -> None:
    """Verify ISO date converts to DD MMM YYYY."""
    assert format_indian_date("2025-11-14") == "14 Nov 2025"


def test_dashboard_metrics_real_and_empty() -> None:
    """Verify dashboard KPIs compute cleanly on real data and empty DataFrame."""
    df = load_expense_dataframe()
    assert not df.empty
    metrics = compute_dashboard_metrics(df)
    assert metrics.total_expenses > 0
    assert metrics.record_count == len(df)
    assert metrics.highest_category in {"Food", "Travel", "Shopping", "Utilities"}

    empty_metrics = compute_dashboard_metrics(pd.DataFrame())
    assert empty_metrics.total_expenses == 0.0
    assert empty_metrics.record_count == 0


@pytest.mark.parametrize(
    ("query", "expected_type"),
    [
        ("How much did I spend on food in November?", ResponseType.SUMMARY),
        ("Which category has the highest spending?", ResponseType.CATEGORY_BREAKDOWN),
        ("Compare my expenses between October and November.", ResponseType.MONTHLY_COMPARISON),
        ("Categorize this expense: Uber ride to office, ₹250", ResponseType.CATEGORIZATION),
        ("Show all my expenses in November.", ResponseType.TRANSACTION_LIST),
        ("How much did I spend on shopping in October?", ResponseType.SUMMARY),
    ],
)
def test_expected_queries_via_service(query: str, expected_type: ResponseType) -> None:
    """Verify all 6 expected queries return structured, non-error responses."""
    service = AssistantService()
    resp = service.send_message(query)
    assert resp.response_type == expected_type
    assert resp.message
    assert resp.error is None


def test_categorization_uses_real_ml_model() -> None:
    """Verify categorization query routes to Task 1 ML classifier."""
    service = AssistantService()
    resp = service.send_message("Categorize this expense: Uber ride to office, ₹250")
    assert resp.source == "ml_classifier"
    assert resp.structured_data["category"] == "Travel"
    assert 0.0 < resp.structured_data["confidence"] <= 1.0


def test_empty_input_returns_error_response() -> None:
    """Verify blank message returns a clean ERROR response instead of crashing."""
    service = AssistantService()
    resp = service.send_message("   ")
    assert resp.response_type == ResponseType.ERROR
    assert resp.error is not None


def test_http_backend_unavailable_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify HTTP backend failures produce graceful ERROR responses."""
    monkeypatch.setattr("frontend.services.assistant_service.CONFIG", type(
        "MockCfg", (), {"force_mock_mode": False, "api_base_url": "http://localhost:9999", "api_timeout_seconds": 1.0}
    )())
    service = AssistantService()

    with patch("frontend.services.assistant_service.requests.post", side_effect=requests.Timeout):
        resp = service.send_message("Food in November")
        assert resp.response_type == ResponseType.ERROR
        assert "timeout" in (resp.error or "").lower()

    with patch("frontend.services.assistant_service.requests.post", side_effect=requests.ConnectionError("refused")):
        resp = service.send_message("Food in November")
        assert resp.response_type == ResponseType.ERROR
        assert "unavailable" in resp.message.lower()


def test_streamlit_app_renders_cleanly() -> None:
    """Verify app.py runs and renders without uncaught exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("app.py", default_timeout=15)
    at.run()
    assert not at.exception
