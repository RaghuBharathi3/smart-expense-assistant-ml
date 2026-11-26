"""Tests for LLM query routing, whitelisted execution, and Streamlit UI."""

from __future__ import annotations

import pytest

from src.backend.router import execute_tool_call
from src.llm.router_llm import route_query


@pytest.mark.parametrize(
    ("query", "expected_fn"),
    [
        ("How much did I spend on food in November?", "calculate_spending"),
        ("Which category has the highest spending?", "get_highest_category"),
        ("Compare my expenses between October and November.", "compare_months"),
        ("Categorize this expense: Uber ride to office, ₹250", "categorize_expense"),
        ("Show all my expenses in November.", "get_expenses"),
        ("How much did I spend on shopping in October?", "calculate_spending"),
    ],
)
def test_sample_queries_route_and_execute(query: str, expected_fn: str) -> None:
    """Verify sample queries route to expected JSON function and execute cleanly."""
    routed = route_query(query)
    assert routed["function"] == expected_fn
    result = execute_tool_call(routed)
    assert result["ok"] is True


def test_router_blocks_unwhitelisted_function() -> None:
    """Verify non-whitelisted functions are rejected."""
    res = execute_tool_call({"function": "eval", "arguments": {"x": "1+1"}})
    assert res["ok"] is False
    assert "not whitelisted" in res["error"]


def test_interface_app_renders() -> None:
    """Verify Gradio Blocks app, chat handler, and charts build cleanly."""
    from src.backend.tools import get_expenses
    from src.interface.app import create_app, filter_ledger, handle_chat_query
    from src.interface.charts import (
        category_distribution_chart,
        monthly_comparison_chart,
        monthly_trend_chart,
        top_categories_chart,
    )
    import pandas as pd

    app = create_app()
    assert app is not None

    _, hist, table_upd, _, feedback, _ = handle_chat_query("Show all my expenses in November.", [])
    assert len(hist) == 2
    assert table_upd["visible"] is True
    assert not table_upd["value"].empty
    assert "Completed" in feedback

    _, hist2, _, chart_upd, _, _ = handle_chat_query("Compare my expenses between October and November.", [])
    assert len(hist2) == 2
    assert chart_upd["visible"] is True

    _, _, _, _, err_feedback, _ = handle_chat_query("   ", [])
    assert "error" in err_feedback

    filtered = filter_ledger("Food", "")
    assert not filtered.empty

    df = pd.DataFrame(get_expenses())
    assert category_distribution_chart(df) is not None
    assert monthly_comparison_chart(df, 10, 11) is not None
    assert monthly_trend_chart(df) is not None
    assert top_categories_chart(df) is not None
