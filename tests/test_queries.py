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
    """Verify src/interface/app.py renders cleanly without exceptions."""
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("src/interface/app.py", default_timeout=15)
    at.run()
    assert not at.exception
