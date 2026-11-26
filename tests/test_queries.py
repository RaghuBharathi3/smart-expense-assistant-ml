"""Tests for LLM query routing, whitelisted execution, and Gradio UI."""

from __future__ import annotations

import pandas as pd
import pytest

from src.backend.router import execute_tool_call, process_query, register_llm_handler
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


@pytest.mark.parametrize(
    ("qwen_payload", "expected_fn"),
    [
        (
            {"intent": "calculate_spending", "category": "Food", "month": "November", "amount": None, "description": None},
            "calculate_spending",
        ),
        (
            {"intent": "highest_category", "category": None, "month": None, "amount": None, "description": None},
            "get_highest_category",
        ),
        (
            {"intent": "compare_months", "category": None, "month": "October and November", "amount": None, "description": None},
            "compare_months",
        ),
        (
            {"intent": "categorize_expense", "category": None, "month": None, "amount": 250, "description": "Uber ride to office"},
            "categorize_expense",
        ),
        (
            {"intent": "show_expenses", "category": None, "month": "November", "amount": None, "description": None},
            "get_expenses",
        ),
    ],
)
def test_qwen_json_schema_execution(qwen_payload: dict, expected_fn: str) -> None:
    """Verify teammate Qwen2.5 JSON payloads normalize, validate, and execute."""
    out = process_query("ignored", llm_json=qwen_payload)
    assert out["ok"] is True
    assert out["function"] == expected_fn
    assert out["answer"]


def test_registered_qwen_handler_and_fallback() -> None:
    """Verify register_llm_handler executes valid Qwen JSON and falls back on malformed output."""
    try:
        register_llm_handler(
            lambda q: '{"intent": "calculate_spending", "category": "Shopping", "month": "October", "amount": null, "description": null}'
        )
        res = process_query("How much did I spend on shopping in October?")
        assert res["ok"] is True
        assert res["source"] == "qwen_llm"
        assert res["function"] == "calculate_spending"

        register_llm_handler(lambda q: "not valid json")
        fallback_res = process_query("How much did I spend on food in November?")
        assert fallback_res["ok"] is True
        assert fallback_res["source"] == "fallback_router"
    finally:
        register_llm_handler(None)


def test_interface_app_renders() -> None:
    """Verify Streamlit app renders cleanly and Plotly charts build without errors."""
    from streamlit.testing.v1 import AppTest

    from src.backend.tools import get_expenses
    from src.interface.charts import (
        category_distribution_chart,
        monthly_comparison_chart,
        monthly_trend_chart,
        top_categories_chart,
    )

    at = AppTest.from_file("src/interface/app.py", default_timeout=20)
    at.run()
    assert not at.exception

    # Verify clicking a welcome prompt executes and populates conversation
    if at.button:
        welcome_btns = [b for b in at.button if b.key and b.key.startswith("welcome_prompt_")]
        assert len(welcome_btns) == 6
        welcome_btns[0].click().run()
        assert not at.exception
        assert len(at.chat_message) >= 2

    df = pd.DataFrame(get_expenses())
    assert category_distribution_chart(df) is not None
    assert monthly_comparison_chart(df, 10, 11) is not None
    assert monthly_trend_chart(df) is not None
    assert top_categories_chart(df) is not None
