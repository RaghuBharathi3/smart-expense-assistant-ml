"""Whitelisted router executing validated LLM JSON tool calls."""

from __future__ import annotations

from typing import Any, Callable

from src.backend import tools
from src.classifier.predict import predict

ALLOWED_TOOLS: dict[str, Callable[..., Any]] = {
    "get_expenses": tools.get_expenses,
    "calculate_spending": tools.calculate_spending,
    "compare_months": tools.compare_months,
    "get_highest_category": tools.get_highest_category,
    "add_expense": tools.add_expense,
    "get_category_breakdown": tools.get_category_breakdown,
    "categorize_expense": predict,
}


def execute_tool_call(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate LLM JSON payload against whitelist and execute predefined tool."""
    if not isinstance(payload, dict):
        return {"ok": False, "error": "Tool payload must be a JSON object."}

    fn_name = payload.get("function")
    if not isinstance(fn_name, str) or fn_name not in ALLOWED_TOOLS:
        return {
            "ok": False,
            "error": f"Function '{fn_name}' is not whitelisted or supported.",
        }

    args = payload.get("arguments", {})
    if not isinstance(args, dict):
        return {"ok": False, "error": "Tool arguments must be a JSON object."}

    try:
        result = ALLOWED_TOOLS[fn_name](**args)
        return {"ok": True, "function": fn_name, "result": result}
    except (TypeError, ValueError) as exc:
        return {"ok": False, "function": fn_name, "error": str(exc)}
