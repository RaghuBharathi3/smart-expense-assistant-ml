"""Format backend tool execution results into natural-language answers."""

from __future__ import annotations

from typing import Any


def format_inr(amount: float) -> str:
    """Format float as Indian Rupees."""
    return f"₹{amount:,.2f}"


def generate_response(tool_output: dict[str, Any]) -> str:
    """Convert structured router output into a clear user-facing response."""
    if not tool_output.get("ok"):
        return f"Unable to process request: {tool_output.get('error', 'Unknown error')}"

    fn = tool_output.get("function")
    result = tool_output.get("result")

    if fn == "calculate_spending":
        return f"Total spending is **{format_inr(float(result))}**."

    if fn == "get_highest_category":
        if not result or not result.get("category"):
            return "No matching expense records were found."
        return f"Highest spending category is **{result['category']}** with **{format_inr(float(result['amount']))}**."

    if fn == "compare_months":
        diff = float(result["difference"])
        direction = "higher" if diff >= 0 else "lower"
        return (
            f"Month {result['month_b']} spending (**{format_inr(float(result['total_b']))}**) is "
            f"**{format_inr(abs(diff))}** ({abs(float(result['percent_change'])):.1f}% {direction}) "
            f"than Month {result['month_a']} (**{format_inr(float(result['total_a']))}**)."
        )

    if fn == "get_expenses":
        count = len(result) if isinstance(result, list) else 0
        total = sum(float(r["amount"]) for r in result) if count else 0.0
        return f"Found **{count}** expense record(s) totaling **{format_inr(total)}**."

    if fn == "get_category_breakdown":
        parts = [f"{cat}: {format_inr(float(amt))}" for cat, amt in result.items()]
        return "Category breakdown — " + ", ".join(parts) + "."

    if fn == "categorize_expense":
        return (
            f"Predicted category for **\"{result['description']}\"** is "
            f"**{result['category']}** (confidence: {float(result['confidence']):.2%})."
        )

    if fn == "add_expense":
        return (
            f"Added expense **{result['expense_id']}**: {result['description']} "
            f"({format_inr(float(result['amount']))}, {result['category']}, {result['date']})."
        )

    return str(result)
