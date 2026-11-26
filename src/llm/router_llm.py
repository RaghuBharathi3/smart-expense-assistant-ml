"""Convert natural-language user queries into structured JSON tool calls."""

from __future__ import annotations

import json
import re
from typing import Any

import requests

from src.config import ALLOWED_CATEGORIES
from src.llm.load_model import load_llm_client
from src.llm.prompts import ROUTER_SYSTEM_PROMPT

MONTH_MAP = {
    "january": 1, "jan": 1, "february": 2, "feb": 2, "march": 3, "mar": 3,
    "april": 4, "apr": 4, "may": 5, "june": 6, "jun": 6, "july": 7, "jul": 7,
    "august": 8, "aug": 8, "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10, "november": 11, "nov": 11, "december": 12, "dec": 12,
}


def _extract_months(text: str) -> list[int]:
    found: list[int] = []
    for name, num in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", text) and num not in found:
            found.append(num)
    return found


def _extract_category(text: str) -> str | None:
    for cat in ALLOWED_CATEGORIES:
        if re.search(rf"\b{cat.lower()}\b", text):
            return cat
    return None


def _extract_year(text: str) -> int | None:
    match = re.search(r"\b(20\d{2})\b", text)
    return int(match.group(1)) if match else 2025


def _rule_based_route(query: str) -> dict[str, Any]:
    lower = query.lower().strip()
    months = _extract_months(lower)
    category = _extract_category(lower)
    year = _extract_year(lower)

    if "categorize" in lower or "classify" in lower:
        cleaned = re.sub(r"^(categorize|classify)\s*(this\s*expense)?\s*[:\-]?\s*", "", query, flags=re.IGNORECASE).strip()
        cleaned = re.sub(r"[₹rs\.\s,]+\d+(?:\.\d+)?\s*$", "", cleaned, flags=re.IGNORECASE).strip(" ,.-")
        return {"function": "categorize_expense", "arguments": {"description": cleaned}}

    if "add" in lower and "expense" in lower:
        amt_match = re.search(r"(?:₹|rs\.?\s*)(\d+(?:\.\d+)?)", lower)
        date_match = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", lower)
        if amt_match and date_match:
            desc = re.sub(r"\badd\s+expense\b", "", query, flags=re.IGNORECASE)
            desc = re.sub(r"(?:₹|rs\.?\s*)\d+(?:\.\d+)?", "", desc, flags=re.IGNORECASE)
            desc = re.sub(r"\b20\d{2}-\d{2}-\d{2}\b", "", desc).strip(" ,.-")
            return {
                "function": "add_expense",
                "arguments": {
                    "description": desc or "Expense entry",
                    "amount": float(amt_match.group(1)),
                    "expense_date": date_match.group(1),
                    "category": category,
                },
            }

    if "compare" in lower and len(months) >= 2:
        return {
            "function": "compare_months",
            "arguments": {"month_a": months[0], "month_b": months[1], "year": year},
        }

    if "highest" in lower or "top category" in lower:
        args: dict[str, Any] = {}
        if months:
            args["month"] = months[0]
            args["year"] = year
        return {"function": "get_highest_category", "arguments": args}

    if "breakdown" in lower or "distribution" in lower:
        args = {"month": months[0], "year": year} if months else {}
        return {"function": "get_category_breakdown", "arguments": args}

    if ("show" in lower or "list" in lower or "all" in lower) and (months or category):
        args = {}
        if category:
            args["category"] = category
        if months:
            args["month"] = months[0]
            args["year"] = year
        return {"function": "get_expenses", "arguments": args}

    if ("how much" in lower or "total" in lower or "spend" in lower or "spent" in lower) and (months or category):
        args = {}
        if category:
            args["category"] = category
        if months:
            args["month"] = months[0]
            args["year"] = year
        return {"function": "calculate_spending", "arguments": args}

    return {"function": None, "error": "Sorry, I can only answer expense retrieval, summary, comparison, or categorization queries."}


def route_query(user_query: str) -> dict[str, Any]:
    """Convert user natural-language query into structured JSON tool call."""
    if not isinstance(user_query, str) or not user_query.strip():
        return {"function": None, "error": "Query cannot be empty."}

    client = load_llm_client()
    if client["mode"] == "api":
        try:
            resp = requests.post(
                client["api_url"],
                headers={"Authorization": f"Bearer {client['api_key']}"},
                json={
                    "model": client["model"],
                    "messages": [
                        {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                        {"role": "user", "content": user_query},
                    ],
                    "response_format": {"type": "json_object"},
                },
                timeout=client["timeout"],
            )
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (requests.RequestException, KeyError, json.JSONDecodeError):
            return _rule_based_route(user_query)

    return _rule_based_route(user_query)
