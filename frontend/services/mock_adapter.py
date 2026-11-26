"""Explicitly labelled development fallback adapter for Task 4.

Used only when Task 2 (expense tools) or Task 3 (LLM integration) are not yet
implemented. All financial aggregations are computed strictly from the real
CSV dataset (data/expenses.csv) and Task 1's ML classifier (src/expense_classifier.py)
so no fake financial figures are ever fabricated.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from frontend.config import CONFIG
from frontend.contracts import AssistantResponse, ResponseType, VisualizationSpec
from frontend.theme import format_inr
from src.expense_classifier import categorize_expense

MONTH_MAP = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

CATEGORIES = ["food", "travel", "shopping", "utilities"]


def _extract_months(text: str) -> list[int]:
    lower = text.lower()
    found: list[int] = []
    for name, num in MONTH_MAP.items():
        if re.search(rf"\b{name}\b", lower) and num not in found:
            found.append(num)
    return found


def _extract_category(text: str) -> str | None:
    lower = text.lower()
    for cat in CATEGORIES:
        if re.search(rf"\b{cat}\b", lower):
            return cat.capitalize()
    return None


def _month_name(month_num: int) -> str:
    names = [
        "", "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ]
    return names[month_num] if 1 <= month_num <= 12 else str(month_num)


class MockAssistantAdapter:
    """Development fallback that answers expected queries using real CSV & Task 1 ML."""

    def __init__(self, data_path: Path = CONFIG.data_path) -> None:
        self.data_path = data_path

    def _load_df(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Expense dataset not found at {self.data_path}")
        df = pd.read_csv(self.data_path)
        df["date_dt"] = pd.to_datetime(df["date"], errors="coerce")
        df["month"] = df["date_dt"].dt.month
        df["year"] = df["date_dt"].dt.year
        return df

    def handle_query(self, user_message: str) -> AssistantResponse:
        """Route natural-language query to real CSV analytics or Task 1 ML classifier."""
        text = user_message.strip()
        if not text:
            return AssistantResponse(
                response_type=ResponseType.ERROR,
                message="Please enter a question or expense description.",
                source="mock",
                error="Empty input",
            )

        lower = text.lower()

        # 1. Categorize a new expense using Task 1's actual ML model
        if "categorize" in lower or "classify" in lower:
            cleaned_desc = re.sub(
                r"^(categorize|classify)\s*(this\s*expense)?\s*[:\-]?\s*",
                "",
                text,
                flags=re.IGNORECASE,
            ).strip()
            # Strip trailing amount like ", ₹250" or "rs 250" for cleaner ML input
            amount_match = re.search(r"[₹rs\.\s,]+(\d+(?:\.\d+)?)\s*$", cleaned_desc, re.IGNORECASE)
            ml_desc = cleaned_desc[: amount_match.start()].strip(" ,.-") if amount_match else cleaned_desc
            if not ml_desc:
                ml_desc = cleaned_desc
            try:
                pred = categorize_expense(ml_desc)
                return AssistantResponse(
                    response_type=ResponseType.CATEGORIZATION,
                    message=(
                        f"Predicted category for **\"{pred['description']}\"** is "
                        f"**{pred['category']}** (confidence: {pred['confidence']:.2%})."
                    ),
                    structured_data=pred,
                    source="ml_classifier",
                )
            except (ValueError, TypeError) as exc:
                return AssistantResponse(
                    response_type=ResponseType.ERROR,
                    message=f"Could not categorize expense: {exc}",
                    source="ml_classifier",
                    error=str(exc),
                )

        df = self._load_df()
        months = _extract_months(lower)
        category = _extract_category(lower)

        # 2. Monthly comparison (e.g., "Compare my expenses between October and November")
        if "compare" in lower and len(months) >= 2:
            m_a, m_b = months[0], months[1]
            target_year = 2025
            df_a = df[(df["month"] == m_a) & (df["year"] == target_year)]
            df_b = df[(df["month"] == m_b) & (df["year"] == target_year)]
            total_a = round(float(df_a["amount"].sum()), 2)
            total_b = round(float(df_b["amount"].sum()), 2)
            diff = round(total_b - total_a, 2)
            pct = round((diff / total_a) * 100, 2) if total_a > 0 else 0.0
            by_cat_a = df_a.groupby("category")["amount"].sum().round(2).to_dict()
            by_cat_b = df_b.groupby("category")["amount"].sum().round(2).to_dict()
            label_a = f"{_month_name(m_a)} {target_year}"
            label_b = f"{_month_name(m_b)} {target_year}"
            direction = "higher" if diff > 0 else "lower"
            return AssistantResponse(
                response_type=ResponseType.MONTHLY_COMPARISON,
                message=(
                    f"In **{label_b}**, you spent **{format_inr(total_b)}** compared to "
                    f"**{format_inr(total_a)}** in **{label_a}** "
                    f"({format_inr(abs(diff))} / {abs(pct):.1f}% {direction})."
                ),
                structured_data={
                    "month_a": label_a,
                    "month_b": label_b,
                    "total_a": total_a,
                    "total_b": total_b,
                    "difference": diff,
                    "percent_change": pct,
                    "by_category_a": by_cat_a,
                    "by_category_b": by_cat_b,
                },
                visualization=VisualizationSpec(
                    chart_type="comparison",
                    title=f"{label_a} vs {label_b} Spending by Category",
                    data=[
                        {"category": cat, label_a: float(by_cat_a.get(cat, 0.0)), label_b: float(by_cat_b.get(cat, 0.0))}
                        for cat in sorted(df["category"].unique())
                    ],
                ),
                source="mock",
            )

        # 3. Highest spending category
        if "highest" in lower or "top category" in lower or "most" in lower:
            subset = df[df["month"] == months[0]] if months else df
            grouped = subset.groupby("category")["amount"].sum().sort_values(ascending=False)
            top_cat = str(grouped.index[0])
            top_val = round(float(grouped.iloc[0]), 2)
            breakdown = {str(k): round(float(v), 2) for k, v in grouped.items()}
            scope = f"in {_month_name(months[0])}" if months else "across the full dataset"
            return AssistantResponse(
                response_type=ResponseType.CATEGORY_BREAKDOWN,
                message=(
                    f"Your highest spending category {scope} is **{top_cat}** "
                    f"with **{format_inr(top_val)}**."
                ),
                structured_data={"highest_category": top_cat, "amount": top_val, "breakdown": breakdown},
                visualization=VisualizationSpec(
                    chart_type="donut",
                    title=f"Spending Breakdown ({scope})",
                    data=breakdown,
                ),
                source="mock",
            )

        # 4. Show all expenses in a month (transaction list)
        if ("show" in lower or "list" in lower or "all" in lower) and months:
            m = months[0]
            target_year = 2025
            subset = df[(df["month"] == m) & (df["year"] == target_year)]
            if category:
                subset = subset[subset["category"] == category]
            subset = subset.sort_values("date", ascending=False)
            records = subset[["expense_id", "description", "amount", "date", "category"]].to_dict(orient="records")
            total = round(float(subset["amount"].sum()), 2)
            return AssistantResponse(
                response_type=ResponseType.TRANSACTION_LIST,
                message=(
                    f"Found **{len(records)}** expense(s) in **{_month_name(m)} {target_year}** "
                    f"totaling **{format_inr(total)}**."
                ),
                structured_data={"count": len(records), "total": total, "records": records},
                source="mock",
            )

        # 5. Category + Month spending summary (e.g., "How much did I spend on food in November?")
        if category or months:
            subset = df.copy()
            parts: list[str] = []
            if category:
                subset = subset[subset["category"] == category]
                parts.append(category)
            if months:
                subset = subset[(subset["month"] == months[0]) & (subset["year"] == 2025)]
                parts.append(f"{_month_name(months[0])} 2025")
            total = round(float(subset["amount"].sum()), 2)
            count = int(len(subset))
            label = " in ".join(parts) if parts else "All expenses"
            return AssistantResponse(
                response_type=ResponseType.SUMMARY,
                message=(
                    f"You spent **{format_inr(total)}** on **{label}** "
                    f"across **{count}** transaction(s)."
                ),
                structured_data={"label": label, "total": total, "count": count},
                source="mock",
            )

        # 6. Unsupported / general question
        return AssistantResponse(
            response_type=ResponseType.TEXT,
            message=(
                "I'm currently running in **Development Fallback mode** (Task 2/3 backend not yet connected). "
                "Try asking about monthly totals (e.g., *'How much did I spend on food in November?'*), "
                "comparisons (*'Compare October and November'*), transaction lists, or categorizing an expense."
            ),
            source="mock",
        )
