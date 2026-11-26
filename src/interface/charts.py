"""Plotly chart builders for the Smart Expense Assistant interface."""

from __future__ import annotations

import pandas as pd
import plotly.express as px

CHART_COLORS = ["#1E3A5F", "#0F766E", "#64748B", "#B45309"]


def _style_chart(fig: px.bar, title: str) -> px.bar:
    fig.update_layout(
        title=title,
        height=310,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(size=12, color="#4B5563"),
        title_font=dict(size=14, color="#111827"),
        margin=dict(t=42, b=16, l=12, r=12),
    )
    fig.update_yaxes(gridcolor="#F3F4F6", title_text="Amount (₹)")
    fig.update_xaxes(showgrid=False)
    return fig


def category_distribution_chart(df: pd.DataFrame):
    """Build category-wise spending donut chart."""
    grouped = df.groupby("category", as_index=False)["amount"].sum()
    fig = px.pie(
        grouped,
        names="category",
        values="amount",
        hole=0.52,
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: ₹%{value:,.2f}<extra></extra>")
    return _style_chart(fig, "Category-Wise Distribution")


def monthly_comparison_chart(df: pd.DataFrame, month_a: int = 10, month_b: int = 11):
    """Build side-by-side category comparison for two months."""
    temp = df.copy()
    temp["month"] = pd.to_datetime(temp["date"]).dt.month
    sub = temp[temp["month"].isin([month_a, month_b])]
    grouped = sub.groupby(["month", "category"], as_index=False)["amount"].sum()
    grouped["Month"] = grouped["month"].astype(str)
    fig = px.bar(
        grouped,
        x="category",
        y="amount",
        color="Month",
        barmode="group",
        color_discrete_sequence=CHART_COLORS[:2],
    )
    fig.update_xaxes(title_text="Category")
    return _style_chart(fig, f"Monthly Comparison (Month {month_a} vs {month_b})")


def monthly_trend_chart(df: pd.DataFrame):
    """Build total monthly spend line chart."""
    temp = df.copy()
    temp["period"] = pd.to_datetime(temp["date"]).dt.to_period("M").astype(str)
    grouped = temp.groupby("period", as_index=False)["amount"].sum().sort_values("period")
    fig = px.line(grouped, x="period", y="amount", markers=True, color_discrete_sequence=[CHART_COLORS[0]])
    fig.update_xaxes(title_text="Month")
    return _style_chart(fig, "Total Monthly Spend")


def top_categories_chart(df: pd.DataFrame):
    """Build top spending categories bar chart."""
    grouped = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    fig = px.bar(grouped, x="category", y="amount", color="category", color_discrete_sequence=CHART_COLORS)
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="Category")
    return _style_chart(fig, "Top Spending Categories")
