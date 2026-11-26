"""Clean, warm-paper Plotly charts integrated naturally into conversations."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Restrained warm editorial palette (Terracotta, Forest Sage, Slate, Ochre)
CHART_COLORS = ["#C2593F", "#2D6A4F", "#5C677D", "#B07D3C"]

MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}


def _empty_figure(title: str, message: str = "No matching expense records found.") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=13, color="#6B6860"),
    )
    fig.update_layout(
        title=title,
        height=300,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(t=46, b=18, l=16, r=16),
    )
    return fig


def _style_chart(fig: go.Figure, title: str, height: int = 310) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=13.5, color="#1A1917"), x=0.01, xanchor="left"),
        height=height,
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        font=dict(
            family="-apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif",
            size=12,
            color="#5C5952",
        ),
        margin=dict(t=46, b=22, l=14, r=14),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.24,
            xanchor="left",
            x=0,
            font=dict(size=11, color="#5C5952"),
        ),
        hoverlabel=dict(
            bgcolor="#1A1917",
            bordercolor="#1A1917",
            font_size=12,
            font_color="#FAF9F6",
        ),
    )
    fig.update_yaxes(gridcolor="#F1EFEA", zerolinecolor="#E6E4DD", title_text="Amount (₹)")
    fig.update_xaxes(showgrid=False, linecolor="#E6E4DD")
    return fig


def category_distribution_chart(df: pd.DataFrame) -> go.Figure:
    """Build category-wise spending donut chart with center total."""
    if df.empty:
        return _empty_figure("Category Distribution")
    grouped = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    total = float(grouped["amount"].sum())
    fig = px.pie(
        grouped,
        names="category",
        values="amount",
        hole=0.62,
        color_discrete_sequence=CHART_COLORS,
    )
    fig.update_traces(
        textinfo="percent",
        textposition="inside",
        hovertemplate="<b>%{label}</b><br>₹%{value:,.2f} (%{percent})<extra></extra>",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    fig.add_annotation(
        text=f"<b>₹{total:,.0f}</b><br><span style='font-size:11px;color:#6B6860'>Total</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=14, color="#1A1917"),
    )
    return _style_chart(fig, "Spending by Category")


def monthly_comparison_chart(df: pd.DataFrame, month_a: int = 10, month_b: int = 11) -> go.Figure:
    """Build side-by-side category comparison for two selected months."""
    title = f"{MONTH_NAMES.get(month_a, month_a)} vs {MONTH_NAMES.get(month_b, month_b)}"
    if df.empty:
        return _empty_figure(title)
    temp = df.copy()
    temp["month"] = pd.to_datetime(temp["date"], errors="coerce").dt.month
    sub = temp[temp["month"].isin([month_a, month_b])]
    if sub.empty:
        return _empty_figure(title)
    grouped = sub.groupby(["month", "category"], as_index=False)["amount"].sum()
    grouped["Month"] = grouped["month"].map(lambda m: MONTH_NAMES.get(int(m), str(m)))
    fig = px.bar(
        grouped,
        x="category",
        y="amount",
        color="Month",
        barmode="group",
        color_discrete_sequence=CHART_COLORS[:2],
    )
    fig.update_traces(hovertemplate="<b>%{x}</b> · %{fullData.name}<br>₹%{y:,.2f}<extra></extra>")
    fig.update_xaxes(title_text="")
    return _style_chart(fig, title)


def monthly_trend_chart(df: pd.DataFrame) -> go.Figure:
    """Build total monthly spend trend line chart."""
    if df.empty:
        return _empty_figure("Monthly Spend Trend")
    temp = df.copy()
    temp["period"] = pd.to_datetime(temp["date"], errors="coerce").dt.to_period("M").astype(str)
    grouped = temp.groupby("period", as_index=False)["amount"].sum().sort_values("period")
    fig = px.line(
        grouped,
        x="period",
        y="amount",
        markers=True,
        line_shape="spline",
        color_discrete_sequence=[CHART_COLORS[0]],
    )
    fig.update_traces(
        line=dict(width=2.2),
        marker=dict(size=6),
        hovertemplate="<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>",
    )
    fig.update_xaxes(title_text="")
    return _style_chart(fig, "Monthly Spend Trend")


def top_categories_chart(df: pd.DataFrame) -> go.Figure:
    """Build top spending categories bar chart."""
    if df.empty:
        return _empty_figure("Top Spending Categories")
    grouped = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    fig = px.bar(grouped, x="category", y="amount", color="category", color_discrete_sequence=CHART_COLORS)
    fig.update_traces(hovertemplate="<b>%{x}</b><br>₹%{y:,.2f}<extra></extra>")
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text="")
    return _style_chart(fig, "Top Spending Categories")


def breakdown_dict_chart(breakdown: dict[str, float], title: str = "Category Breakdown") -> go.Figure:
    """Render a minimal donut chart from a category-to-amount dictionary."""
    if not breakdown:
        return _empty_figure(title)
    df_b = pd.DataFrame({"category": list(breakdown.keys()), "amount": list(breakdown.values())})
    fig = px.pie(df_b, names="category", values="amount", hole=0.62, color_discrete_sequence=CHART_COLORS)
    fig.update_traces(
        textinfo="percent",
        hovertemplate="<b>%{label}</b><br>₹%{value:,.2f}<extra></extra>",
        marker=dict(line=dict(color="#FFFFFF", width=2)),
    )
    return _style_chart(fig, title, height=290)
