"""Financial dashboard and interactive Plotly visualizations for Task 4."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.services.assistant_service import compute_dashboard_metrics
from frontend.theme import PALETTE, format_indian_date, format_inr


def _render_kpi_card(col: Any, label: str, value: str, subtitle: str) -> None:
    col.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_dashboard_panel(df: pd.DataFrame) -> None:
    """Render summary KPI cards, interactive charts, and searchable expense table."""
    st.subheader("Financial Overview & Analytics")

    if df.empty:
        st.warning("No expense records available. Ensure data/expenses.csv exists or connect Task 2.")
        return

    metrics = compute_dashboard_metrics(df)

    # A. Summary KPI Cards
    c1, c2, c3, c4 = st.columns(4)
    _render_kpi_card(
        c1,
        "Total Expenses",
        format_inr(metrics.total_expenses),
        f"{metrics.record_count} recorded transactions",
    )
    _render_kpi_card(
        c2,
        f"Latest Month ({metrics.current_month_label})",
        format_inr(metrics.current_month_spending),
        "Most recent month in dataset",
    )
    _render_kpi_card(
        c3,
        "Highest Category",
        metrics.highest_category,
        f"Total: {format_inr(metrics.highest_category_amount)}",
    )
    _render_kpi_card(
        c4,
        "Average Daily Spending",
        format_inr(metrics.average_daily_spending),
        "Across active transaction days",
    )

    st.markdown("")

    # B. Interactive Visualizations
    cat_totals = df.groupby("category", as_index=False)["amount"].sum().sort_values("amount", ascending=False)
    df_trend = df.copy()
    df_trend["month_period"] = df_trend["date"].dt.to_period("M").astype(str)
    monthly_totals = df_trend.groupby("month_period", as_index=False)["amount"].sum().sort_values("month_period")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_donut = px.pie(
            cat_totals,
            names="category",
            values="amount",
            hole=0.52,
            title="Category-Wise Spending (Donut)",
            color_discrete_sequence=PALETTE["chart_colors"],
        )
        fig_donut.update_traces(textinfo="label+percent", hovertemplate="%{label}: ₹%{value:,.2f}<extra></extra>")
        fig_donut.update_layout(margin=dict(t=45, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig_donut, use_container_width=True, key="dash_donut")

    with col_right:
        fig_line = px.line(
            monthly_totals,
            x="month_period",
            y="amount",
            markers=True,
            title="Monthly Expense Trend",
            color_discrete_sequence=[PALETTE["secondary"]],
        )
        fig_line.update_xaxes(title_text="Month")
        fig_line.update_yaxes(title_text="Amount (₹)")
        fig_line.update_traces(hovertemplate="%{x}: ₹%{y:,.2f}<extra></extra>")
        fig_line.update_layout(margin=dict(t=45, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig_line, use_container_width=True, key="dash_line")

    col_bar, col_comp = st.columns(2)
    with col_bar:
        fig_bar = px.bar(
            cat_totals,
            x="category",
            y="amount",
            title="Category Comparison",
            color="category",
            color_discrete_sequence=PALETTE["chart_colors"],
        )
        fig_bar.update_xaxes(title_text="Category")
        fig_bar.update_yaxes(title_text="Amount (₹)")
        fig_bar.update_layout(showlegend=False, margin=dict(t=45, b=10, l=10, r=10), height=340)
        st.plotly_chart(fig_bar, use_container_width=True, key="dash_bar")

    with col_comp:
        oct_nov = df_trend[df_trend["month_period"].isin(["2025-10", "2025-11"])]
        if not oct_nov.empty:
            oct_nov_grp = oct_nov.groupby(["month_period", "category"], as_index=False)["amount"].sum()
            oct_nov_grp["Month"] = oct_nov_grp["month_period"].map(
                {"2025-10": "October 2025", "2025-11": "November 2025"}
            )
            fig_comp = px.bar(
                oct_nov_grp,
                x="category",
                y="amount",
                color="Month",
                barmode="group",
                title="Monthly Comparison: October vs November 2025",
                color_discrete_sequence=[PALETTE["secondary"], PALETTE["accent"]],
            )
            fig_comp.update_xaxes(title_text="Category")
            fig_comp.update_yaxes(title_text="Amount (₹)")
            fig_comp.update_layout(margin=dict(t=45, b=10, l=10, r=10), height=340)
            st.plotly_chart(fig_comp, use_container_width=True, key="dash_oct_nov")
        else:
            st.info("No October/November 2025 records found for comparison.")

    # C. Searchable, Filterable Expense Transaction Table
    st.subheader("Recent Expense Transactions")
    f1, f2, f3 = st.columns([2, 1, 1])
    search_text = f1.text_input("Search description", placeholder="e.g., Swiggy, Uber, electricity")
    categories = ["All"] + sorted(df["category"].unique().tolist())
    selected_cat = f2.selectbox("Category", categories)
    sort_option = f3.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Amount (High to Low)", "Amount (Low to High)"])

    filtered = df.copy()
    if search_text.strip():
        filtered = filtered[filtered["description"].str.contains(search_text.strip(), case=False, na=False)]
    if selected_cat != "All":
        filtered = filtered[filtered["category"] == selected_cat]

    if sort_option == "Date (Newest)":
        filtered = filtered.sort_values("date", ascending=False)
    elif sort_option == "Date (Oldest)":
        filtered = filtered.sort_values("date", ascending=True)
    elif sort_option == "Amount (High to Low)":
        filtered = filtered.sort_values("amount", ascending=False)
    else:
        filtered = filtered.sort_values("amount", ascending=True)

    display_df = filtered.head(100).copy()
    display_df["Date"] = display_df["date"].dt.strftime("%Y-%m-%d").apply(format_indian_date)
    display_df["Amount"] = display_df["amount"].apply(format_inr)
    st.dataframe(
        display_df[["expense_id", "Date", "description", "category", "Amount"]].rename(
            columns={"expense_id": "ID", "description": "Description", "category": "Category"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = filtered[["expense_id", "description", "amount", "date", "category"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Filtered Transactions (CSV)",
        data=csv_bytes,
        file_name="filtered_expenses.csv",
        mime="text/csv",
    )
