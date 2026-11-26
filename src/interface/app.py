"""Apple-inspired minimal Streamlit application for Smart Expense Assistant."""

from __future__ import annotations

import html
from datetime import date, datetime
from typing import Any

import pandas as pd
import streamlit as st

from src.backend.router import _LLM_HANDLER, execute_tool_call, process_query
from src.backend.tools import get_expenses
from src.classifier.predict import predict
from src.config import ALLOWED_CATEGORIES
from src.interface.charts import (
    MONTH_NAMES,
    breakdown_dict_chart,
    category_distribution_chart,
    monthly_comparison_chart,
    monthly_trend_chart,
    top_categories_chart,
)
from src.llm.respond import format_inr

SAMPLE_QUERIES = [
    ("Food in November", "How much did I spend on food in November?"),
    ("Highest Category", "Which category has the highest spending?"),
    ("Oct vs Nov", "Compare my expenses between October and November."),
    ("Categorize ₹250 Uber", "Categorize this expense: Uber ride to office, ₹250"),
    ("All Nov Expenses", "Show all my expenses in November."),
    ("Shopping in Oct", "How much did I spend on shopping in October?"),
]

CUSTOM_CSS = """
<style>
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", system-ui, sans-serif;
        color: #1D1D1F;
        -webkit-font-smoothing: antialiased;
    }
    .stApp {
        background-color: #F5F5F7;
    }
    .block-container {
        max-width: 1180px;
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
    }
    .ea-topbar {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 12px;
        padding-bottom: 14px;
        border-bottom: 1px solid rgba(0, 0, 0, 0.08);
        margin-bottom: 14px;
    }
    .ea-title {
        font-size: 1.45rem;
        font-weight: 600;
        color: #1D1D1F;
        letter-spacing: -0.022em;
        margin: 0;
    }
    .ea-subtitle {
        font-size: 0.86rem;
        color: #6E6E73;
        margin: 3px 0 0 0;
    }
    .ea-status-pill {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-radius: 999px;
        padding: 5px 13px;
        font-size: 0.76rem;
        font-weight: 500;
        color: #1D1D1F;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
    }
    .ea-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background-color: #0F766E;
        display: inline-block;
    }
    .ea-kpi-card {
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.07);
        border-radius: 12px;
        padding: 15px 17px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }
    .ea-kpi-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.05);
    }
    .ea-kpi-label {
        font-size: 0.71rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.045em;
        color: #6E6E73;
        margin-bottom: 5px;
    }
    .ea-kpi-value {
        font-size: 1.32rem;
        font-weight: 600;
        color: #1D1D1F;
        font-variant-numeric: tabular-nums;
        letter-spacing: -0.015em;
    }
    .ea-kpi-sub {
        font-size: 0.76rem;
        color: #6E6E73;
        margin-top: 4px;
    }
    .ea-preview-pill {
        display: inline-block;
        background: #FFFFFF;
        border: 1px solid rgba(0, 0, 0, 0.08);
        border-left: 3px solid #0066CC;
        border-radius: 8px;
        padding: 8px 13px;
        font-size: 0.83rem;
        color: #1D1D1F;
        margin: 4px 0 10px 0;
    }
    .stButton > button {
        background-color: #FFFFFF;
        color: #1D1D1F;
        border: 1px solid rgba(0, 0, 0, 0.12);
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 500;
        transition: background-color 0.15s ease, border-color 0.15s ease, transform 0.1s ease;
    }
    .stButton > button:hover {
        background-color: #F2F2F7;
        border-color: #0066CC;
        color: #1D1D1F;
    }
    .stButton > button:active {
        transform: scale(0.985);
    }
    .stButton > button:focus-visible {
        outline: 2px solid #0066CC;
        outline-offset: 2px;
    }
    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.9rem;
            padding-right: 0.9rem;
            padding-top: 1rem;
        }
        .ea-kpi-value {
            font-size: 1.12rem;
        }
    }
</style>
"""


def _load_df() -> pd.DataFrame:
    records = get_expenses()
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["expense_id", "description", "amount", "date", "category"]
    )


def _format_indian_date(iso_date: str) -> str:
    try:
        return datetime.strptime(str(iso_date).strip(), "%Y-%m-%d").strftime("%d %b %Y")
    except ValueError:
        return str(iso_date)


def _render_kpi_card(col: Any, label: str, value: str, subtitle: str) -> None:
    col.markdown(
        f"""
        <div class="ea-kpi-card">
            <div class="ea-kpi-label">{html.escape(label)}</div>
            <div class="ea-kpi-value">{html.escape(value)}</div>
            <div class="ea-kpi-sub">{html.escape(subtitle)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_structured_result(tool_output: dict[str, Any], df: pd.DataFrame, key_prefix: str) -> None:
    """Render structured cards, interactive charts, or tables inside a chat message."""
    if not tool_output.get("ok"):
        return
    fn = tool_output.get("function")
    res = tool_output.get("result")

    if fn == "categorize_expense" and isinstance(res, dict):
        c1, c2 = st.columns(2)
        c1.metric("Predicted Category", str(res.get("category", "")))
        conf = float(res.get("confidence", 0.0))
        c2.metric("Confidence", f"{conf:.1%}")
        st.progress(min(max(conf, 0.0), 1.0))

    elif fn == "get_highest_category" and isinstance(res, dict) and res.get("category"):
        c1, c2 = st.columns(2)
        c1.metric("Top Category", str(res["category"]))
        c2.metric("Total Spend", format_inr(float(res["amount"])))

    elif fn == "calculate_spending" and res is not None:
        st.metric("Total Calculated Spend", format_inr(float(res)))

    elif fn == "get_category_breakdown" and isinstance(res, dict) and res:
        st.plotly_chart(
            breakdown_dict_chart(res, "Category Breakdown"),
            use_container_width=True,
            key=f"{key_prefix}_breakdown",
        )

    elif fn == "compare_months" and isinstance(res, dict):
        c1, c2, c3 = st.columns(3)
        c1.metric(f"Month {res['month_a']}", format_inr(float(res["total_a"])))
        c2.metric(f"Month {res['month_b']}", format_inr(float(res["total_b"])))
        c3.metric("Change", f"{float(res['percent_change']):+.1f}%")
        st.plotly_chart(
            monthly_comparison_chart(df, int(res["month_a"]), int(res["month_b"])),
            use_container_width=True,
            key=f"{key_prefix}_compare",
        )

    elif fn == "get_expenses" and isinstance(res, list):
        if res:
            df_res = pd.DataFrame(res).copy()
            df_res["Date"] = df_res["date"].apply(_format_indian_date)
            df_res["Amount"] = df_res["amount"].apply(lambda x: format_inr(float(x)))
            st.dataframe(
                df_res[["Date", "description", "category", "Amount"]].rename(
                    columns={"description": "Description", "category": "Category"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No matching expense records found.")


def _init_session_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = {"Chat 1": []}
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = "Chat 1"
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None
    if "pending_qwen_json" not in st.session_state:
        st.session_state.pending_qwen_json = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.subheader("Workspace")
        if st.button("+ New Conversation", use_container_width=True):
            name = f"Chat {len(st.session_state.conversations) + 1}"
            st.session_state.conversations[name] = []
            st.session_state.active_chat = name
            st.rerun()

        chat_names = list(st.session_state.conversations.keys())
        st.session_state.active_chat = st.radio(
            "Conversations",
            chat_names,
            index=chat_names.index(st.session_state.active_chat),
            label_visibility="collapsed",
        )

        if st.button("Clear Current Chat", use_container_width=True):
            st.session_state.conversations[st.session_state.active_chat] = []
            st.rerun()

        st.divider()
        engine_label = "Live Qwen2.5 Handler" if _LLM_HANDLER is not None else "Python Backend + Offline Router"
        st.caption("Routing Engine")
        st.info(engine_label)

        with st.expander("Test Raw Qwen2.5 JSON"):
            raw_json = st.text_area(
                "JSON Payload",
                value='{\n  "intent": "calculate_spending",\n  "category": "Food",\n  "month": "November",\n  "amount": null,\n  "description": null\n}',
                height=160,
            )
            if st.button("Run Qwen JSON", use_container_width=True):
                st.session_state.pending_qwen_json = raw_json


def _render_assistant_tab(df: pd.DataFrame) -> None:
    st.markdown("**Quick Prompts**")
    cols = st.columns(3)
    for idx, (label, full_query) in enumerate(SAMPLE_QUERIES):
        if cols[idx % 3].button(label, key=f"quick_{idx}", use_container_width=True):
            st.session_state.pending_query = full_query

    history: list[dict[str, Any]] = st.session_state.conversations[st.session_state.active_chat]

    for idx, msg in enumerate(history):
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                out = msg.get("output", {})
                if not out.get("ok"):
                    st.error(msg["content"])
                    if st.button("Retry", key=f"retry_{idx}") and idx > 0:
                        st.session_state.pending_query = history[idx - 1]["content"]
                        st.rerun()
                else:
                    st.markdown(msg["content"])
                    _render_structured_result(out, df, key_prefix=f"hist_{idx}")
            else:
                st.markdown(msg["content"])

    typed_query = st.chat_input("Ask anything about your expenses (e.g., 'How much did I spend on food in November?')")
    qwen_raw = st.session_state.pending_qwen_json
    st.session_state.pending_qwen_json = None

    query_to_run = typed_query or st.session_state.pending_query
    st.session_state.pending_query = None

    if qwen_raw:
        user_label = f"Qwen JSON: `{qwen_raw.strip()}`"
        history.append({"role": "user", "content": user_label})
        with st.chat_message("user"):
            st.markdown(user_label)
        with st.chat_message("assistant"):
            with st.spinner("Executing Qwen JSON via backend..."):
                out = process_query("", llm_json=qwen_raw)
            if not out.get("ok"):
                st.error(out["answer"])
            else:
                st.markdown(out["answer"])
                _render_structured_result(out, df, key_prefix=f"live_{len(history)}")
        history.append({"role": "assistant", "content": out["answer"], "output": out})

    elif query_to_run:
        history.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing expenses..."):
                out = process_query(query_to_run)
            if not out.get("ok"):
                st.error(out["answer"])
            else:
                st.markdown(out["answer"])
                _render_structured_result(out, df, key_prefix=f"live_{len(history)}")
        history.append({"role": "assistant", "content": out["answer"], "output": out})


def _render_dashboard_tab(df: pd.DataFrame) -> None:
    if df.empty:
        st.warning("No expense records available.")
        return

    total = float(df["amount"].sum())
    dates = pd.to_datetime(df["date"], errors="coerce")
    latest_period = dates.dt.to_period("M").max()
    latest_spend = float(df[dates.dt.to_period("M") == latest_period]["amount"].sum())
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_cat = str(cat_totals.index[0])
    top_amt = float(cat_totals.iloc[0])
    avg_daily = total / max(int(dates.dt.date.nunique()), 1)

    k1, k2, k3, k4 = st.columns(4)
    _render_kpi_card(k1, "Total Expenses", format_inr(total), f"{len(df):,} recorded entries")
    _render_kpi_card(k2, f"Latest ({latest_period.strftime('%b %Y')})", format_inr(latest_spend), "Most recent period")
    _render_kpi_card(k3, "Top Category", top_cat, f"{format_inr(top_amt)} total")
    _render_kpi_card(k4, "Avg Daily Spend", format_inr(avg_daily), "Across active days")

    st.markdown("")
    r1_a, r1_b = st.columns(2)
    r1_a.plotly_chart(category_distribution_chart(df), use_container_width=True, key="dash_donut")
    r1_b.plotly_chart(monthly_trend_chart(df), use_container_width=True, key="dash_trend")

    r2_a, r2_b = st.columns(2)
    r2_a.plotly_chart(top_categories_chart(df), use_container_width=True, key="dash_bar")
    with r2_b:
        m_a_col, m_b_col = st.columns(2)
        month_choices = list(MONTH_NAMES.keys())
        month_a = m_a_col.selectbox(
            "Month A", month_choices, index=9, format_func=lambda m: MONTH_NAMES[m]
        )
        month_b = m_b_col.selectbox(
            "Month B", month_choices, index=10, format_func=lambda m: MONTH_NAMES[m]
        )
        st.plotly_chart(
            monthly_comparison_chart(df, int(month_a), int(month_b)),
            use_container_width=True,
            key="dash_compare",
        )

    st.subheader("Expense Ledger")
    f1, f2, f3 = st.columns([2, 1, 1])
    search_text = f1.text_input("Search description", placeholder="e.g., Swiggy, Uber, electricity")
    selected_cat = f2.selectbox("Category", ["All", *ALLOWED_CATEGORIES])
    sort_by = f3.selectbox("Sort by", ["Date (Newest)", "Date (Oldest)", "Amount (High to Low)", "Amount (Low to High)"])

    filtered = df.copy()
    if selected_cat != "All":
        filtered = filtered[filtered["category"] == selected_cat]
    if search_text.strip():
        filtered = filtered[filtered["description"].str.contains(search_text.strip(), case=False, na=False)]

    if sort_by == "Date (Newest)":
        filtered = filtered.sort_values("date", ascending=False)
    elif sort_by == "Date (Oldest)":
        filtered = filtered.sort_values("date", ascending=True)
    elif sort_by == "Amount (High to Low)":
        filtered = filtered.sort_values("amount", ascending=False)
    else:
        filtered = filtered.sort_values("amount", ascending=True)

    disp = filtered.head(150).copy()
    disp["Date"] = disp["date"].apply(_format_indian_date)
    disp["Amount"] = disp["amount"].apply(lambda x: format_inr(float(x)))
    st.dataframe(
        disp[["expense_id", "Date", "description", "category", "Amount"]].rename(
            columns={"expense_id": "ID", "description": "Description", "category": "Category"}
        ),
        use_container_width=True,
        hide_index=True,
    )

    csv_bytes = filtered[["expense_id", "description", "amount", "date", "category"]].to_csv(index=False).encode("utf-8")
    st.download_button(
        "Export Filtered CSV",
        data=csv_bytes,
        file_name="smart_expense_filtered.csv",
        mime="text/csv",
    )


def _render_add_expense_tab() -> None:
    st.caption("Validated entries are written to `data/runtime/expenses.csv` while `data/expenses.csv` remains read-only.")
    c1, c2 = st.columns(2)
    desc = c1.text_input("Description", placeholder="e.g., Uber ride to office")
    amt = c2.number_input("Amount (₹)", min_value=1.0, value=250.0, step=10.0)

    c3, c4 = st.columns(2)
    exp_date = c3.text_input("Date (YYYY-MM-DD)", value=date.today().isoformat())
    cat_choice = c4.selectbox("Category", ["Auto (ML Classifier)", *ALLOWED_CATEGORIES])

    # Live ML category preview micro-interaction
    if cat_choice == "Auto (ML Classifier)":
        if len(desc.strip()) >= 3:
            try:
                pred = predict(desc.strip())
                st.markdown(
                    f"<div class='ea-preview-pill'>ML Preview: <b>{html.escape(pred['category'])}</b> "
                    f"({pred['confidence']:.1%} confidence)</div>",
                    unsafe_allow_html=True,
                )
            except (ValueError, TypeError):
                pass
        else:
            st.markdown(
                "<div class='ea-preview-pill'>Type at least 3 characters for live ML category preview.</div>",
                unsafe_allow_html=True,
            )

    if st.button("Add Expense", type="primary"):
        cat_arg = None if cat_choice == "Auto (ML Classifier)" else cat_choice
        out = execute_tool_call({
            "function": "add_expense",
            "arguments": {
                "description": desc,
                "amount": amt,
                "expense_date": exp_date,
                "category": cat_arg,
            },
        })
        if out.get("ok"):
            row = out["result"]
            st.success(
                f"Added {row['expense_id']}: {row['description']} — "
                f"{format_inr(float(row['amount']))} ({row['category']}, {row['date']})"
            )
        else:
            st.error(out.get("error", "Could not add expense."))


def main() -> None:
    """Configure page layout and render the Streamlit Smart Expense Assistant."""
    st.set_page_config(
        page_title="Smart Expense Assistant",
        page_icon="₹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    _init_session_state()

    df = _load_df()
    st.markdown(
        f"""
        <div class="ea-topbar">
            <div>
                <h1 class="ea-title">Smart Expense Assistant</h1>
                <p class="ea-subtitle">Conversational expense intelligence, ML categorization & INR (₹) analytics</p>
            </div>
            <div class="ea-status-pill">
                <span class="ea-dot"></span>
                <span>{len(df):,} Expenses · ML Classifier Ready</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_sidebar()

    tab_chat, tab_analytics, tab_add = st.tabs(["Assistant", "Analytics Dashboard", "Add Expense"])
    with tab_chat:
        _render_assistant_tab(df)
    with tab_analytics:
        _render_dashboard_tab(df)
    with tab_add:
        _render_add_expense_tab()


if __name__ == "__main__":
    main()
