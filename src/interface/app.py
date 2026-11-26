"""Conversation-first Streamlit interface inspired by ChatGPT and Claude."""

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

SUGGESTED_PROMPTS = [
    ("Food spending in November", "How much did I spend on food in November?"),
    ("Highest spending category", "Which category has the highest spending?"),
    ("Compare October & November", "Compare my expenses between October and November."),
    ("Categorize: Uber ride ₹250", "Categorize this expense: Uber ride to office, ₹250"),
    ("Show November expenses", "Show all my expenses in November."),
    ("Shopping spending in October", "How much did I spend on shopping in October?"),
]

DESIGN_CSS = """
<style>
    :root {
        --bg-canvas: #FAF9F6;
        --bg-sidebar: #F4F3EE;
        --bg-surface: #FFFFFF;
        --bg-user-bubble: #EFECE6;
        --border-subtle: #E6E4DD;
        --ink-primary: #1A1917;
        --ink-secondary: #5C5952;
        --ink-muted: #848077;
        --accent: #C2593F;
    }

    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", system-ui, sans-serif;
        color: var(--ink-primary);
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background-color: var(--bg-canvas);
    }

    [data-testid="stSidebar"] {
        background-color: var(--bg-sidebar);
        border-right: 1px solid var(--border-subtle);
    }

    .block-container {
        max-width: 820px;
        padding-top: 2rem;
        padding-bottom: 5rem;
    }

    .block-container.wide-mode {
        max-width: 1120px;
    }

    /* Welcome Hero (only visible in empty chat) */
    .welcome-wrap {
        text-align: center;
        padding: 3.2rem 1rem 1.8rem 1rem;
    }
    .welcome-mark {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 38px;
        height: 38px;
        border-radius: 10px;
        background-color: var(--accent);
        color: #FFFFFF;
        font-weight: 600;
        font-size: 1.05rem;
        margin-bottom: 14px;
    }
    .welcome-title {
        font-size: 1.65rem;
        font-weight: 600;
        color: var(--ink-primary);
        letter-spacing: -0.025em;
        margin: 0 0 8px 0;
        line-height: 1.25;
    }
    .welcome-sub {
        font-size: 0.94rem;
        color: var(--ink-secondary);
        margin: 0 auto;
        max-width: 520px;
        line-height: 1.55;
    }

    /* Chat Message Polish */
    [data-testid="stChatMessage"] {
        background-color: transparent;
        padding: 0.85rem 0;
        border-bottom: 1px solid rgba(230, 228, 221, 0.5);
    }

    .user-bubble {
        background-color: var(--bg-user-bubble);
        color: var(--ink-primary);
        border-radius: 14px;
        padding: 11px 15px;
        font-size: 0.95rem;
        line-height: 1.55;
        display: inline-block;
        max-width: 100%;
    }

    .msg-meta {
        font-size: 0.72rem;
        color: var(--ink-muted);
        margin-top: 4px;
    }

    .inline-stat-row {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin: 10px 0 6px 0;
    }

    .inline-stat {
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-radius: 10px;
        padding: 12px 14px;
    }
    .inline-stat-label {
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--ink-secondary);
    }
    .inline-stat-val {
        font-size: 1.18rem;
        font-weight: 600;
        color: var(--ink-primary);
        font-variant-numeric: tabular-nums;
        margin-top: 3px;
    }

    .ml-preview-chip {
        display: inline-block;
        background: var(--bg-surface);
        border: 1px solid var(--border-subtle);
        border-left: 3px solid var(--accent);
        border-radius: 8px;
        padding: 7px 12px;
        font-size: 0.83rem;
        color: var(--ink-primary);
        margin: 6px 0 12px 0;
    }

    /* Restrained buttons & controls */
    .stButton > button {
        background-color: var(--bg-surface);
        color: var(--ink-primary);
        border: 1px solid var(--border-subtle);
        border-radius: 9px;
        font-size: 0.84rem;
        font-weight: 500;
        padding: 0.45rem 0.85rem;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover {
        background-color: var(--bg-user-bubble);
        border-color: var(--ink-secondary);
        color: var(--ink-primary);
    }
    .stButton > button[kind="primary"] {
        background-color: var(--ink-primary);
        color: #FFFFFF;
        border-color: var(--ink-primary);
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #33312D;
        border-color: #33312D;
        color: #FFFFFF;
    }
    .stButton > button:focus-visible {
        outline: 2px solid var(--accent);
        outline-offset: 2px;
    }

    /* Composer polish */
    [data-testid="stChatInput"] textarea {
        font-size: 0.95rem !important;
        line-height: 1.5 !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-left: 0.85rem;
            padding-right: 0.85rem;
            padding-top: 1.2rem;
        }
        .welcome-title {
            font-size: 1.35rem;
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


def _make_chat_title(prompt: str) -> str:
    cleaned = " ".join(prompt.strip().split())
    return cleaned[:34] + ("…" if len(cleaned) > 34 else "")


def _init_state() -> None:
    if "chats" not in st.session_state:
        st.session_state.chats = {"c1": {"title": "New conversation", "messages": []}}
    if "active_chat_id" not in st.session_state:
        st.session_state.active_chat_id = "c1"
    if "nav_page" not in st.session_state:
        st.session_state.nav_page = "Chat"
    if "queued_prompt" not in st.session_state:
        st.session_state.queued_prompt = None
    if "queued_qwen_json" not in st.session_state:
        st.session_state.queued_qwen_json = None
    if "copy_index" not in st.session_state:
        st.session_state.copy_index = None


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### Smart Expense")
        if st.button("+ New chat", type="primary", width="stretch"):
            new_id = f"c{len(st.session_state.chats) + 1}"
            st.session_state.chats[new_id] = {"title": "New conversation", "messages": []}
            st.session_state.active_chat_id = new_id
            st.session_state.nav_page = "Chat"
            st.rerun()

        st.markdown("")
        nav = st.radio(
            "Workspace",
            ["Chat", "Analytics", "Add Expense"],
            index=["Chat", "Analytics", "Add Expense"].index(st.session_state.nav_page),
            label_visibility="collapsed",
        )
        st.session_state.nav_page = nav

        st.divider()
        st.caption("Conversations")
        chat_ids = list(reversed(list(st.session_state.chats.keys())))
        for cid in chat_ids:
            title = st.session_state.chats[cid]["title"]
            prefix = "● " if cid == st.session_state.active_chat_id else ""
            if st.button(f"{prefix}{title}", key=f"nav_chat_{cid}", width="stretch"):
                st.session_state.active_chat_id = cid
                st.session_state.nav_page = "Chat"
                st.rerun()

        if st.button("Clear current chat", width="stretch"):
            st.session_state.chats[st.session_state.active_chat_id]["messages"] = []
            st.session_state.chats[st.session_state.active_chat_id]["title"] = "New conversation"
            st.rerun()

        st.divider()
        mode_text = "Qwen2.5 Handler Active" if _LLM_HANDLER is not None else "Backend + Offline Router"
        st.caption(f"Engine: {mode_text}")

        with st.expander("Developer: Qwen JSON"):
            raw_json = st.text_area(
                "Paste Qwen2.5 JSON",
                value='{\n  "intent": "calculate_spending",\n  "category": "Food",\n  "month": "November",\n  "amount": null,\n  "description": null\n}',
                height=145,
            )
            if st.button("Run Qwen JSON", width="stretch"):
                st.session_state.queued_qwen_json = raw_json
                st.session_state.nav_page = "Chat"
                st.rerun()


def _render_inline_visual(out: dict[str, Any], df: pd.DataFrame, key_prefix: str) -> None:
    if not out.get("ok"):
        return
    fn = out.get("function")
    res = out.get("result")

    if fn == "categorize_expense" and isinstance(res, dict):
        conf = float(res.get("confidence", 0.0))
        st.markdown(
            f"""
            <div class="inline-stat-row">
                <div class="inline-stat">
                    <div class="inline-stat-label">Category</div>
                    <div class="inline-stat-val">{html.escape(str(res.get("category", "")))}</div>
                </div>
                <div class="inline-stat">
                    <div class="inline-stat-label">Confidence</div>
                    <div class="inline-stat-val">{conf:.1%}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif fn == "get_highest_category" and isinstance(res, dict) and res.get("category"):
        st.markdown(
            f"""
            <div class="inline-stat-row">
                <div class="inline-stat">
                    <div class="inline-stat-label">Top Category</div>
                    <div class="inline-stat-val">{html.escape(str(res["category"]))}</div>
                </div>
                <div class="inline-stat">
                    <div class="inline-stat-label">Total Spend</div>
                    <div class="inline-stat-val">{format_inr(float(res["amount"]))}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    elif fn == "get_category_breakdown" and isinstance(res, dict) and res:
        st.plotly_chart(
            breakdown_dict_chart(res, "Category Breakdown"),
            width="stretch",
            key=f"{key_prefix}_donut",
        )

    elif fn == "compare_months" and isinstance(res, dict):
        st.markdown(
            f"""
            <div class="inline-stat-row">
                <div class="inline-stat">
                    <div class="inline-stat-label">Month {res['month_a']}</div>
                    <div class="inline-stat-val">{format_inr(float(res['total_a']))}</div>
                </div>
                <div class="inline-stat">
                    <div class="inline-stat-label">Month {res['month_b']}</div>
                    <div class="inline-stat-val">{format_inr(float(res['total_b']))}</div>
                </div>
                <div class="inline-stat">
                    <div class="inline-stat-label">Change</div>
                    <div class="inline-stat-val">{float(res['percent_change']):+.1f}%</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.plotly_chart(
            monthly_comparison_chart(df, int(res["month_a"]), int(res["month_b"])),
            width="stretch",
            key=f"{key_prefix}_cmp",
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
                width="stretch",
                hide_index=True,
            )
        else:
            st.info("No matching expense records found.")


def _render_chat_page(df: pd.DataFrame) -> None:
    chat = st.session_state.chats[st.session_state.active_chat_id]
    messages: list[dict[str, Any]] = chat["messages"]

    # Calm empty welcome state (disappears once conversation starts)
    if not messages and not st.session_state.queued_prompt and not st.session_state.queued_qwen_json:
        st.markdown(
            """
            <div class="welcome-wrap">
                <div class="welcome-mark">₹</div>
                <h1 class="welcome-title">How can I help with your expenses?</h1>
                <p class="welcome-sub">
                    Ask in plain English for monthly totals, category comparisons, transaction history,
                    or instant ML categorization.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        for idx, (label, prompt_text) in enumerate(SUGGESTED_PROMPTS):
            col = c1 if idx % 2 == 0 else c2
            if col.button(label, key=f"welcome_prompt_{idx}", width="stretch"):
                st.session_state.queued_prompt = prompt_text
                st.rerun()

    # Render existing conversation turns
    for idx, msg in enumerate(messages):
        if msg["role"] == "user":
            with st.chat_message("user"):
                st.markdown(
                    f"<div class='user-bubble'>{html.escape(msg['content'])}</div>"
                    f"<div class='msg-meta'>{html.escape(msg.get('time', ''))}</div>",
                    unsafe_allow_html=True,
                )
        else:
            with st.chat_message("assistant"):
                out = msg.get("output", {})
                if out and not out.get("ok"):
                    st.error(msg["content"])
                    if st.button("Retry", key=f"retry_{idx}") and idx > 0:
                        st.session_state.queued_prompt = messages[idx - 1]["content"]
                        st.rerun()
                else:
                    st.markdown(msg["content"])
                    _render_inline_visual(out, df, key_prefix=f"msg_{idx}")

                    action_col1, action_col2 = st.columns([1, 5])
                    if action_col1.button("Copy", key=f"copy_btn_{idx}"):
                        st.session_state.copy_index = None if st.session_state.copy_index == idx else idx
                    action_col2.markdown(
                        f"<div class='msg-meta'>{html.escape(msg.get('time', ''))}</div>",
                        unsafe_allow_html=True,
                    )
                    if st.session_state.copy_index == idx:
                        st.code(msg["content"], language="markdown")

    # Comfortable composer (Enter to send, Shift+Enter for new line)
    typed = st.chat_input("Message Smart Expense Assistant…")
    qwen_raw = st.session_state.queued_qwen_json
    st.session_state.queued_qwen_json = None

    queued = st.session_state.queued_prompt
    st.session_state.queued_prompt = None

    user_input = typed or queued
    now_str = datetime.now().strftime("%I:%M %p").lstrip("0")

    if qwen_raw:
        label = f"Qwen JSON payload"
        if chat["title"] == "New conversation":
            chat["title"] = "Qwen JSON test"
        messages.append({"role": "user", "content": qwen_raw.strip(), "time": now_str})
        with st.chat_message("user"):
            st.code(qwen_raw.strip(), language="json")
        with st.chat_message("assistant"):
            with st.spinner("Running Qwen JSON via backend…"):
                out = process_query("", llm_json=qwen_raw)
            if not out.get("ok"):
                st.error(out["answer"])
            else:
                st.markdown(out["answer"])
                _render_inline_visual(out, df, key_prefix=f"live_{len(messages)}")
        messages.append({"role": "assistant", "content": out["answer"], "output": out, "time": now_str})

    elif user_input:
        if chat["title"] == "New conversation":
            chat["title"] = _make_chat_title(user_input)
        messages.append({"role": "user", "content": user_input, "time": now_str})
        with st.chat_message("user"):
            st.markdown(
                f"<div class='user-bubble'>{html.escape(user_input)}</div>"
                f"<div class='msg-meta'>{now_str}</div>",
                unsafe_allow_html=True,
            )

        with st.chat_message("assistant"):
            stop_placeholder = st.empty()
            if stop_placeholder.button("Stop generating", key=f"stop_{len(messages)}"):
                st.stop()
            with st.spinner("Thinking…"):
                out = process_query(user_input)
            stop_placeholder.empty()

            if not out.get("ok"):
                st.error(out["answer"])
            else:
                st.markdown(out["answer"])
                _render_inline_visual(out, df, key_prefix=f"live_{len(messages)}")
                st.markdown(f"<div class='msg-meta'>{now_str}</div>", unsafe_allow_html=True)

        messages.append({"role": "assistant", "content": out["answer"], "output": out, "time": now_str})


def _render_analytics_page(df: pd.DataFrame) -> None:
    st.markdown("## Expense Analytics")
    st.caption("Interactive visual breakdown of all recorded INR (₹) transactions.")

    if df.empty:
        st.info("No expense records available.")
        return

    total = float(df["amount"].sum())
    dates = pd.to_datetime(df["date"], errors="coerce")
    latest_period = dates.dt.to_period("M").max()
    latest_spend = float(df[dates.dt.to_period("M") == latest_period]["amount"].sum())
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_cat = str(cat_totals.index[0])
    top_amt = float(cat_totals.iloc[0])
    avg_daily = total / max(int(dates.dt.date.nunique()), 1)

    st.markdown(
        f"""
        <div class="inline-stat-row">
            <div class="inline-stat">
                <div class="inline-stat-label">Total Expenses</div>
                <div class="inline-stat-val">{format_inr(total)}</div>
            </div>
            <div class="inline-stat">
                <div class="inline-stat-label">Latest ({latest_period.strftime('%b %Y')})</div>
                <div class="inline-stat-val">{format_inr(latest_spend)}</div>
            </div>
            <div class="inline-stat">
                <div class="inline-stat-label">Top Category</div>
                <div class="inline-stat-val">{html.escape(top_cat)} ({format_inr(top_amt)})</div>
            </div>
            <div class="inline-stat">
                <div class="inline-stat-label">Avg Daily Spend</div>
                <div class="inline-stat-val">{format_inr(avg_daily)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    c1.plotly_chart(category_distribution_chart(df), width="stretch", key="an_donut")
    c2.plotly_chart(monthly_trend_chart(df), width="stretch", key="an_trend")

    c3, c4 = st.columns(2)
    c3.plotly_chart(top_categories_chart(df), width="stretch", key="an_bar")
    with c4:
        ma_col, mb_col = st.columns(2)
        month_choices = list(MONTH_NAMES.keys())
        month_a = ma_col.selectbox("Month A", month_choices, index=9, format_func=lambda m: MONTH_NAMES[m])
        month_b = mb_col.selectbox("Month B", month_choices, index=10, format_func=lambda m: MONTH_NAMES[m])
        st.plotly_chart(monthly_comparison_chart(df, int(month_a), int(month_b)), width="stretch", key="an_cmp")

    st.markdown("### Ledger")
    f1, f2, f3 = st.columns([2, 1, 1])
    search_text = f1.text_input("Search description", placeholder="Swiggy, Uber, electricity…")
    selected_cat = f2.selectbox("Category", ["All", *ALLOWED_CATEGORIES])
    sort_by = f3.selectbox("Sort", ["Date (Newest)", "Date (Oldest)", "Amount (High → Low)", "Amount (Low → High)"])

    filtered = df.copy()
    if selected_cat != "All":
        filtered = filtered[filtered["category"] == selected_cat]
    if search_text.strip():
        filtered = filtered[filtered["description"].str.contains(search_text.strip(), case=False, na=False)]

    if sort_by == "Date (Newest)":
        filtered = filtered.sort_values("date", ascending=False)
    elif sort_by == "Date (Oldest)":
        filtered = filtered.sort_values("date", ascending=True)
    elif sort_by == "Amount (High → Low)":
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
        width="stretch",
        hide_index=True,
    )

    csv_bytes = filtered[["expense_id", "description", "amount", "date", "category"]].to_csv(index=False).encode("utf-8")
    st.download_button("Export CSV", data=csv_bytes, file_name="smart_expense_filtered.csv", mime="text/csv")


def _render_add_expense_page() -> None:
    st.markdown("## Add Expense")
    st.caption("New entries are validated and saved to `data/runtime/expenses.csv` (seed data remains read-only).")

    c1, c2 = st.columns(2)
    desc = c1.text_input("Description", placeholder="e.g., Uber ride to office")
    amt = c2.number_input("Amount (₹)", min_value=1.0, value=250.0, step=10.0)

    c3, c4 = st.columns(2)
    exp_date = c3.text_input("Date (YYYY-MM-DD)", value=date.today().isoformat())
    cat_choice = c4.selectbox("Category", ["Auto (ML Classifier)", *ALLOWED_CATEGORIES])

    if cat_choice == "Auto (ML Classifier)":
        if len(desc.strip()) >= 3:
            try:
                pred = predict(desc.strip())
                st.markdown(
                    f"<div class='ml-preview-chip'>ML Prediction: <b>{html.escape(pred['category'])}</b> "
                    f"({pred['confidence']:.1%} confidence)</div>",
                    unsafe_allow_html=True,
                )
            except (ValueError, TypeError):
                pass
        else:
            st.markdown(
                "<div class='ml-preview-chip'>Type a description to preview ML categorization.</div>",
                unsafe_allow_html=True,
            )

    if st.button("Save Expense", type="primary"):
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
                f"Saved {row['expense_id']}: {row['description']} — "
                f"{format_inr(float(row['amount']))} ({row['category']}, {row['date']})"
            )
        else:
            st.error(out.get("error", "Could not save expense."))


def main() -> None:
    """Run the conversation-first Smart Expense Assistant application."""
    st.set_page_config(
        page_title="Smart Expense Assistant",
        page_icon="₹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _init_state()
    st.markdown(DESIGN_CSS, unsafe_allow_html=True)
    _render_sidebar()

    df = _load_df()
    page = st.session_state.nav_page
    if page == "Chat":
        _render_chat_page(df)
    elif page == "Analytics":
        _render_analytics_page(df)
    else:
        _render_add_expense_page()


if __name__ == "__main__":
    main()
