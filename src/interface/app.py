"""Apple-level minimal Gradio interface for Smart Expense Assistant."""

from __future__ import annotations

import html
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd

from src.backend.router import execute_tool_call, process_query
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
from src.llm.respond import format_inr, generate_response

SAMPLE_QUERIES = [
    ("Food in November", "How much did I spend on food in November?"),
    ("Highest Category", "Which category has the highest spending?"),
    ("Oct vs Nov", "Compare my expenses between October and November."),
    ("Categorize ₹250 Uber", "Categorize this expense: Uber ride to office, ₹250"),
    ("All Nov Expenses", "Show all my expenses in November."),
    ("Shopping in Oct", "How much did I spend on shopping in October?"),
]

CUSTOM_CSS = """
@keyframes eaFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding: 8px 20px 28px 20px !important;
    background-color: #F5F5F7 !important;
    color: #1D1D1F !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
    color-scheme: light !important;
}

footer { display: none !important; }

.ea-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 14px 0 14px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    margin-bottom: 14px;
}

.ea-title {
    font-size: 1.38rem;
    font-weight: 600;
    color: #1D1D1F;
    letter-spacing: -0.022em;
    margin: 0;
    line-height: 1.2;
}

.ea-subtitle {
    font-size: 0.85rem;
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

.ea-card {
    background: #FFFFFF;
    border: 1px solid rgba(0, 0, 0, 0.07);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    animation: eaFadeIn 0.2s ease-out;
}

.ea-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 4px 0 16px 0;
    animation: eaFadeIn 0.22s ease-out;
}

.ea-kpi-card {
    background: #FFFFFF;
    border: 1px solid rgba(0, 0, 0, 0.07);
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    transition: transform 0.18s cubic-bezier(0.25, 0.1, 0.25, 1), box-shadow 0.18s ease;
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
    margin-bottom: 6px;
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

.ea-feedback {
    background: #FFFFFF;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-left: 3px solid #0F766E;
    border-radius: 8px;
    padding: 9px 13px;
    font-size: 0.83rem;
    color: #1D1D1F;
    animation: eaFadeIn 0.18s ease-out;
}

.ea-feedback.error {
    background: #FEF2F2;
    border-color: #FECACA;
    border-left-color: #B91C1C;
    color: #991B1B;
}

.ea-preview-pill {
    display: inline-block;
    background: #F2F2F7;
    border: 1px solid rgba(0, 0, 0, 0.07);
    border-radius: 8px;
    padding: 7px 12px;
    font-size: 0.81rem;
    color: #1D1D1F;
}

.tab-nav button {
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: #6E6E73 !important;
    padding: 8px 14px !important;
    transition: color 0.15s ease !important;
}

.tab-nav button.selected {
    color: #1D1D1F !important;
    font-weight: 600 !important;
    border-bottom: 2px solid #0066CC !important;
}

button {
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: transform 0.12s ease, background-color 0.15s ease, border-color 0.15s ease !important;
}

button:active {
    transform: scale(0.985) !important;
}

button:focus-visible, input:focus-visible, textarea:focus-visible, select:focus-visible {
    outline: 2px solid #0066CC !important;
    outline-offset: 2px !important;
}

@media (max-width: 768px) {
    .gradio-container {
        padding: 6px 12px 20px 12px !important;
    }
    .ea-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
"""

DEFAULT_INSPECTOR_HTML = """
<div class="ea-card">
    <div class="ea-kpi-label">Assistant Workspace</div>
    <div style="font-size:0.95rem;font-weight:600;color:#1D1D1F;margin-bottom:8px;">
        Ready for Expense Queries
    </div>
    <p style="font-size:0.83rem;color:#6E6E73;line-height:1.5;margin:0 0 10px 0;">
        Ask a natural-language question or select a quick prompt. Matching charts, category breakdowns,
        and transaction tables will appear here automatically.
    </p>
    <div style="font-size:0.78rem;color:#6E6E73;border-top:1px solid #F2F2F7;padding-top:10px;line-height:1.6;">
        <b>Coverage:</b> Jan 2025 – Mar 2026<br>
        <b>Categories:</b> Food · Travel · Shopping · Utilities<br>
        <b>Currency:</b> Indian Rupees (₹)
    </div>
</div>
"""


def _load_df() -> pd.DataFrame:
    records = get_expenses()
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["expense_id", "description", "amount", "date", "category"]
    )


def _build_kpi_html(df: pd.DataFrame) -> str:
    if df.empty:
        return "<div class='ea-kpi-card'>No expense records available.</div>"
    total = float(df["amount"].sum())
    dates = pd.to_datetime(df["date"], errors="coerce")
    latest_period = dates.dt.to_period("M").max()
    latest_spend = float(df[dates.dt.to_period("M") == latest_period]["amount"].sum())
    cat_totals = df.groupby("category")["amount"].sum().sort_values(ascending=False)
    top_cat = str(cat_totals.index[0])
    top_amt = float(cat_totals.iloc[0])
    avg_daily = total / max(int(dates.dt.date.nunique()), 1)

    return f"""
    <div class="ea-kpi-grid">
        <div class="ea-kpi-card">
            <div class="ea-kpi-label">Total Expenses</div>
            <div class="ea-kpi-value">{format_inr(total)}</div>
            <div class="ea-kpi-sub">{len(df):,} recorded entries</div>
        </div>
        <div class="ea-kpi-card">
            <div class="ea-kpi-label">Latest Month ({latest_period.strftime('%b %Y')})</div>
            <div class="ea-kpi-value">{format_inr(latest_spend)}</div>
            <div class="ea-kpi-sub">Most recent period</div>
        </div>
        <div class="ea-kpi-card">
            <div class="ea-kpi-label">Top Category</div>
            <div class="ea-kpi-value">{html.escape(top_cat)}</div>
            <div class="ea-kpi-sub">{format_inr(top_amt)} total</div>
        </div>
        <div class="ea-kpi-card">
            <div class="ea-kpi-label">Average Daily Spend</div>
            <div class="ea-kpi-value">{format_inr(avg_daily)}</div>
            <div class="ea-kpi-sub">Across active days</div>
        </div>
    </div>
    """


def _build_inspector_card(fn: str, res: Any) -> str:
    if fn == "calculate_spending":
        return f"""
        <div class="ea-card">
            <div class="ea-kpi-label">Calculated Total</div>
            <div class="ea-kpi-value">{format_inr(float(res))}</div>
            <div class="ea-kpi-sub">Computed directly from verified expense records</div>
        </div>
        """
    if fn == "get_highest_category" and isinstance(res, dict):
        cat = html.escape(str(res.get("category") or "N/A"))
        amt = format_inr(float(res.get("amount", 0.0)))
        return f"""
        <div class="ea-card">
            <div class="ea-kpi-label">Highest Spending Category</div>
            <div class="ea-kpi-value">{cat}</div>
            <div class="ea-kpi-sub">Total spend: {amt}</div>
        </div>
        """
    if fn == "categorize_expense" and isinstance(res, dict):
        cat = html.escape(str(res.get("category", "")))
        desc = html.escape(str(res.get("description", "")))
        conf = float(res.get("confidence", 0.0))
        return f"""
        <div class="ea-card">
            <div class="ea-kpi-label">ML Classification Result</div>
            <div class="ea-kpi-value">{cat}</div>
            <div class="ea-kpi-sub">Description: “{desc}” · Confidence: {conf:.1%}</div>
        </div>
        """
    if fn == "compare_months" and isinstance(res, dict):
        diff = float(res.get("difference", 0.0))
        pct = float(res.get("percent_change", 0.0))
        sign = "+" if diff >= 0 else "-"
        return f"""
        <div class="ea-card">
            <div class="ea-kpi-label">Monthly Delta</div>
            <div class="ea-kpi-value">{sign}{format_inr(abs(diff))}</div>
            <div class="ea-kpi-sub">{pct:+.1f}% change between Month {res['month_a']} and Month {res['month_b']}</div>
        </div>
        """
    return DEFAULT_INSPECTOR_HTML


def _format_chat_outcome(
    user_label: str,
    tool_output: dict[str, Any],
    chat_history: list[dict[str, str]],
) -> tuple[str, list[dict[str, str]], Any, Any, str, str, str]:
    """Update chat history, progressive disclosure table/chart, and inspector card."""
    empty_df = pd.DataFrame(columns=["expense_id", "description", "amount", "date", "category"])
    answer = str(tool_output.get("answer") or generate_response(tool_output))
    chat_history.append({"role": "user", "content": user_label})
    chat_history.append({"role": "assistant", "content": answer})

    if not tool_output.get("ok"):
        feedback = f"<div class='ea-feedback error'>{html.escape(answer)}</div>"
        return (
            "",
            chat_history,
            gr.update(visible=False, value=empty_df),
            gr.update(visible=False, value=None),
            feedback,
            user_label,
            DEFAULT_INSPECTOR_HTML,
        )

    fn = str(tool_output.get("function"))
    res = tool_output.get("result")
    feedback = f"<div class='ea-feedback'>Completed <b>{html.escape(fn)}</b>.</div>"
    inspector_html = _build_inspector_card(fn, res)

    table_update: Any = gr.update(visible=False, value=empty_df)
    chart_update: Any = gr.update(visible=False, value=None)

    if fn == "get_expenses" and isinstance(res, list) and res:
        table_update = gr.update(visible=True, value=pd.DataFrame(res))
        inspector_html = (
            f"<div class='ea-card'><div class='ea-kpi-label'>Matching Records</div>"
            f"<div class='ea-kpi-value'>{len(res)} entries</div>"
            f"<div class='ea-kpi-sub'>Sorted newest first</div></div>"
        )
    elif fn == "get_category_breakdown" and isinstance(res, dict):
        chart_update = gr.update(visible=True, value=breakdown_dict_chart(res, "Category Breakdown"))
    elif fn == "compare_months" and isinstance(res, dict):
        chart_update = gr.update(
            visible=True,
            value=monthly_comparison_chart(_load_df(), int(res["month_a"]), int(res["month_b"])),
        )

    return "", chat_history, table_update, chart_update, feedback, user_label, inspector_html


def handle_chat_query(
    user_message: str,
    history: list[dict[str, str]] | None,
) -> tuple[str, list[dict[str, str]], Any, Any, str, str, str]:
    """Send user query strictly to Python backend (process_query) and render results."""
    chat_history = list(history or [])
    empty_df = pd.DataFrame(columns=["expense_id", "description", "amount", "date", "category"])

    if not isinstance(user_message, str) or not user_message.strip():
        feedback = "<div class='ea-feedback error'>Please enter a question or expense description.</div>"
        return (
            "",
            chat_history,
            gr.update(visible=False, value=empty_df),
            gr.update(visible=False, value=None),
            feedback,
            "",
            DEFAULT_INSPECTOR_HTML,
        )

    query = user_message.strip()
    tool_output = process_query(query)
    return _format_chat_outcome(query, tool_output, chat_history)


def handle_qwen_json_test(
    raw_json: str,
    history: list[dict[str, str]] | None,
) -> tuple[str, list[dict[str, str]], Any, Any, str, str, str]:
    """Execute a raw Qwen2.5 JSON payload directly through the Python backend."""
    chat_history = list(history or [])
    tool_output = process_query("", llm_json=raw_json)
    _, updated_hist, tbl, chrt, fb, _, insp = _format_chat_outcome(
        f"Qwen JSON: {raw_json.strip()}", tool_output, chat_history
    )
    return raw_json, updated_hist, tbl, chrt, fb, "", insp


def filter_ledger(category: str, search_text: str) -> pd.DataFrame:
    """Filter expense ledger by category and description substring."""
    cat_arg = None if category == "All" else category
    df = pd.DataFrame(get_expenses(category=cat_arg))
    if df.empty:
        return df
    if search_text and search_text.strip():
        df = df[df["description"].str.contains(search_text.strip(), case=False, na=False)]
    return df.head(150)


def export_filtered_csv(category: str, search_text: str) -> str:
    """Export filtered ledger rows to a temporary CSV for download."""
    df = filter_ledger(category, search_text)
    tmp_path = Path(tempfile.gettempdir()) / "smart_expense_filtered.csv"
    df.to_csv(tmp_path, index=False)
    return str(tmp_path)


def preview_ml_category(description: str, selected_category: str) -> str:
    """Provide instant ML category preview while typing an expense description."""
    if selected_category != "Auto (ML Classifier)":
        return f"<div class='ea-preview-pill'>Manual Category: <b>{html.escape(selected_category)}</b></div>"
    if not isinstance(description, str) or len(description.strip()) < 3:
        return "<div class='ea-preview-pill'>Type at least 3 characters for live ML category preview.</div>"
    try:
        res = predict(description.strip())
        return (
            f"<div class='ea-preview-pill'>ML Preview: <b>{html.escape(res['category'])}</b> "
            f"({res['confidence']:.1%} confidence)</div>"
        )
    except (ValueError, TypeError):
        return "<div class='ea-preview-pill'>ML Preview unavailable for this input.</div>"


def handle_add_expense(
    description: str,
    amount: float,
    expense_date: str,
    category: str,
) -> tuple[str, str, Any, Any, Any, Any, pd.DataFrame]:
    """Validate and add an expense via backend router, then refresh dashboard."""
    cat_arg = None if category == "Auto (ML Classifier)" else category
    tool_output = execute_tool_call({
        "function": "add_expense",
        "arguments": {
            "description": description,
            "amount": amount,
            "expense_date": expense_date,
            "category": cat_arg,
        },
    })
    msg = generate_response(tool_output)
    css_cls = "ea-feedback" if tool_output.get("ok") else "ea-feedback error"
    status_html = f"<div class='{css_cls}'>{html.escape(msg)}</div>"
    df = _load_df()
    return (
        status_html,
        _build_kpi_html(df),
        category_distribution_chart(df),
        monthly_trend_chart(df),
        top_categories_chart(df),
        monthly_comparison_chart(df, 10, 11),
        df.head(150),
    )


def create_app() -> gr.Blocks:
    """Build and return the Gradio Blocks application."""
    df_init = _load_df()

    with gr.Blocks(title="Smart Expense Assistant") as demo:
        last_query_state = gr.State("")

        gr.HTML(
            f"""
            <div class="ea-topbar">
                <div>
                    <h1 class="ea-title">Smart Expense Assistant</h1>
                    <p class="ea-subtitle">Conversational expense intelligence, ML categorization & INR (₹) analytics</p>
                </div>
                <div class="ea-status-pill">
                    <span class="ea-dot"></span>
                    <span>{len(df_init):,} Expenses · ML Classifier Ready</span>
                </div>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Assistant"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=440,
                            placeholder="Ask a question about your expenses or pick a quick prompt below.",
                        )
                        feedback_html = gr.HTML("")
                        with gr.Row():
                            query_box = gr.Textbox(
                                placeholder="Ask anything (e.g. 'How much did I spend on food in November?')",
                                show_label=False,
                                scale=6,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                            retry_btn = gr.Button("Retry", scale=1)
                            clear_btn = gr.Button("Clear", scale=1)

                        gr.Markdown("**Quick Prompts**")
                        sample_btns: list[tuple[str, gr.Button]] = []
                        with gr.Row():
                            for label, full_q in SAMPLE_QUERIES[:3]:
                                sample_btns.append((full_q, gr.Button(label, size="sm")))
                        with gr.Row():
                            for label, full_q in SAMPLE_QUERIES[3:]:
                                sample_btns.append((full_q, gr.Button(label, size="sm")))

                        with gr.Accordion("Integration Tester: Run Raw Qwen2.5 JSON", open=False):
                            qwen_json_box = gr.Textbox(
                                label="Qwen JSON Output",
                                lines=6,
                                value='{\n    "intent": "calculate_spending",\n    "category": "Food",\n    "month": "November",\n    "amount": null,\n    "description": null\n}',
                            )
                            qwen_run_btn = gr.Button("Run Qwen JSON via Backend", size="sm")

                    with gr.Column(scale=2):
                        inspector_html = gr.HTML(DEFAULT_INSPECTOR_HTML)
                        chat_chart = gr.Plot(label="Visual Breakdown", visible=False)
                        chat_table = gr.Dataframe(
                            label="Matching Transactions",
                            interactive=False,
                            wrap=True,
                            visible=False,
                        )

                outputs_chat = [
                    query_box,
                    chatbot,
                    chat_table,
                    chat_chart,
                    feedback_html,
                    last_query_state,
                    inspector_html,
                ]
                send_btn.click(
                    fn=handle_chat_query,
                    inputs=[query_box, chatbot],
                    outputs=outputs_chat,
                    show_progress="minimal",
                )
                query_box.submit(
                    fn=handle_chat_query,
                    inputs=[query_box, chatbot],
                    outputs=outputs_chat,
                    show_progress="minimal",
                )
                retry_btn.click(
                    fn=handle_chat_query,
                    inputs=[last_query_state, chatbot],
                    outputs=outputs_chat,
                    show_progress="minimal",
                )
                clear_btn.click(
                    fn=lambda: (
                        [],
                        gr.update(visible=False, value=pd.DataFrame()),
                        gr.update(visible=False, value=None),
                        "",
                        "",
                        DEFAULT_INSPECTOR_HTML,
                    ),
                    outputs=[chatbot, chat_table, chat_chart, feedback_html, last_query_state, inspector_html],
                    show_progress="hidden",
                )
                for q_text, q_btn in sample_btns:
                    q_btn.click(
                        fn=lambda h, text=q_text: handle_chat_query(text, h),
                        inputs=[chatbot],
                        outputs=outputs_chat,
                        show_progress="minimal",
                    )
                qwen_run_btn.click(
                    fn=handle_qwen_json_test,
                    inputs=[qwen_json_box, chatbot],
                    outputs=[qwen_json_box, chatbot, chat_table, chat_chart, feedback_html, last_query_state, inspector_html],
                    show_progress="minimal",
                )

            with gr.Tab("Analytics Dashboard"):
                kpi_html = gr.HTML(_build_kpi_html(df_init))
                with gr.Row():
                    plot_donut = gr.Plot(category_distribution_chart(df_init))
                    plot_trend = gr.Plot(monthly_trend_chart(df_init))
                with gr.Row():
                    plot_bar = gr.Plot(top_categories_chart(df_init))
                    with gr.Column():
                        with gr.Row():
                            month_a_dd = gr.Dropdown(
                                choices=[(v, k) for k, v in MONTH_NAMES.items()],
                                value=10,
                                label="Month A",
                            )
                            month_b_dd = gr.Dropdown(
                                choices=[(v, k) for k, v in MONTH_NAMES.items()],
                                value=11,
                                label="Month B",
                            )
                        plot_comp = gr.Plot(monthly_comparison_chart(df_init, 10, 11))

                month_a_dd.change(
                    fn=lambda a, b: monthly_comparison_chart(_load_df(), int(a), int(b)),
                    inputs=[month_a_dd, month_b_dd],
                    outputs=[plot_comp],
                    show_progress="minimal",
                )
                month_b_dd.change(
                    fn=lambda a, b: monthly_comparison_chart(_load_df(), int(a), int(b)),
                    inputs=[month_a_dd, month_b_dd],
                    outputs=[plot_comp],
                    show_progress="minimal",
                )

                gr.Markdown("### Expense Ledger")
                with gr.Row():
                    cat_filter = gr.Dropdown(
                        choices=["All", *ALLOWED_CATEGORIES],
                        value="All",
                        label="Category",
                        scale=2,
                    )
                    search_filter = gr.Textbox(
                        placeholder="Search description (e.g. Swiggy, Uber, electricity)...",
                        label="Search",
                        scale=4,
                    )
                    export_btn = gr.DownloadButton("Export CSV", scale=1)

                ledger_table = gr.Dataframe(value=df_init.head(150), interactive=False, wrap=True)
                cat_filter.change(
                    fn=filter_ledger,
                    inputs=[cat_filter, search_filter],
                    outputs=[ledger_table],
                    show_progress="hidden",
                )
                search_filter.change(
                    fn=filter_ledger,
                    inputs=[cat_filter, search_filter],
                    outputs=[ledger_table],
                    show_progress="hidden",
                )
                export_btn.click(
                    fn=export_filtered_csv,
                    inputs=[cat_filter, search_filter],
                    outputs=[export_btn],
                )

            with gr.Tab("Add Expense"):
                gr.Markdown("Record a new expense into `data/runtime/expenses.csv` (seed data remains read-only).")
                with gr.Row():
                    add_desc = gr.Textbox(label="Description", placeholder="e.g. Uber ride to office")
                    add_amt = gr.Number(label="Amount (₹)", value=250.0, minimum=1.0)
                with gr.Row():
                    add_date = gr.Textbox(label="Date (YYYY-MM-DD)", value=date.today().isoformat())
                    add_cat = gr.Dropdown(
                        choices=["Auto (ML Classifier)", *ALLOWED_CATEGORIES],
                        value="Auto (ML Classifier)",
                        label="Category",
                    )
                preview_html = gr.HTML(preview_ml_category("", "Auto (ML Classifier)"))
                with gr.Row():
                    add_btn = gr.Button("Add Expense", variant="primary")
                    reset_btn = gr.Button("Reset Form")
                add_status = gr.HTML()

                add_desc.change(
                    fn=preview_ml_category,
                    inputs=[add_desc, add_cat],
                    outputs=[preview_html],
                    show_progress="hidden",
                )
                add_cat.change(
                    fn=preview_ml_category,
                    inputs=[add_desc, add_cat],
                    outputs=[preview_html],
                    show_progress="hidden",
                )
                add_btn.click(
                    fn=handle_add_expense,
                    inputs=[add_desc, add_amt, add_date, add_cat],
                    outputs=[
                        add_status,
                        kpi_html,
                        plot_donut,
                        plot_trend,
                        plot_bar,
                        plot_comp,
                        ledger_table,
                    ],
                    show_progress="minimal",
                )
                reset_btn.click(
                    fn=lambda: (
                        "",
                        250.0,
                        date.today().isoformat(),
                        "Auto (ML Classifier)",
                        "",
                        preview_ml_category("", "Auto (ML Classifier)"),
                    ),
                    outputs=[add_desc, add_amt, add_date, add_cat, add_status, preview_html],
                    show_progress="hidden",
                )

    return demo


def launch_app(share: bool = False) -> None:
    """Launch the Gradio application (set share=True in Google Colab)."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.blue,
        neutral_hue=gr.themes.colors.gray,
    )
    app = create_app()
    app.launch(share=share, theme=theme, css=CUSTOM_CSS)


if __name__ == "__main__":
    launch_app(share=False)
