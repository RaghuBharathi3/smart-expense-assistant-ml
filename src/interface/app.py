"""Apple-level minimal Gradio interface for Smart Expense Assistant."""

from __future__ import annotations

from datetime import date
from typing import Any

import gradio as gr
import pandas as pd

from src.backend.router import execute_tool_call
from src.backend.tools import get_expenses
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
from src.llm.router_llm import route_query

SAMPLE_QUERIES = [
    "How much did I spend on food in November?",
    "Which category has the highest spending?",
    "Compare my expenses between October and November.",
    "Categorize this expense: Uber ride to office, ₹250",
    "Show all my expenses in November.",
    "How much did I spend on shopping in October?",
]

CUSTOM_CSS = """
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    background-color: #F5F5F7 !important;
    color: #1D1D1F !important;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "SF Pro Display", "Inter", system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

.ea-topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
    padding: 16px 0 14px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.08);
    margin-bottom: 14px;
}

.ea-title {
    font-size: 1.38rem;
    font-weight: 600;
    color: #1D1D1F;
    letter-spacing: -0.02em;
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
    gap: 6px;
    background: #FFFFFF;
    border: 1px solid rgba(0, 0, 0, 0.08);
    border-radius: 999px;
    padding: 5px 12px;
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

.ea-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
    margin: 4px 0 16px 0;
    animation: fadeSlideUp 0.22s ease-out;
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
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
}

.ea-kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
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
    border-radius: 10px;
    padding: 10px 14px;
    font-size: 0.84rem;
    color: #1D1D1F;
    animation: fadeSlideUp 0.18s ease-out;
}

.ea-feedback.error {
    background: #FEF2F2;
    border-color: #FECACA;
    color: #991B1B;
}

button {
    border-radius: 8px !important;
    transition: transform 0.12s ease, background-color 0.15s ease, border-color 0.15s ease !important;
}

button:active {
    transform: scale(0.985) !important;
}

button:focus-visible, input:focus-visible, textarea:focus-visible {
    outline: 2px solid #0066CC !important;
    outline-offset: 2px !important;
}

@media (max-width: 768px) {
    .ea-kpi-grid {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
}
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
            <div class="ea-kpi-value">{top_cat}</div>
            <div class="ea-kpi-sub">{format_inr(top_amt)} total</div>
        </div>
        <div class="ea-kpi-card">
            <div class="ea-kpi-label">Average Daily Spend</div>
            <div class="ea-kpi-value">{format_inr(avg_daily)}</div>
            <div class="ea-kpi-sub">Across active days</div>
        </div>
    </div>
    """


def handle_chat_query(
    user_message: str,
    history: list[dict[str, str]] | None,
) -> tuple[str, list[dict[str, str]], Any, Any, str, str]:
    """Process User -> LLM Router -> Backend Router -> LLM Response with progressive disclosure."""
    chat_history = list(history or [])
    empty_df = pd.DataFrame(columns=["expense_id", "description", "amount", "date", "category"])

    if not isinstance(user_message, str) or not user_message.strip():
        feedback = "<div class='ea-feedback error'>Please enter a question or expense description.</div>"
        return "", chat_history, gr.update(visible=False, value=empty_df), gr.update(visible=False, value=None), feedback, ""

    query = user_message.strip()
    routed = route_query(query)
    if not routed.get("function"):
        reply = routed.get("error", "Unsupported query.")
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": reply})
        feedback = f"<div class='ea-feedback error'>{reply}</div>"
        return "", chat_history, gr.update(visible=False, value=empty_df), gr.update(visible=False, value=None), feedback, query

    tool_output = execute_tool_call(routed)
    answer = generate_response(tool_output)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    if not tool_output.get("ok"):
        feedback = f"<div class='ea-feedback error'>{answer}</div>"
        return "", chat_history, gr.update(visible=False, value=empty_df), gr.update(visible=False, value=None), feedback, query

    fn = tool_output.get("function")
    res = tool_output.get("result")
    feedback = f"<div class='ea-feedback'>Completed <b>{fn}</b> cleanly.</div>"

    table_update: Any = gr.update(visible=False, value=empty_df)
    chart_update: Any = gr.update(visible=False, value=None)

    if fn == "get_expenses" and isinstance(res, list) and res:
        table_update = gr.update(visible=True, value=pd.DataFrame(res))
    elif fn == "get_category_breakdown" and isinstance(res, dict):
        chart_update = gr.update(visible=True, value=breakdown_dict_chart(res, "Category Breakdown"))
    elif fn == "compare_months" and isinstance(res, dict):
        chart_update = gr.update(
            visible=True,
            value=monthly_comparison_chart(_load_df(), int(res["month_a"]), int(res["month_b"])),
        )

    return "", chat_history, table_update, chart_update, feedback, query


def filter_ledger(category: str, search_text: str) -> pd.DataFrame:
    """Filter expense ledger by category and description substring."""
    cat_arg = None if category == "All" else category
    df = pd.DataFrame(get_expenses(category=cat_arg))
    if df.empty:
        return df
    if search_text and search_text.strip():
        df = df[df["description"].str.contains(search_text.strip(), case=False, na=False)]
    return df.head(150)


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
    status_html = f"<div class='{css_cls}'>{msg}</div>"
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
                    <p class="ea-subtitle">Natural-language expense intelligence, ML categorization & INR (₹) analytics</p>
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
                            height=430,
                            placeholder="Ask a question about your expenses or pick a prompt below.",
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
                            for q in SAMPLE_QUERIES[:3]:
                                sample_btns.append((q, gr.Button(q, size="sm")))
                        with gr.Row():
                            for q in SAMPLE_QUERIES[3:]:
                                sample_btns.append((q, gr.Button(q, size="sm")))

                    with gr.Column(scale=2):
                        chat_chart = gr.Plot(label="Visual Breakdown", visible=False)
                        chat_table = gr.Dataframe(
                            label="Matching Transactions",
                            interactive=False,
                            wrap=True,
                            visible=False,
                        )

                outputs_chat = [query_box, chatbot, chat_table, chat_chart, feedback_html, last_query_state]
                send_btn.click(fn=handle_chat_query, inputs=[query_box, chatbot], outputs=outputs_chat)
                query_box.submit(fn=handle_chat_query, inputs=[query_box, chatbot], outputs=outputs_chat)
                retry_btn.click(fn=handle_chat_query, inputs=[last_query_state, chatbot], outputs=outputs_chat)
                clear_btn.click(
                    fn=lambda: ([], gr.update(visible=False, value=pd.DataFrame()), gr.update(visible=False, value=None), "", ""),
                    outputs=[chatbot, chat_table, chat_chart, feedback_html, last_query_state],
                )
                for q_text, q_btn in sample_btns:
                    q_btn.click(
                        fn=lambda h, text=q_text: handle_chat_query(text, h),
                        inputs=[chatbot],
                        outputs=outputs_chat,
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
                )
                month_b_dd.change(
                    fn=lambda a, b: monthly_comparison_chart(_load_df(), int(a), int(b)),
                    inputs=[month_a_dd, month_b_dd],
                    outputs=[plot_comp],
                )

                gr.Markdown("### Expense Ledger")
                with gr.Row():
                    cat_filter = gr.Dropdown(
                        choices=["All", *ALLOWED_CATEGORIES],
                        value="All",
                        label="Category",
                    )
                    search_filter = gr.Textbox(
                        placeholder="Search description (e.g. Swiggy, Uber, electricity)...",
                        label="Search",
                    )
                ledger_table = gr.Dataframe(value=df_init.head(150), interactive=False, wrap=True)
                cat_filter.change(fn=filter_ledger, inputs=[cat_filter, search_filter], outputs=[ledger_table])
                search_filter.change(fn=filter_ledger, inputs=[cat_filter, search_filter], outputs=[ledger_table])

            with gr.Tab("Add Expense"):
                gr.Markdown("Record a new expense into `data/runtime/expenses.csv` (seed data stays read-only).")
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
                with gr.Row():
                    add_btn = gr.Button("Add Expense", variant="primary")
                    reset_btn = gr.Button("Reset Form")
                add_status = gr.HTML()

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
                )
                reset_btn.click(
                    fn=lambda: ("", 250.0, date.today().isoformat(), "Auto (ML Classifier)", ""),
                    outputs=[add_desc, add_amt, add_date, add_cat, add_status],
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
