"""Enterprise Gradio chatbot and financial analytics interface."""

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
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
    background-color: #F9FAFB !important;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif !important;
}
.ea-header {
    border-bottom: 1px solid #E5E7EB;
    padding: 12px 0 14px 0;
    margin-bottom: 12px;
}
.ea-header h1 {
    font-size: 1.4rem;
    font-weight: 600;
    color: #111827;
    margin: 0;
    letter-spacing: -0.01em;
}
.ea-header p {
    font-size: 0.88rem;
    color: #4B5563;
    margin: 4px 0 0 0;
}
.ea-kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin-bottom: 12px;
}
.ea-kpi-card {
    background: #FFFFFF;
    border: 1px solid #E5E7EB;
    border-radius: 6px;
    padding: 14px 16px;
}
.ea-kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #4B5563;
    margin-bottom: 4px;
}
.ea-kpi-value {
    font-size: 1.25rem;
    font-weight: 600;
    color: #111827;
    font-variant-numeric: tabular-nums;
}
.ea-kpi-sub {
    font-size: 0.75rem;
    color: #6B7280;
    margin-top: 3px;
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
            <div class="ea-kpi-sub">{len(df)} transactions</div>
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
) -> tuple[str, list[dict[str, str]], pd.DataFrame, Any]:
    """Process User -> LLM Router -> Backend Router -> LLM Response and update UI."""
    chat_history = list(history or [])
    empty_df = pd.DataFrame(columns=["expense_id", "description", "amount", "date", "category"])

    if not isinstance(user_message, str) or not user_message.strip():
        chat_history.append({"role": "assistant", "content": "Please enter a question or expense description."})
        return "", chat_history, empty_df, None

    query = user_message.strip()
    routed = route_query(query)
    if not routed.get("function"):
        reply = routed.get("error", "Unsupported query.")
        chat_history.append({"role": "user", "content": query})
        chat_history.append({"role": "assistant", "content": reply})
        return "", chat_history, empty_df, None

    tool_output = execute_tool_call(routed)
    answer = generate_response(tool_output)
    chat_history.append({"role": "user", "content": query})
    chat_history.append({"role": "assistant", "content": answer})

    table_out = empty_df
    chart_out = None
    if tool_output.get("ok"):
        fn = tool_output.get("function")
        res = tool_output.get("result")
        if fn == "get_expenses" and isinstance(res, list) and res:
            table_out = pd.DataFrame(res)
        elif fn == "get_category_breakdown" and isinstance(res, dict):
            chart_out = breakdown_dict_chart(res, "Category Breakdown")
        elif fn == "compare_months" and isinstance(res, dict):
            chart_out = monthly_comparison_chart(_load_df(), int(res["month_a"]), int(res["month_b"]))

    return "", chat_history, table_out, chart_out


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
    status_msg = generate_response(tool_output)
    df = _load_df()
    return (
        status_msg,
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
        gr.HTML(
            """
            <div class="ea-header">
                <h1>Smart Expense Assistant</h1>
                <p>Conversational expense intelligence, ML categorization, and INR (₹) analytics</p>
            </div>
            """
        )

        with gr.Tabs():
            with gr.Tab("Assistant"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="Conversation",
                            height=420,
                        )
                        with gr.Row():
                            query_box = gr.Textbox(
                                placeholder="Ask about your expenses (e.g., 'How much did I spend on food in November?')",
                                show_label=False,
                                scale=5,
                            )
                            send_btn = gr.Button("Send", variant="primary", scale=1)
                            clear_btn = gr.Button("Clear", scale=1)

                        gr.Markdown("**Sample Queries**")
                        sample_btns: list[tuple[str, gr.Button]] = []
                        with gr.Row():
                            for q in SAMPLE_QUERIES[:3]:
                                sample_btns.append((q, gr.Button(q, size="sm")))
                        with gr.Row():
                            for q in SAMPLE_QUERIES[3:]:
                                sample_btns.append((q, gr.Button(q, size="sm")))
                    with gr.Column(scale=2):
                        chat_chart = gr.Plot(label="Query Visualization")
                        chat_table = gr.Dataframe(
                            label="Matching Transactions",
                            interactive=False,
                            wrap=True,
                        )

                for q_text, q_btn in sample_btns:
                    q_btn.click(
                        fn=lambda h, text=q_text: handle_chat_query(text, h),
                        inputs=[chatbot],
                        outputs=[query_box, chatbot, chat_table, chat_chart],
                    )

                # Wire chat controls cleanly to all outputs
                send_btn.click(
                    fn=handle_chat_query,
                    inputs=[query_box, chatbot],
                    outputs=[query_box, chatbot, chat_table, chat_chart],
                )
                query_box.submit(
                    fn=handle_chat_query,
                    inputs=[query_box, chatbot],
                    outputs=[query_box, chatbot, chat_table, chat_chart],
                )
                clear_btn.click(
                    fn=lambda: ([], pd.DataFrame(), None),
                    outputs=[chatbot, chat_table, chat_chart],
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
                        label="Filter Category",
                    )
                    search_filter = gr.Textbox(
                        placeholder="Search description (e.g., Swiggy, Uber, electricity)...",
                        label="Search",
                    )
                ledger_table = gr.Dataframe(value=df_init.head(150), interactive=False, wrap=True)
                cat_filter.change(fn=filter_ledger, inputs=[cat_filter, search_filter], outputs=[ledger_table])
                search_filter.change(fn=filter_ledger, inputs=[cat_filter, search_filter], outputs=[ledger_table])

            with gr.Tab("Add Expense"):
                gr.Markdown("Add a validated expense to `data/runtime/expenses.csv` (seed CSV remains read-only).")
                with gr.Row():
                    add_desc = gr.Textbox(label="Description", placeholder="e.g., Uber ride to office")
                    add_amt = gr.Number(label="Amount (₹)", value=250.0, minimum=1.0)
                with gr.Row():
                    add_date = gr.Textbox(label="Date (YYYY-MM-DD)", value=date.today().isoformat())
                    add_cat = gr.Dropdown(
                        choices=["Auto (ML Classifier)", *ALLOWED_CATEGORIES],
                        value="Auto (ML Classifier)",
                        label="Category",
                    )
                add_btn = gr.Button("Add Expense", variant="primary")
                add_status = gr.Markdown()

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

    return demo


def launch_app(share: bool = False) -> None:
    """Launch the Gradio application (set share=True in Google Colab)."""
    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Inter"),
    )
    app = create_app()
    app.launch(share=share, theme=theme, css=CUSTOM_CSS)


if __name__ == "__main__":
    launch_app(share=False)
