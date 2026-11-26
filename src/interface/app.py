"""Streamlit chatbot and expense analytics application."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.backend.router import execute_tool_call
from src.backend.tools import get_expenses
from src.interface.charts import (
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


def _process_query(query: str) -> tuple[str, dict]:
    """Run User -> LLM Router -> Backend Router -> LLM Response."""
    routed = route_query(query)
    if not routed.get("function"):
        return routed.get("error", "Unsupported query."), {"ok": False, "error": routed.get("error")}
    tool_output = execute_tool_call(routed)
    return generate_response(tool_output), tool_output


def main() -> None:
    """Render the Smart Expense Assistant interface."""
    st.set_page_config(page_title="Smart Expense Assistant", page_icon="₹", layout="wide")
    st.markdown(
        """
        <style>
            .stApp { background-color: #F9FAFB; color: #111827; }
            .block-container { max-width: 1180px; padding-top: 1.4rem; }
            h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #111827; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Smart Expense Assistant")
    st.caption("Natural-language expense analytics, categorization, and reporting (INR ₹)")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None

    df = pd.DataFrame(get_expenses())
    tab_chat, tab_analytics = st.tabs(["Assistant", "Analytics & Charts"])

    with tab_chat:
        st.markdown("**Sample queries:**")
        cols = st.columns(3)
        for idx, q in enumerate(SAMPLE_QUERIES):
            if cols[idx % 3].button(q, key=f"sample_{idx}", use_container_width=True):
                st.session_state.pending_query = q

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("table") is not None:
                    st.dataframe(msg["table"], use_container_width=True, hide_index=True)

        user_input = st.chat_input("Ask about your expenses...")
        query = user_input or st.session_state.pending_query
        if query:
            st.session_state.pending_query = None
            st.session_state.messages.append({"role": "user", "content": query})
            with st.chat_message("user"):
                st.markdown(query)

            answer, tool_output = _process_query(query)
            table_df = None
            if tool_output.get("ok") and tool_output.get("function") == "get_expenses":
                table_df = pd.DataFrame(tool_output["result"])

            with st.chat_message("assistant"):
                if not tool_output.get("ok"):
                    st.error(answer)
                else:
                    st.markdown(answer)
                    if table_df is not None and not table_df.empty:
                        st.dataframe(table_df, use_container_width=True, hide_index=True)

            st.session_state.messages.append({"role": "assistant", "content": answer, "table": table_df})

    with tab_analytics:
        if df.empty:
            st.warning("No expense records available.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Spending", format_inr(float(df["amount"].sum())))
            c2.metric("Transactions", str(len(df)))
            top_cat = df.groupby("category")["amount"].sum().idxmax()
            c3.metric("Top Category", str(top_cat))

            r1_a, r1_b = st.columns(2)
            r1_a.plotly_chart(category_distribution_chart(df), use_container_width=True)
            r1_b.plotly_chart(monthly_trend_chart(df), use_container_width=True)

            r2_a, r2_b = st.columns(2)
            r2_a.plotly_chart(top_categories_chart(df), use_container_width=True)
            r2_b.plotly_chart(monthly_comparison_chart(df), use_container_width=True)

            st.subheader("Expense Records")
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
