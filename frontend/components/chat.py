"""Conversational chatbot UI component with structured response rendering."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.contracts import AssistantResponse, ResponseType, VisualizationSpec
from frontend.services.assistant_service import AssistantService
from frontend.theme import PALETTE, format_indian_date, format_inr

SUGGESTED_QUERIES = [
    "How much did I spend on food in November?",
    "Which category has the highest spending?",
    "Compare my expenses between October and November.",
    "Categorize this expense: Uber ride to office, ₹250",
    "Show all my expenses in November.",
    "How much did I spend on shopping in October?",
]


def _render_source_badge(source: str) -> None:
    labels = {
        "backend": "Live Backend (Task 3)",
        "ml_classifier": "Task 1 ML Classifier",
        "mock": "Development Fallback (Real CSV)",
        "frontend": "Validation",
    }
    label = labels.get(source, source)
    st.markdown(f'<span class="source-badge">{label}</span>', unsafe_allow_html=True)


def _render_visualization(viz: VisualizationSpec, key_prefix: str) -> None:
    if viz.chart_type == "donut" and isinstance(viz.data, dict) and viz.data:
        df_plot = pd.DataFrame(
            {"Category": list(viz.data.keys()), "Amount": list(viz.data.values())}
        )
        fig = px.pie(
            df_plot,
            names="Category",
            values="Amount",
            hole=0.5,
            title=viz.title,
            color_discrete_sequence=PALETTE["chart_colors"],
        )
        fig.update_traces(textinfo="label+percent", hovertemplate="%{label}: ₹%{value:,.2f}<extra></extra>")
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_donut")
    elif viz.chart_type == "comparison" and isinstance(viz.data, list) and viz.data:
        df_plot = pd.DataFrame(viz.data)
        value_cols = [c for c in df_plot.columns if c != "category"]
        df_melted = df_plot.melt(id_vars="category", value_vars=value_cols, var_name="Month", value_name="Amount")
        fig = px.bar(
            df_melted,
            x="category",
            y="Amount",
            color="Month",
            barmode="group",
            title=viz.title,
            color_discrete_sequence=PALETTE["chart_colors"],
        )
        fig.update_yaxes(title_text=viz.y_label)
        fig.update_xaxes(title_text="Category")
        fig.update_layout(margin=dict(t=40, b=10, l=10, r=10), height=320)
        st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}_comp")


def _render_structured_payload(resp: AssistantResponse, idx: int) -> None:
    data = resp.structured_data or {}

    if resp.response_type == ResponseType.CATEGORIZATION and "category" in data:
        col1, col2 = st.columns(2)
        col1.metric("Predicted Category", str(data["category"]))
        col2.metric("Model Confidence", f"{float(data.get('confidence', 0.0)):.2%}")
        st.caption("Confidence reflects relative LinearSVC margin strength, not a calibrated probability.")

    elif resp.response_type == ResponseType.TRANSACTION_LIST and "records" in data:
        records = data["records"]
        if records:
            df_rec = pd.DataFrame(records)
            df_rec["Amount"] = df_rec["amount"].apply(format_inr)
            df_rec["Date"] = df_rec["date"].apply(format_indian_date)
            st.dataframe(
                df_rec[["Date", "description", "category", "Amount"]].rename(
                    columns={"description": "Description", "category": "Category"}
                ),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No matching transactions found.")

    if resp.visualization is not None:
        _render_visualization(resp.visualization, key_prefix=f"chat_viz_{idx}")


def render_chatbot_panel(service: AssistantService) -> None:
    """Render the conversational expense assistant with sidebar history and retry."""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {"Conversation 1": []}
    if "active_conv" not in st.session_state:
        st.session_state.active_conv = "Conversation 1"
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    with st.sidebar:
        st.subheader("Conversations")
        if st.button("+ New Conversation", use_container_width=True):
            new_name = f"Conversation {len(st.session_state.conversations) + 1}"
            st.session_state.conversations[new_name] = []
            st.session_state.active_conv = new_name
            st.rerun()

        conv_names = list(st.session_state.conversations.keys())
        selected = st.radio(
            "Select conversation",
            conv_names,
            index=conv_names.index(st.session_state.active_conv),
            label_visibility="collapsed",
        )
        st.session_state.active_conv = selected

        if st.button("Clear Current Chat", use_container_width=True):
            st.session_state.conversations[st.session_state.active_conv] = []
            st.rerun()

        st.divider()
        st.caption("Backend Connection")
        st.info(service.get_backend_mode())

    st.subheader("Conversational Expense Assistant")
    st.caption("Ask about monthly spending, category breakdowns, comparisons, or categorize a new expense.")

    # Suggested clickable queries
    st.markdown("**Suggested queries:**")
    cols = st.columns(3)
    for idx, query in enumerate(SUGGESTED_QUERIES):
        if cols[idx % 3].button(query, key=f"suggest_{idx}", use_container_width=True):
            st.session_state.pending_prompt = query

    history: list[dict[str, Any]] = st.session_state.conversations[st.session_state.active_conv]

    for idx, entry in enumerate(history):
        role = entry["role"]
        with st.chat_message(role):
            if role == "assistant" and "response" in entry:
                resp: AssistantResponse = entry["response"]
                _render_source_badge(resp.source)
                if resp.response_type == ResponseType.ERROR:
                    st.error(resp.message)
                    if st.button("Retry", key=f"retry_{idx}"):
                        if idx > 0 and history[idx - 1]["role"] == "user":
                            st.session_state.pending_prompt = history[idx - 1]["content"]
                            st.rerun()
                else:
                    st.markdown(resp.message)
                    _render_structured_payload(resp, idx)
            else:
                st.markdown(entry["content"])

    user_input = st.chat_input("Ask about your expenses (e.g., 'How much did I spend on food in November?')")
    prompt = user_input or st.session_state.pending_prompt
    if prompt:
        st.session_state.pending_prompt = None
        history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Analyzing expenses..."):
                api_history = [
                    {"role": m["role"], "content": m.get("content", "")}
                    for m in history[:-1]
                ]
                response = service.send_message(prompt, history=api_history)
            _render_source_badge(response.source)
            if response.response_type == ResponseType.ERROR:
                st.error(response.message)
            else:
                st.markdown(response.message)
                _render_structured_payload(response, len(history))

        history.append({"role": "assistant", "content": response.message, "response": response})
