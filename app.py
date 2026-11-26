"""Smart Expense Assistant — Main Streamlit Entrypoint (Task 4)."""

from __future__ import annotations

import streamlit as st

from frontend.components.chat import render_chatbot_panel
from frontend.components.dashboard import render_dashboard_panel
from frontend.config import CONFIG
from frontend.services.assistant_service import AssistantService, load_expense_dataframe
from frontend.theme import get_custom_css


def main() -> None:
    """Configure page layout and render Chatbot + Financial Dashboard tabs."""
    st.set_page_config(
        page_title=CONFIG.app_title,
        page_icon="₹",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(get_custom_css(), unsafe_allow_html=True)

    st.markdown(
        """
        <div class="app-header">
            <h1>Smart Expense Assistant</h1>
            <p>Expense categorization, retrieval analytics, and reporting in Indian Rupees (₹)</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    service = AssistantService()
    df = load_expense_dataframe()

    tab_chat, tab_dashboard = st.tabs(["Assistant", "Analytics Dashboard"])
    with tab_chat:
        render_chatbot_panel(service)
    with tab_dashboard:
        render_dashboard_panel(df)


if __name__ == "__main__":
    main()
