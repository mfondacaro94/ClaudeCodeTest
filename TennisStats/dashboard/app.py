"""Streamlit dashboard for ATP Tennis Betting Model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="ATP Tennis Betting Model",
    page_icon="\U0001F3BE",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.backtest import render as render_backtest
from dashboard.model_performance import render as render_performance


def main():
    st.sidebar.title("\U0001F3BE ATP Tennis Model")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["Backtest", "Model Performance"],
    )

    if page == "Backtest":
        render_backtest()
    elif page == "Model Performance":
        render_performance()


if __name__ == "__main__":
    main()
