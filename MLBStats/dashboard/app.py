"""MLB Stats Betting Model -- Streamlit Dashboard.

Main entry point with sidebar navigation across all pages.
Launch: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="MLB Betting Model",
    page_icon="baseball",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("MLB Betting Model")
page = st.sidebar.radio("Navigate", [
    "Backtest",
    "Odds Comparison",
    "Model Performance",
])

if page == "Backtest":
    from dashboard.backtest import render
    render()

elif page == "Odds Comparison":
    st.title("Odds Comparison")
    st.info("Input real odds to calculate edge, EV, and Kelly bet size.")
    col1, col2 = st.columns(2)
    with col1:
        home_odds = st.number_input("Home Moneyline (American)", value=-150)
    with col2:
        away_odds = st.number_input("Away Moneyline (American)", value=130)
    model_prob = st.slider("Model Home Win Probability", 0.0, 1.0, 0.55)

    from utils.odds_math import (
        american_to_decimal, american_to_implied_prob,
        compute_ev, kelly_fraction, remove_vig,
    )

    home_impl = american_to_implied_prob(home_odds)
    away_impl = american_to_implied_prob(away_odds)
    home_fair, away_fair = remove_vig(home_impl, away_impl)
    dec = american_to_decimal(home_odds)
    ev = compute_ev(model_prob, dec)
    kelly = kelly_fraction(model_prob, dec)

    col1, col2, col3 = st.columns(3)
    col1.metric("Edge", f"{(model_prob - home_fair)*100:.1f}%")
    col2.metric("Expected Value ($1 bet)", f"${ev:.3f}")
    col3.metric("Kelly Bet Size (quarter)", f"{kelly*100:.1f}%")

    if ev > 0:
        st.success("Positive EV -- Value bet detected")
    else:
        st.error("Negative EV -- No value")

elif page == "Model Performance":
    st.title("Model Performance")
    from utils.helpers import MODELS_SAVED, load_json
    import pandas as pd

    manifest_path = MODELS_SAVED / "model_manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        metrics = manifest.get("metrics", {})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy", f"{metrics.get('accuracy', 0)*100:.1f}%")
        col2.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.4f}")
        col3.metric("Brier Score", f"{metrics.get('brier_score', 0):.4f}")
        col4.metric("Log Loss", f"{metrics.get('log_loss', 0):.4f}")

        st.write(f"**Test Size:** {metrics.get('n_test', 'N/A')} games")
        st.write(f"**Models in Ensemble:** {manifest.get('n_catboost', 0)} CatBoost + {manifest.get('n_xgboost', 0)} XGBoost")
        st.write(f"**Features:** {len(manifest.get('features', []))}")

        # Feature importance from first CatBoost model
        from catboost import CatBoostClassifier
        model_path = MODELS_SAVED / "model_1.cbm"
        if model_path.exists():
            m = CatBoostClassifier()
            m.load_model(str(model_path))
            fi = pd.DataFrame({
                "feature": m.feature_names_,
                "importance": m.get_feature_importance(),
            }).sort_values("importance", ascending=True).tail(30)

            import plotly.express as px
            fig = px.bar(fi, x="importance", y="feature", orientation="h",
                         title="Top 30 Feature Importance")
            fig.update_layout(height=700, template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No model manifest found. Run `python models/train.py` first.")
