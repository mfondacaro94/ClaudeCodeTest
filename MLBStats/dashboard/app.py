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
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar navigation
st.sidebar.title("⚾ MLB Betting Model")
page = st.sidebar.radio("Navigate", [
    "Today's Games",
    "Matchup Predictor",
    "Odds Comparison",
    "Team Comparison",
    "Backtest",
    "Totals/Props",
    "Model Performance",
])

if page == "Today's Games":
    from dashboard.upcoming import render
    render()
elif page == "Matchup Predictor":
    st.title("🔮 Matchup Predictor")
    st.info("Select two teams and their starting pitchers to predict the outcome.")
    # TODO: Implement team/pitcher selection + prediction display
    st.write("Coming soon: dropdown selectors for teams and starting pitchers, "
             "win probability gauge, radar chart, and tale of the tape.")
elif page == "Odds Comparison":
    st.title("💰 Odds Comparison")
    st.info("Input custom odds to calculate edge, EV, and Kelly bet size.")
    col1, col2 = st.columns(2)
    with col1:
        home_odds = st.number_input("Home Moneyline (American)", value=-150)
    with col2:
        away_odds = st.number_input("Away Moneyline (American)", value=130)
    model_prob = st.slider("Model Home Win Probability", 0.0, 1.0, 0.55)

    from utils.odds_math import american_to_decimal, american_to_implied_prob, compute_ev, kelly_fraction, remove_vig

    home_impl = american_to_implied_prob(home_odds)
    away_impl = american_to_implied_prob(away_odds)
    home_fair, away_fair = remove_vig(home_impl, away_impl)
    dec = american_to_decimal(home_odds)
    ev = compute_ev(model_prob, dec)
    kelly = kelly_fraction(model_prob, dec)

    st.metric("Edge", f"{(model_prob - home_fair)*100:.1f}%")
    st.metric("Expected Value ($1 bet)", f"${ev:.3f}")
    st.metric("Kelly Bet Size (quarter)", f"{kelly*100:.1f}%")

    if ev > 0:
        st.success("✅ Positive EV -- Value bet detected")
    else:
        st.error("❌ Negative EV -- No value")

elif page == "Team Comparison":
    st.title("📊 Team Comparison")
    st.write("Coming soon: side-by-side team stats, rolling form charts, head-to-head records.")

elif page == "Backtest":
    from dashboard.backtest import render
    render()

elif page == "Totals/Props":
    st.title("📈 Totals / Over-Under Props")
    st.write("Coming soon: over/under predictions for each game, model vs. posted line.")

elif page == "Model Performance":
    st.title("📉 Model Performance")
    from utils.helpers import MODELS_SAVED, load_json
    import pandas as pd

    metrics_path = MODELS_SAVED / "evaluation_metrics.json"
    if metrics_path.exists():
        metrics = load_json(metrics_path)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
        col2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
        col3.metric("Brier Score", f"{metrics['brier_score']:.4f}")
        col4.metric("Log Loss", f"{metrics['log_loss']:.4f}")

        st.write(f"**Test Period:** {metrics.get('test_period_start', 'N/A')} to {metrics.get('test_period_end', 'N/A')}")
        st.write(f"**Test Size:** {metrics.get('n_test', 'N/A')} games")
        st.write(f"**Models in Ensemble:** {metrics.get('n_models', 'N/A')}")

        # Feature importance
        fi_path = MODELS_SAVED / "feature_importance.csv"
        if fi_path.exists():
            fi = pd.read_csv(fi_path)
            st.subheader("Top 30 Features")
            import plotly.express as px
            fig = px.bar(fi.head(30), x="importance", y="feature", orientation="h",
                         title="Feature Importance (CatBoost)")
            fig.update_layout(yaxis=dict(autorange="reversed"), height=600)
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("No evaluation metrics found. Run `python models/evaluate.py` first.")
