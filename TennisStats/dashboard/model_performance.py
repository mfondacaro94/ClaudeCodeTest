"""Model Performance page: test-set metrics and feature importance."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.helpers import MODELS_SAVED, load_json, get_logger

logger = get_logger("model_performance")


def render():
    st.title("Model Performance")

    metrics_path = MODELS_SAVED / "evaluation_metrics.json"
    if not metrics_path.exists():
        st.warning("No evaluation metrics found. Run `python models/evaluate.py` first.")
        return

    metrics = load_json(metrics_path)

    st.subheader("Test Set Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    col2.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    col3.metric("Log Loss", f"{metrics['log_loss']:.4f}")
    col4.metric("Brier Score", f"{metrics['brier_score']:.4f}")

    col5, col6, col7 = st.columns(3)
    col5.metric("Test Samples", f"{metrics['n_test']:,}")
    col6.metric("Test Period", f"{metrics.get('test_period_start', 'N/A')} to {metrics.get('test_period_end', 'N/A')}")
    col7.metric("Ensemble Models", f"{metrics.get('n_models', 7)}")

    # Feature Importance
    fi_path = MODELS_SAVED / "feature_importance.csv"
    if fi_path.exists():
        st.subheader("Top 30 Features")
        fi_df = pd.read_csv(fi_path)
        top30 = fi_df.head(30)

        fig = px.bar(
            top30.iloc[::-1],
            x="importance", y="feature",
            orientation="h",
            title="Feature Importance (CatBoost)",
            labels={"importance": "Importance", "feature": "Feature"},
        )
        fig.update_layout(template="plotly_white", height=700)
        st.plotly_chart(fig, use_container_width=True)
