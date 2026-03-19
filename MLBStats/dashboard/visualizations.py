"""Shared Plotly chart utilities for the dashboard."""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np


def calibration_curve(y_true, y_prob, n_bins=10):
    """Create a calibration curve plot."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_true = []
    bin_pred = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() > 0:
            bin_true.append(y_true[mask].mean())
            bin_pred.append(y_prob[mask].mean())
        else:
            bin_true.append(None)
            bin_pred.append(None)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bin_pred, y=bin_true, mode="markers+lines",
        name="Model", marker=dict(size=10)
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Perfect", line=dict(dash="dash", color="gray")
    ))
    fig.update_layout(
        title="Calibration Curve",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Fraction of Positives",
        width=600, height=500,
    )
    return fig


def balance_curve(results_df):
    """Plot Kelly and flat staking balance over time."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=results_df["date"], y=results_df["balance_kelly"],
        name="Kelly", mode="lines"
    ))
    fig.add_trace(go.Scatter(
        x=results_df["date"], y=results_df["balance_flat"],
        name="Flat Staking", mode="lines"
    ))
    fig.update_layout(
        title="Bankroll Over Time",
        xaxis_title="Date",
        yaxis_title="Balance ($)",
        hovermode="x unified",
    )
    return fig


def edge_histogram(edges):
    """Histogram of betting edges."""
    fig = px.histogram(x=edges, nbins=30, title="Distribution of Edges")
    fig.update_layout(xaxis_title="Edge (%)", yaxis_title="Count")
    return fig


def roi_curve(results_df):
    """Cumulative ROI over time."""
    bets = results_df[results_df["bet"] != "skip"].copy()
    if bets.empty:
        return go.Figure()

    bets["cumulative_profit_kelly"] = bets["balance_kelly"] - bets["balance_kelly"].iloc[0]
    n_bets = range(1, len(bets) + 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(n_bets), y=bets["cumulative_profit_kelly"].values,
        name="Cumulative Profit (Kelly)", mode="lines",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        title="Cumulative Profit",
        xaxis_title="Bet Number",
        yaxis_title="Profit ($)",
    )
    return fig
