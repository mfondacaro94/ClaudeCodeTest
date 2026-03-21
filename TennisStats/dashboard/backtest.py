"""Backtest page: simulate betting on historical matches with real bookmaker odds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import DATA_PROCESSED, MODELS_SAVED, load_json, get_logger
from utils.odds_math import (
    decimal_to_implied_prob, remove_vig,
    kelly_fraction, compute_ev, compute_edge,
)

logger = get_logger("backtest")


@st.cache_data
def load_predictions():
    """Load test set with model predictions and real odds."""
    ml_path = DATA_PROCESSED / "ml_ready.csv"
    if not ml_path.exists():
        return None

    df = pd.read_csv(ml_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Test set = last 10%
    n = len(df)
    test_df = df.iloc[int(n * 0.9):].copy()

    # Load models and generate predictions
    from catboost import CatBoostClassifier
    from xgboost import XGBClassifier

    exclude = {"p1_win", "date", "p1_name", "p2_name",
               "p1_odds_ps", "p2_odds_ps", "p1_odds_max", "p2_odds_max",
               "p1_odds_avg", "p2_odds_avg", "p1_odds_b365", "p2_odds_b365",
               "n_books"}
    feature_cols = [c for c in test_df.columns if c not in exclude]
    X_test = test_df[feature_cols].fillna(0)

    models = []
    for i in range(1, 6):
        p = MODELS_SAVED / f"model_{i}.cbm"
        if p.exists():
            m = CatBoostClassifier()
            m.load_model(str(p))
            models.append(m)
    for i in range(1, 3):
        p = MODELS_SAVED / f"xgb_model_{i}.json"
        if p.exists():
            m = XGBClassifier()
            m.load_model(str(p))
            models.append(m)

    if not models:
        return None

    probs = []
    for model in models:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_test)[:, 1]
        else:
            p = model.predict(X_test, prediction_type="Probability")[:, 1]
        probs.append(p)

    test_df["model_prob"] = np.mean(probs, axis=0)

    # Ensure odds columns exist
    for col in ["p1_odds_ps", "p2_odds_ps", "p1_odds_max", "p2_odds_max"]:
        if col not in test_df.columns:
            test_df[col] = np.nan
    if "n_books" not in test_df.columns:
        test_df["n_books"] = 0
    else:
        test_df["n_books"] = test_df["n_books"].fillna(0)

    # Filter to matches with real odds (need at least Pinnacle or Bet365)
    before = len(test_df)
    has_ps = test_df["p1_odds_ps"].notna() & test_df["p2_odds_ps"].notna()
    has_max = test_df["p1_odds_max"].notna() & test_df["p2_odds_max"].notna()
    test_df = test_df[has_ps | has_max].copy()
    logger.info(f"Filtered to {len(test_df)}/{before} matches with real bookmaker odds")

    return test_df


def run_backtest(test_df, bankroll, kelly_frac, min_edge, use_best_line):
    """Simulate betting with Kelly criterion + flat staking."""
    results = []
    bal_kelly = bankroll
    bal_flat = bankroll
    flat_pct = 0.02

    for _, row in test_df.iterrows():
        prob = row["model_prob"]
        actual = int(row["p1_win"])

        # Odds in decimal format
        if use_best_line:
            p1_dec = row.get("p1_odds_max", np.nan)
            p2_dec = row.get("p2_odds_max", np.nan)
        else:
            p1_dec = row.get("p1_odds_ps", np.nan)
            p2_dec = row.get("p2_odds_ps", np.nan)

        if pd.isna(p1_dec) or pd.isna(p2_dec) or p1_dec <= 1 or p2_dec <= 1:
            continue

        p1_ip = decimal_to_implied_prob(p1_dec)
        p2_ip = decimal_to_implied_prob(p2_dec)

        p1_edge = prob - p1_ip
        p2_edge = (1 - prob) - p2_ip

        bet_side = None
        bet_prob = None
        bet_dec = None
        edge = 0

        if p1_edge > min_edge and p1_edge >= p2_edge:
            bet_side = "p1"
            bet_prob = prob
            bet_dec = p1_dec
            edge = p1_edge
        elif p2_edge > min_edge:
            bet_side = "p2"
            bet_prob = 1 - prob
            bet_dec = p2_dec
            edge = p2_edge

        if bet_side is None:
            results.append({
                "date": row["date"],
                "p1_name": row.get("p1_name", ""),
                "p2_name": row.get("p2_name", ""),
                "bet": "skip",
                "edge": 0,
                "won": None,
                "kelly_wager": 0,
                "flat_wager": 0,
                "bal_kelly": bal_kelly,
                "bal_flat": bal_flat,
                "model_prob": prob,
            })
            continue

        k = kelly_fraction(bet_prob, bet_dec, fraction=kelly_frac)
        k = min(k, 0.05)  # Cap at 5% of bankroll per bet to prevent blow-ups
        kelly_wager = bal_kelly * k
        flat_wager = bankroll * flat_pct

        won = (bet_side == "p1" and actual == 1) or (bet_side == "p2" and actual == 0)

        if won:
            bal_kelly += kelly_wager * (bet_dec - 1)
            bal_flat += flat_wager * (bet_dec - 1)
        else:
            bal_kelly -= kelly_wager
            bal_flat -= flat_wager

        results.append({
            "date": row["date"],
            "p1_name": row.get("p1_name", ""),
            "p2_name": row.get("p2_name", ""),
            "bet": bet_side,
            "edge": round(edge, 4),
            "model_prob": round(bet_prob, 4),
            "odds_dec": round(bet_dec, 3),
            "kelly_pct": round(k * 100, 2),
            "kelly_wager": round(kelly_wager, 2),
            "flat_wager": round(flat_wager, 2),
            "won": won,
            "bal_kelly": round(bal_kelly, 2),
            "bal_flat": round(bal_flat, 2),
        })

    return pd.DataFrame(results)


def render():
    st.title("Backtest - Model vs Bookmakers")

    test_df = load_predictions()
    if test_df is None:
        st.warning("No data available. Run the full pipeline first.")
        return

    st.info(f"Simulating bets on **{len(test_df)}** test matches with real bookmaker odds "
            f"(Pinnacle + market). Uses fractional Kelly criterion for bet sizing.")

    st.subheader("Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bankroll = st.number_input("Starting Bankroll ($)", value=10000, step=1000, min_value=100)
    with col2:
        kelly_frac = st.slider("Kelly Fraction", 0.05, 0.50, 0.10, 0.05,
                                help="0.10 = tenth Kelly (conservative), 0.25 = quarter Kelly. Max bet capped at 5%.")
    with col3:
        min_edge = st.slider("Min Edge to Bet", 0.0, 0.15, 0.03, 0.01,
                              help="Only bet when model edge exceeds this threshold")
    with col4:
        use_best_line = st.checkbox("Use Best Available Line", value=True,
                                     help="Use maximum odds across all bookmakers")

    results = run_backtest(test_df, bankroll, kelly_frac, min_edge, use_best_line)

    if results.empty:
        st.warning("No results generated. Check that odds data is available for the test period.")
        return

    bets = results[results["bet"] != "skip"].copy()

    if bets.empty:
        st.warning("No bets placed with these settings. Try lowering the minimum edge.")
        return

    # Summary Metrics
    st.subheader("Performance Summary")
    n_bets = len(bets)
    n_wins = bets["won"].sum()
    win_rate = n_wins / n_bets
    final_kelly = bets["bal_kelly"].iloc[-1]
    final_flat = bets["bal_flat"].iloc[-1]
    kelly_roi = (final_kelly - bankroll) / bankroll * 100
    flat_roi = (final_flat - bankroll) / bankroll * 100

    bets_kelly = bets["bal_kelly"].values
    peak = np.maximum.accumulate(bets_kelly)
    drawdown = (peak - bets_kelly) / peak
    max_dd = drawdown.max() * 100

    # Flat bet ROI (most honest metric)
    flat_bets_only = bets[bets["flat_wager"] > 0]
    if len(flat_bets_only) > 0:
        flat_profit_per_bet = (final_flat - bankroll) / (bankroll * 0.02 * n_bets) * 100 if n_bets > 0 else 0
    else:
        flat_profit_per_bet = 0

    # Max drawdown for flat betting
    bets_flat = bets["bal_flat"].values
    peak_flat = np.maximum.accumulate(bets_flat)
    dd_flat = (peak_flat - bets_flat) / peak_flat
    max_dd_flat = dd_flat.max() * 100

    # Aggregate to daily returns for proper Sharpe calculation (use flat)
    bets["date_day"] = pd.to_datetime(bets["date"]).dt.date
    daily_bal = bets.groupby("date_day")["bal_flat"].last()
    daily_returns = daily_bal.pct_change().dropna()
    if len(daily_returns) > 1 and daily_returns.std() > 0:
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252)
    else:
        sharpe = 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Bets", f"{n_bets}")
    col2.metric("Win Rate", f"{win_rate:.1%}")
    col3.metric("Flat Bet ROI", f"{flat_roi:+.1f}%", f"${final_flat - bankroll:+,.0f}")
    col4.metric("Kelly P&L", f"${final_kelly - bankroll:+,.0f}", f"{(final_kelly - bankroll) / bankroll * 100:+.1f}%")
    col5.metric("Max Drawdown (Flat)", f"{max_dd_flat:.1f}%")
    col6.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # Balance Curves
    st.subheader("Balance Curves")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bets["date"], y=bets["bal_kelly"],
        name=f"Kelly ({kelly_frac:.0%})", mode="lines",
        line=dict(color="#2196F3", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=bets["date"], y=bets["bal_flat"],
        name="Flat (2%)", mode="lines",
        line=dict(color="#FF9800", width=2),
    ))
    fig.add_hline(y=bankroll, line_dash="dash", line_color="gray",
                  annotation_text=f"Starting: ${bankroll:,}")
    fig.update_layout(
        title="Bankroll Over Time",
        xaxis_title="Date", yaxis_title="Balance ($)",
        template="plotly_white", height=450,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Monthly P&L
    st.subheader("Monthly P&L")
    bets["month"] = bets["date"].dt.to_period("M").astype(str)
    monthly = bets.groupby("month").agg(
        n_bets=("won", "count"),
        wins=("won", "sum"),
        kelly_start=("bal_kelly", "first"),
        kelly_end=("bal_kelly", "last"),
    ).reset_index()
    monthly["kelly_pnl"] = monthly["kelly_end"] - monthly["kelly_start"]
    monthly["win_rate"] = monthly["wins"] / monthly["n_bets"]

    colors = ["#4CAF50" if x >= 0 else "#F44336" for x in monthly["kelly_pnl"]]
    fig2 = go.Figure(go.Bar(
        x=monthly["month"], y=monthly["kelly_pnl"],
        marker_color=colors,
        text=[f"${x:+,.0f}" for x in monthly["kelly_pnl"]],
        textposition="outside",
    ))
    fig2.update_layout(
        title="Monthly Kelly P&L",
        xaxis_title="Month", yaxis_title="P&L ($)",
        template="plotly_white", height=400,
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Edge Distribution
    st.subheader("Edge Distribution")
    col1, col2 = st.columns(2)
    with col1:
        fig3 = px.histogram(bets, x="edge", nbins=30, color="won",
                             color_discrete_map={True: "#4CAF50", False: "#F44336"},
                             title="Edge Distribution (Wins vs Losses)",
                             labels={"edge": "Model Edge", "won": "Won?"})
        fig3.update_layout(template="plotly_white", height=350)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        bets["edge_bucket"] = pd.cut(bets["edge"], bins=[0, 0.03, 0.05, 0.08, 0.10, 0.15, 1.0],
                                      labels=["0-3%", "3-5%", "5-8%", "8-10%", "10-15%", "15%+"])
        bucket_stats = bets.groupby("edge_bucket", observed=True).agg(
            n=("won", "count"), wins=("won", "sum")
        ).reset_index()
        bucket_stats["win_rate"] = bucket_stats["wins"] / bucket_stats["n"]

        fig4 = go.Figure(go.Bar(
            x=bucket_stats["edge_bucket"].astype(str),
            y=bucket_stats["win_rate"],
            text=[f"{x:.0%}\n({n} bets)" for x, n in zip(bucket_stats["win_rate"], bucket_stats["n"])],
            textposition="outside",
            marker_color="#2196F3",
        ))
        fig4.add_hline(y=0.5, line_dash="dash", line_color="red", annotation_text="Breakeven")
        fig4.update_layout(title="Win Rate by Edge Bucket",
                           xaxis_title="Edge", yaxis_title="Win Rate",
                           template="plotly_white", height=350,
                           yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig4, use_container_width=True)

    # Recent Bets Table
    st.subheader("Recent Bets")
    display_cols = ["date", "p1_name", "p2_name", "bet", "edge", "model_prob",
                    "odds_dec", "kelly_pct", "kelly_wager", "won"]
    available = [c for c in display_cols if c in bets.columns]
    st.dataframe(
        bets[available].tail(50).sort_values("date", ascending=False),
        use_container_width=True, height=400,
    )
