"""Backtest page: simulate betting on historical games with real Vegas odds."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.helpers import DATA_PROCESSED, DATA_RAW, MODELS_SAVED, load_json
from utils.odds_math import (
    american_to_decimal, american_to_implied_prob, remove_vig,
    kelly_fraction, compute_ev, compute_edge,
)


@st.cache_data
def load_predictions():
    """Load test set with model predictions and real odds."""
    ml_path = DATA_PROCESSED / "ml_ready.csv"
    odds_path = DATA_RAW / ".." / "odds" / "historical_odds.csv"

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

    exclude = {"home_win", "date", "home_team", "away_team"}
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

    # Merge real odds
    if odds_path.exists():
        odds = pd.read_csv(odds_path)
        odds["date"] = pd.to_datetime(odds["date"])
        test_df = test_df.merge(
            odds[["date", "home_team", "away_team", "home_ml", "away_ml",
                  "best_home_ml", "best_away_ml", "n_books"]],
            on=["date", "home_team", "away_team"],
            how="left",
        )

    # Fill missing odds with -110
    for col in ["home_ml", "away_ml", "best_home_ml", "best_away_ml"]:
        if col not in test_df.columns:
            test_df[col] = -110
        else:
            test_df[col] = test_df[col].fillna(-110)

    if "n_books" not in test_df.columns:
        test_df["n_books"] = 0
    else:
        test_df["n_books"] = test_df["n_books"].fillna(0)

    return test_df


def run_backtest(test_df, bankroll, kelly_frac, min_edge, use_best_line):
    """Simulate betting with Kelly criterion + flat staking."""
    results = []
    bal_kelly = bankroll
    bal_flat = bankroll
    flat_pct = 0.02  # 2% of initial bankroll

    for _, row in test_df.iterrows():
        prob = row["model_prob"]
        actual = int(row["home_win"])
        hml = row["best_home_ml"] if use_best_line else row["home_ml"]
        aml = row["best_away_ml"] if use_best_line else row["away_ml"]

        if hml == 0 or aml == 0:
            continue

        home_dec = american_to_decimal(hml)
        away_dec = american_to_decimal(aml)
        home_ip = american_to_implied_prob(hml)
        away_ip = american_to_implied_prob(aml)

        home_edge = prob - home_ip
        away_edge = (1 - prob) - away_ip

        bet_side = None
        bet_prob = None
        bet_dec = None
        edge = 0

        if home_edge > min_edge and home_edge >= away_edge:
            bet_side = "home"
            bet_prob = prob
            bet_dec = home_dec
            edge = home_edge
        elif away_edge > min_edge:
            bet_side = "away"
            bet_prob = 1 - prob
            bet_dec = away_dec
            edge = away_edge

        if bet_side is None:
            results.append({
                "date": row["date"],
                "home_team": row["home_team"],
                "away_team": row["away_team"],
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

        # Kelly sizing
        k = kelly_fraction(bet_prob, bet_dec, fraction=kelly_frac)
        kelly_wager = bal_kelly * k
        flat_wager = bankroll * flat_pct

        won = (bet_side == "home" and actual == 1) or (bet_side == "away" and actual == 0)

        if won:
            bal_kelly += kelly_wager * (bet_dec - 1)
            bal_flat += flat_wager * (bet_dec - 1)
        else:
            bal_kelly -= kelly_wager
            bal_flat -= flat_wager

        results.append({
            "date": row["date"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
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
    st.title("Backtest — Model vs Vegas")

    test_df = load_predictions()
    if test_df is None:
        st.warning("No data available. Run the full pipeline first.")
        return

    has_real_odds = (test_df["n_books"] > 0).sum()
    st.info(f"Simulating bets on **{len(test_df)}** test games "
            f"({has_real_odds} with real Vegas odds, rest use -110 fallback). "
            f"Uses fractional Kelly criterion for bet sizing.")

    # --- Settings ---
    st.subheader("Settings")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        bankroll = st.number_input("Starting Bankroll ($)", value=10000, step=1000, min_value=100)
    with col2:
        kelly_frac = st.slider("Kelly Fraction", 0.05, 1.0, 0.25, 0.05,
                                help="1.0 = full Kelly (aggressive), 0.25 = quarter Kelly (conservative)")
    with col3:
        min_edge = st.slider("Min Edge to Bet", 0.0, 0.15, 0.03, 0.01,
                              help="Only bet when model edge exceeds this threshold")
    with col4:
        use_best_line = st.checkbox("Use Best Available Line", value=True,
                                     help="Shop across 6 sportsbooks for best odds")

    # --- Run Backtest ---
    results = run_backtest(test_df, bankroll, kelly_frac, min_edge, use_best_line)
    bets = results[results["bet"] != "skip"].copy()

    if bets.empty:
        st.warning("No bets placed with these settings. Try lowering the minimum edge.")
        return

    # --- Summary Metrics ---
    st.subheader("Performance Summary")
    n_bets = len(bets)
    n_wins = bets["won"].sum()
    win_rate = n_wins / n_bets
    final_kelly = bets["bal_kelly"].iloc[-1]
    final_flat = bets["bal_flat"].iloc[-1]
    kelly_roi = (final_kelly - bankroll) / bankroll * 100
    flat_roi = (final_flat - bankroll) / bankroll * 100

    # Calculate max drawdown for Kelly
    bets_kelly = bets["bal_kelly"].values
    peak = np.maximum.accumulate(bets_kelly)
    drawdown = (peak - bets_kelly) / peak
    max_dd = drawdown.max() * 100

    # Sharpe-like ratio (using daily returns)
    bets["kelly_return"] = bets["bal_kelly"].pct_change().fillna(0)
    if bets["kelly_return"].std() > 0:
        sharpe = bets["kelly_return"].mean() / bets["kelly_return"].std() * np.sqrt(252)
    else:
        sharpe = 0

    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("Total Bets", f"{n_bets}")
    col2.metric("Win Rate", f"{win_rate:.1%}")
    col3.metric("Kelly P&L", f"${final_kelly - bankroll:+,.0f}", f"{kelly_roi:+.1f}%")
    col4.metric("Flat P&L", f"${final_flat - bankroll:+,.0f}", f"{flat_roi:+.1f}%")
    col5.metric("Max Drawdown", f"{max_dd:.1f}%")
    col6.metric("Sharpe Ratio", f"{sharpe:.2f}")

    # --- Balance Curves ---
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

    # --- Monthly P&L ---
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

    # --- Edge Distribution ---
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
        # Win rate by edge bucket
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

    # --- Recent Bets Table ---
    st.subheader("Recent Bets")
    display_cols = ["date", "home_team", "away_team", "bet", "edge", "model_prob",
                    "odds_dec", "kelly_pct", "kelly_wager", "won"]
    available = [c for c in display_cols if c in bets.columns]
    st.dataframe(
        bets[available].tail(50).sort_values("date", ascending=False),
        use_container_width=True, height=400,
    )
