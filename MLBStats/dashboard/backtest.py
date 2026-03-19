"""Backtest page: simulate betting on historical games with Kelly criterion."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import DATA_PROCESSED, DATA_RAW, MODELS_SAVED, load_json
from utils.odds_math import american_to_decimal, american_to_implied_prob, remove_vig, kelly_fraction


def load_backtest_data():
    """Load test set predictions merged with historical odds."""
    ml_path = DATA_PROCESSED / "ml_ready.csv"
    odds_path = DATA_RAW / "historical_odds.csv"

    if not ml_path.exists():
        return None

    df = pd.read_csv(ml_path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Use test set (last 10%)
    n = len(df)
    test_df = df.iloc[int(n * 0.9):].copy()

    # Try to merge historical odds
    if odds_path.exists():
        odds = pd.read_csv(odds_path)
        odds["date"] = pd.to_datetime(odds["date"])
        test_df = test_df.merge(
            odds[["date", "home_team", "away_team", "home_odds", "away_odds",
                  "home_implied_prob", "away_implied_prob"]],
            on=["date", "home_team", "away_team"],
            how="left"
        )

    # For games without odds, use -110 both sides as fallback
    if "home_odds" not in test_df.columns:
        test_df["home_odds"] = -110
        test_df["away_odds"] = -110
        test_df["home_implied_prob"] = 0.5
        test_df["away_implied_prob"] = 0.5
    else:
        test_df["home_odds"] = test_df["home_odds"].fillna(-110)
        test_df["away_odds"] = test_df["away_odds"].fillna(-110)
        test_df["home_implied_prob"] = test_df["home_implied_prob"].fillna(0.5)
        test_df["away_implied_prob"] = test_df["away_implied_prob"].fillna(0.5)

    return test_df


def run_backtest(test_df, model_probs, bankroll=1000, kelly_frac=0.25, min_edge=0.02):
    """Simulate betting with Kelly criterion."""
    results = []
    balance_kelly = bankroll
    balance_flat = bankroll
    flat_bet = bankroll * 0.02  # 2% flat stake

    for i, (_, row) in enumerate(test_df.iterrows()):
        prob = model_probs[i]
        actual = row["home_win"]
        home_odds = row["home_odds"]
        away_odds = row["away_odds"]

        # Decide whether to bet home or away
        home_dec = american_to_decimal(home_odds)
        away_dec = american_to_decimal(away_odds)
        home_edge = prob - row["home_implied_prob"]
        away_edge = (1 - prob) - row["away_implied_prob"]

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
                "date": row.get("date"),
                "bet": "skip",
                "balance_kelly": balance_kelly,
                "balance_flat": balance_flat,
            })
            continue

        # Kelly sizing
        k = kelly_fraction(bet_prob, bet_dec, fraction=kelly_frac)
        kelly_wager = balance_kelly * k
        won = (bet_side == "home" and actual == 1) or (bet_side == "away" and actual == 0)

        if won:
            balance_kelly += kelly_wager * (bet_dec - 1)
            balance_flat += flat_bet * (bet_dec - 1)
        else:
            balance_kelly -= kelly_wager
            balance_flat -= flat_bet

        results.append({
            "date": row.get("date"),
            "bet": bet_side,
            "edge": round(edge, 4),
            "prob": round(bet_prob, 4),
            "odds": round(bet_dec, 3),
            "kelly_wager": round(kelly_wager, 2),
            "won": won,
            "balance_kelly": round(balance_kelly, 2),
            "balance_flat": round(balance_flat, 2),
        })

    return pd.DataFrame(results)


def render():
    st.title("📈 Backtest")

    test_df = load_backtest_data()
    if test_df is None:
        st.warning("No data available. Run the full pipeline first.")
        return

    # Check if model is trained
    manifest_path = MODELS_SAVED / "model_manifest.json"
    if not manifest_path.exists():
        st.warning("Model not trained yet. Run `python models/train.py` first.")
        st.info("Once the model is trained, this page will simulate betting on historical games "
                "using Kelly criterion and show balance curves, Sharpe ratio, and ROI.")
        return

    st.info("Backtest simulates the model betting on the test set with real historical odds. "
            "Uses fractional Kelly criterion for bet sizing.")

    # Settings
    col1, col2, col3 = st.columns(3)
    with col1:
        bankroll = st.number_input("Starting Bankroll ($)", value=1000, step=100)
    with col2:
        kelly_frac = st.slider("Kelly Fraction", 0.1, 1.0, 0.25)
    with col3:
        min_edge = st.slider("Minimum Edge to Bet", 0.0, 0.10, 0.02)

    # TODO: Load model and generate predictions on test set
    # For now, show placeholder
    st.write(f"**Test set:** {len(test_df)} games")
    st.write("Model predictions will be generated here once the training pipeline is complete.")
