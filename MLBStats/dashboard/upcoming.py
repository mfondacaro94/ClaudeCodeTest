"""Today's Games page: live odds + model predictions."""

import sys
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd

from utils.helpers import DATA_ODDS, load_json
from utils.odds_math import (
    american_to_implied_prob, american_to_decimal,
    remove_vig, compute_ev, compute_edge, kelly_fraction,
)


def load_cached_odds() -> list[dict]:
    cache_path = DATA_ODDS / "upcoming_cache.json"
    if not cache_path.exists():
        return []
    cache = load_json(cache_path)
    from scraper.scrape_odds import get_consensus_odds
    return get_consensus_odds(cache)


def render():
    st.title("⚾ Today's Games")

    games = load_cached_odds()

    if not games:
        st.warning("No upcoming games found. Run `python scraper/scrape_odds.py` to fetch live odds.")
        st.info("Once odds are fetched, this page will show model predictions, edges, and bet recommendations.")
        return

    st.write(f"**{len(games)} games** with odds available")

    # Try to load the model for predictions
    try:
        from dashboard.predictor import load_models
        models, manifest = load_models()
        has_model = True
    except Exception:
        has_model = False
        st.info("Model not yet trained. Showing odds data only. Run the training pipeline to see predictions.")

    rows = []
    for game in games:
        row = {
            "Time": game.get("commence_time", "")[:16].replace("T", " "),
            "Away": game["away_team"],
            "Home": game["home_team"],
            "Away ML": game["away_odds"],
            "Home ML": game["home_odds"],
            "Home Impl%": f"{game['home_implied_prob']*100:.1f}%",
            "Away Impl%": f"{game['away_implied_prob']*100:.1f}%",
        }

        if has_model:
            # TODO: Build features for this game and predict
            # For now, show placeholder
            row["Model Home%"] = "—"
            row["Edge"] = "—"
            row["Bet"] = "—"

        rows.append(row)

    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Odds sourced from TheOddsAPI. Model predictions require scraped data + trained model.")
