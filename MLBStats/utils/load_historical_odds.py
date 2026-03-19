"""Load and normalize historical MLB betting odds from CSV sources."""

import pandas as pd
import numpy as np
from pathlib import Path
from utils.helpers import get_logger, DATA_RAW
from utils.data_cleaning import normalize_team_name

logger = get_logger(__name__)


def detect_format(df: pd.DataFrame) -> str:
    """Auto-detect the CSV format based on column names."""
    cols = set(df.columns.str.lower())
    if {"home_team", "away_team", "home_odds", "away_odds"}.issubset(cols):
        return "standard"
    if {"team1", "team2", "odds1", "odds2"}.issubset(cols):
        return "generic"
    if {"home", "away", "ml_home", "ml_away"}.issubset(cols):
        return "moneyline"
    logger.warning(f"Unknown format. Columns: {df.columns.tolist()}")
    return "unknown"


def load_and_normalize(input_path: Path = None, fmt: str = "auto") -> pd.DataFrame:
    """Load historical odds CSV, normalize to standard format.

    Output columns: date, home_team, away_team, home_odds, away_odds,
                    home_implied_prob, away_implied_prob, vig
    """
    if input_path is None:
        input_path = DATA_RAW / "MLB_betting_odds.csv"

    if not input_path.exists():
        logger.error(f"Odds file not found: {input_path}")
        return pd.DataFrame()

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")

    if fmt == "auto":
        fmt = detect_format(df)

    if fmt == "standard":
        df = df.rename(columns={
            "home_team": "home_team", "away_team": "away_team",
            "home_odds": "home_odds", "away_odds": "away_odds",
            "date": "date"
        })
    elif fmt == "generic":
        df = df.rename(columns={
            "team1": "home_team", "team2": "away_team",
            "odds1": "home_odds", "odds2": "away_odds",
        })
    elif fmt == "moneyline":
        df = df.rename(columns={
            "home": "home_team", "away": "away_team",
            "ml_home": "home_odds", "ml_away": "away_odds",
        })

    # Normalize team names
    df["home_team"] = df["home_team"].apply(normalize_team_name)
    df["away_team"] = df["away_team"].apply(normalize_team_name)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Convert odds to float
    df["home_odds"] = pd.to_numeric(df["home_odds"], errors="coerce")
    df["away_odds"] = pd.to_numeric(df["away_odds"], errors="coerce")
    df = df.dropna(subset=["home_odds", "away_odds"])

    # Compute implied probabilities
    from utils.odds_math import american_to_implied_prob, remove_vig
    df["home_implied_raw"] = df["home_odds"].apply(american_to_implied_prob)
    df["away_implied_raw"] = df["away_odds"].apply(american_to_implied_prob)
    df["vig"] = df["home_implied_raw"] + df["away_implied_raw"] - 1

    probs = df.apply(
        lambda r: remove_vig(r["home_implied_raw"], r["away_implied_raw"]),
        axis=1, result_type="expand"
    )
    df["home_implied_prob"] = probs[0]
    df["away_implied_prob"] = probs[1]
    df = df.drop(columns=["home_implied_raw", "away_implied_raw"])

    output_path = DATA_RAW / "historical_odds.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(df)} normalized odds to {output_path}")
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=None)
    parser.add_argument("--format", type=str, default="auto")
    args = parser.parse_args()
    input_path = Path(args.input) if args.input else None
    load_and_normalize(input_path, args.format)
