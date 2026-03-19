"""Feature engineering: raw data -> master.csv -> ml_ready.csv.

Transforms scraped game results, team stats, and pitcher stats into
ML-ready matchup features using rolling windows and diff/ratio construction.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, DATA_RAW, DATA_PROCESSED
from utils.data_cleaning import normalize_team_name, safe_float

logger = get_logger("feature_engineering")

ROLLING_WINDOWS = [10, 20, 40]

# Team batting stats to compute rolling averages for
TEAM_BAT_ROLLING = ["runs", "hits", "hr", "bb", "so", "sb", "obp", "slg", "ops"]
# Team pitching stats to compute rolling averages for
TEAM_PITCH_ROLLING = ["runs_allowed", "era", "whip", "k9", "bb9", "hr9"]


def load_games() -> pd.DataFrame:
    """Load and clean game results."""
    path = DATA_RAW / "games.csv"
    if not path.exists():
        logger.error(f"Games file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)

    # Ensure numeric
    df["home_runs"] = pd.to_numeric(df["home_runs"], errors="coerce")
    df["away_runs"] = pd.to_numeric(df["away_runs"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")

    # Filter out incomplete rows
    df = df.dropna(subset=["home_runs", "away_runs", "home_win"])

    logger.info(f"Loaded {len(df)} games")
    return df


def load_pitchers() -> pd.DataFrame:
    """Load pitcher season stats."""
    path = DATA_RAW / "pitchers.csv"
    if not path.exists():
        logger.warning("Pitchers file not found. Proceeding without pitcher features.")
        return pd.DataFrame()

    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} pitcher-season rows")
    return df


def compute_rolling_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling window stats for each team at each game date.

    CRITICAL: Uses .shift(1) to avoid data leakage -- only uses games
    prior to the current one.
    """
    # Build a per-team game log
    home_logs = games[["date", "home_team", "home_runs", "away_runs", "home_win"]].copy()
    home_logs.columns = ["date", "team", "runs", "runs_allowed", "won"]

    away_logs = games[["date", "away_team", "away_runs", "home_runs", "home_win"]].copy()
    away_logs.columns = ["date", "team", "runs", "runs_allowed", "won"]
    away_logs["won"] = 1 - away_logs["won"]

    team_log = pd.concat([home_logs, away_logs], ignore_index=True)
    team_log = team_log.sort_values(["team", "date"]).reset_index(drop=True)

    # Derived stats per game
    team_log["run_diff"] = team_log["runs"] - team_log["runs_allowed"]

    # Compute rolling stats for each window, shifted by 1 to prevent leakage
    rolling_dfs = []
    for window in ROLLING_WINDOWS:
        grouped = team_log.groupby("team")

        roll = pd.DataFrame()
        roll["date"] = team_log["date"]
        roll["team"] = team_log["team"]

        # Rolling means (shifted)
        roll[f"roll{window}_runs"] = grouped["runs"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )
        roll[f"roll{window}_runs_allowed"] = grouped["runs_allowed"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )
        roll[f"roll{window}_run_diff"] = grouped["run_diff"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )
        roll[f"roll{window}_win_pct"] = grouped["won"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )

        rolling_dfs.append(roll)

    # Merge all rolling windows
    result = rolling_dfs[0]
    for rdf in rolling_dfs[1:]:
        result = result.merge(rdf, on=["date", "team"], how="outer")

    # Season-to-date stats
    team_log["game_num"] = team_log.groupby(["team", team_log["date"].dt.year]).cumcount() + 1
    grouped = team_log.groupby(["team", team_log["date"].dt.year])
    team_log["szn_runs_pg"] = grouped["runs"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    team_log["szn_ra_pg"] = grouped["runs_allowed"].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    team_log["szn_win_pct"] = grouped["won"].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    result = result.merge(
        team_log[["date", "team", "game_num", "szn_runs_pg", "szn_ra_pg", "szn_win_pct"]],
        on=["date", "team"],
        how="left"
    )

    # Deduplicate (a team can play a doubleheader)
    result = result.drop_duplicates(subset=["date", "team"], keep="last")

    return result


def build_pitcher_features(pitchers: pd.DataFrame) -> pd.DataFrame:
    """Build per-pitcher season features for matchup merging."""
    if pitchers.empty:
        return pd.DataFrame()

    # Select key columns and rename
    cols_map = {
        "player_id": "sp_id",
        "year_id": "year",
        "team_name_abbr": "team",
        "p_earned_run_avg": "sp_era",
        "p_fip": "sp_fip",
        "p_whip": "sp_whip",
        "p_so_per_nine": "sp_k9",
        "p_bb_per_nine": "sp_bb9",
        "p_hr_per_nine": "sp_hr9",
        "p_ip": "sp_ip",
        "p_war": "sp_war",
        "p_w": "sp_wins",
        "p_l": "sp_losses",
        "p_strikeouts_per_base_on_balls": "sp_k_bb",
        "p_earned_run_avg_plus": "sp_era_plus",
        "p_gs": "sp_gs",
    }

    available = [c for c in cols_map if c in pitchers.columns]
    pdf = pitchers[available].rename(columns={c: cols_map[c] for c in available}).copy()

    # Convert types
    for col in pdf.columns:
        if col not in ("sp_id", "team", "year"):
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    if "year" in pdf.columns:
        pdf["year"] = pdf["year"].astype(int, errors="ignore")

    return pdf


def build_matchup_features(games: pd.DataFrame, rolling_stats: pd.DataFrame,
                           pitcher_features: pd.DataFrame) -> pd.DataFrame:
    """Merge team rolling stats and pitcher stats into game-level matchup features.

    Then compute diff/ratio features between home and away.
    """
    # Merge home team rolling stats
    home_stats = rolling_stats.copy()
    home_cols = {c: f"home_{c}" for c in home_stats.columns if c not in ("date", "team")}
    home_stats = home_stats.rename(columns=home_cols)
    home_stats = home_stats.rename(columns={"team": "home_team"})

    games = games.merge(home_stats, on=["date", "home_team"], how="left")

    # Merge away team rolling stats
    away_stats = rolling_stats.copy()
    away_cols = {c: f"away_{c}" for c in away_stats.columns if c not in ("date", "team")}
    away_stats = away_stats.rename(columns=away_cols)
    away_stats = away_stats.rename(columns={"team": "away_team"})

    games = games.merge(away_stats, on=["date", "away_team"], how="left")

    # Merge starting pitcher features (by player_id/URL matching)
    # This requires the games.csv to have sp_id or sp_url columns
    # For now, we merge by team + year as a fallback
    if not pitcher_features.empty and "home_sp" in games.columns:
        # TODO: Implement direct pitcher-to-game matching via player ID
        # For MVP, use team's aggregate pitching stats from rolling windows
        pass

    # Build diff features (home - away)
    stat_cols = [c for c in games.columns if c.startswith("home_") and c != "home_team"
                 and not c.endswith("_team") and not c.startswith("home_sp")]

    for home_col in stat_cols:
        suffix = home_col.replace("home_", "")
        away_col = f"away_{suffix}"
        if away_col in games.columns:
            games[f"diff_{suffix}"] = games[home_col] - games[away_col]

            # Ratio features (avoid division by zero)
            games[f"ratio_{suffix}"] = games.apply(
                lambda r: r[home_col] / r[away_col]
                if pd.notna(r[away_col]) and r[away_col] != 0
                else 1.0,
                axis=1
            )

    # Contextual features
    games["day_of_week"] = games["date"].dt.dayofweek
    games["month"] = games["date"].dt.month
    games["is_weekend"] = games["day_of_week"].isin([5, 6]).astype(int)

    return games


def create_ml_ready(master: pd.DataFrame) -> pd.DataFrame:
    """Select only ML features + target + date for training."""
    feature_cols = [c for c in master.columns
                    if c.startswith(("diff_", "ratio_"))
                    or c in ("day_of_week", "month", "is_weekend")]

    target_col = "home_win"
    date_col = "date"

    available = [c for c in feature_cols if c in master.columns]
    ml_df = master[available + [target_col, date_col]].copy()

    # Drop rows with too many NaN features
    thresh = len(available) * 0.5  # need at least 50% of features
    ml_df = ml_df.dropna(thresh=int(thresh) + 2)  # +2 for target and date

    logger.info(f"ML-ready dataset: {len(ml_df)} rows, {len(available)} features")
    return ml_df


def main():
    # Step 1: Load raw data
    games = load_games()
    if games.empty:
        logger.error("No games data. Run scraper/scrape_games.py first.")
        return

    pitchers = load_pitchers()

    # Step 2: Compute rolling team stats
    logger.info("Computing rolling team statistics...")
    rolling_stats = compute_rolling_team_stats(games)

    # Step 3: Build pitcher features
    pitcher_features = build_pitcher_features(pitchers)

    # Step 4: Build matchup features
    logger.info("Building matchup features...")
    master = build_matchup_features(games, rolling_stats, pitcher_features)

    # Step 5: Save master
    master_path = DATA_PROCESSED / "master.csv"
    master.to_csv(master_path, index=False)
    logger.info(f"Saved master.csv: {master.shape}")

    # Step 6: Create ML-ready dataset
    ml_ready = create_ml_ready(master)
    ml_path = DATA_PROCESSED / "ml_ready.csv"
    ml_ready.to_csv(ml_path, index=False)
    logger.info(f"Saved ml_ready.csv: {ml_ready.shape}")


if __name__ == "__main__":
    main()
