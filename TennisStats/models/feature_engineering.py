"""Feature engineering: raw merged data -> master.csv -> ml_ready.csv.

Computes ELO ratings, rolling stats, H2H, fatigue, serve/return stats,
distance-to-home, and matchup diff/ratio features. All with shift(1) to
prevent look-ahead bias.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from collections import defaultdict

from utils.helpers import get_logger, DATA_RAW, DATA_PROCESSED
from utils.data_cleaning import (
    normalize_player_name, compute_distance_to_home, safe_float,
    COUNTRY_COORDS,
)

logger = get_logger("feature_engineering")

ROLLING_WINDOWS = [5, 10, 20]


# ---------------------------------------------------------------------------
# 1. ELO Rating System
# ---------------------------------------------------------------------------

def compute_elo_ratings(matches: pd.DataFrame) -> dict:
    """Compute overall ELO + surface-specific ELO for all players.

    Returns dict mapping (player_name_norm, date_str, match_idx) -> {
        'elo_before': float,
        'surface_elo_before': float,
    }
    """
    elo = defaultdict(lambda: 1500.0)           # overall ELO
    surface_elo = defaultdict(lambda: 1500.0)    # keyed by (player, surface)
    match_count = defaultdict(int)               # for K-factor

    elo_records = {}

    for idx, row in matches.iterrows():
        w = row["winner_norm"]
        l = row["loser_norm"]
        surface = row.get("Surface", "Hard")
        if pd.isna(surface):
            surface = "Hard"

        # K-factor based on experience
        def get_k(player):
            n = match_count[player]
            if n < 50:
                return 32
            elif n < 200:
                return 24
            return 16

        # Pre-match ratings
        w_elo = elo[w]
        l_elo = elo[l]
        w_s_elo = surface_elo[(w, surface)]
        l_s_elo = surface_elo[(l, surface)]

        elo_records[(idx, "winner")] = {
            "elo_before": w_elo,
            "surface_elo_before": w_s_elo,
        }
        elo_records[(idx, "loser")] = {
            "elo_before": l_elo,
            "surface_elo_before": l_s_elo,
        }

        # Expected scores
        e_w = 1.0 / (1.0 + 10 ** ((l_elo - w_elo) / 400))
        e_l = 1.0 - e_w

        e_w_s = 1.0 / (1.0 + 10 ** ((l_s_elo - w_s_elo) / 400))
        e_l_s = 1.0 - e_w_s

        # Update ratings
        k_w = get_k(w)
        k_l = get_k(l)

        elo[w] = w_elo + k_w * (1.0 - e_w)
        elo[l] = l_elo + k_l * (0.0 - e_l)

        surface_elo[(w, surface)] = w_s_elo + k_w * (1.0 - e_w_s)
        surface_elo[(l, surface)] = l_s_elo + k_l * (0.0 - e_l_s)

        match_count[w] += 1
        match_count[l] += 1

    return elo_records


# ---------------------------------------------------------------------------
# 2. Rolling Stats per Player
# ---------------------------------------------------------------------------

def build_player_match_log(matches: pd.DataFrame) -> pd.DataFrame:
    """Create a unified per-player match log from winner/loser rows.

    Each row = one player's perspective on one match.
    """
    rows = []

    for idx, r in matches.iterrows():
        base = {
            "match_idx": idx,
            "date": r["date"],
            "surface": r.get("Surface", "Hard"),
            "tournament": r.get("Tournament", ""),
            "round": r.get("Round", ""),
            "best_of": r.get("Best of", 3),
            "n_sets": safe_float(r.get("Wsets", 0), 0) + safe_float(r.get("Lsets", 0), 0),
        }

        # Winner perspective
        winner_row = {
            **base,
            "player": r["winner_norm"],
            "opponent": r["loser_norm"],
            "won": 1,
            "ioc": r.get("winner_ioc", ""),
            "ace": safe_float(r.get("w_ace"), None),
            "df": safe_float(r.get("w_df"), None),
            "svpt": safe_float(r.get("w_svpt"), None),
            "first_in": safe_float(r.get("w_1stIn"), None),
            "first_won": safe_float(r.get("w_1stWon"), None),
            "second_won": safe_float(r.get("w_2ndWon"), None),
            "sv_gms": safe_float(r.get("w_SvGms"), None),
            "bp_saved": safe_float(r.get("w_bpSaved"), None),
            "bp_faced": safe_float(r.get("w_bpFaced"), None),
            "opp_bp_saved": safe_float(r.get("l_bpSaved"), None),
            "opp_bp_faced": safe_float(r.get("l_bpFaced"), None),
            "opp_svpt": safe_float(r.get("l_svpt"), None),
            "opp_first_in": safe_float(r.get("l_1stIn"), None),
            "opp_first_won": safe_float(r.get("l_1stWon"), None),
            "opp_second_won": safe_float(r.get("l_2ndWon"), None),
        }
        rows.append(winner_row)

        # Loser perspective
        loser_row = {
            **base,
            "player": r["loser_norm"],
            "opponent": r["winner_norm"],
            "won": 0,
            "ioc": r.get("loser_ioc", ""),
            "ace": safe_float(r.get("l_ace"), None),
            "df": safe_float(r.get("l_df"), None),
            "svpt": safe_float(r.get("l_svpt"), None),
            "first_in": safe_float(r.get("l_1stIn"), None),
            "first_won": safe_float(r.get("l_1stWon"), None),
            "second_won": safe_float(r.get("l_2ndWon"), None),
            "sv_gms": safe_float(r.get("l_SvGms"), None),
            "bp_saved": safe_float(r.get("l_bpSaved"), None),
            "bp_faced": safe_float(r.get("l_bpFaced"), None),
            "opp_bp_saved": safe_float(r.get("w_bpSaved"), None),
            "opp_bp_faced": safe_float(r.get("w_bpFaced"), None),
            "opp_svpt": safe_float(r.get("w_svpt"), None),
            "opp_first_in": safe_float(r.get("w_1stIn"), None),
            "opp_first_won": safe_float(r.get("w_1stWon"), None),
            "opp_second_won": safe_float(r.get("w_2ndWon"), None),
        }
        rows.append(loser_row)

    plog = pd.DataFrame(rows)
    plog["date"] = pd.to_datetime(plog["date"])
    plog = plog.sort_values(["player", "date", "match_idx"]).reset_index(drop=True)

    # Derived serve/return rates
    plog["ace_rate"] = plog["ace"] / plog["sv_gms"].replace(0, np.nan)
    plog["df_rate"] = plog["df"] / plog["sv_gms"].replace(0, np.nan)
    plog["first_serve_pct"] = plog["first_in"] / plog["svpt"].replace(0, np.nan)
    plog["first_serve_won_pct"] = plog["first_won"] / plog["first_in"].replace(0, np.nan)
    plog["second_serve_won_pct"] = plog["second_won"] / (plog["svpt"] - plog["first_in"]).replace(0, np.nan)
    plog["bp_save_pct"] = plog["bp_saved"] / plog["bp_faced"].replace(0, np.nan)

    # Return stats (from opponent's serve data)
    plog["bp_convert_pct"] = (plog["opp_bp_faced"] - plog["opp_bp_saved"]) / plog["opp_bp_faced"].replace(0, np.nan)
    opp_return_pts = plog["opp_svpt"] - plog["opp_first_won"] - plog["opp_second_won"]
    plog["return_pts_won_pct"] = opp_return_pts / plog["opp_svpt"].replace(0, np.nan)

    return plog


def compute_rolling_features(plog: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling stats for each player. Uses shift(1) to prevent leakage."""
    stat_cols = [
        "won", "ace_rate", "df_rate", "first_serve_pct",
        "first_serve_won_pct", "second_serve_won_pct",
        "bp_save_pct", "bp_convert_pct", "return_pts_won_pct",
    ]

    grouped = plog.groupby("player")

    for window in ROLLING_WINDOWS:
        for col in stat_cols:
            col_name = f"roll{window}_{col}"
            plog[col_name] = grouped[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=3).mean()
            )

    # Career cumulative win rate (shifted)
    plog["career_win_rate"] = grouped["won"].transform(
        lambda x: x.shift(1).expanding(min_periods=5).mean()
    )

    # Career match count (shifted)
    plog["career_matches"] = grouped["won"].transform(
        lambda x: x.shift(1).expanding().count()
    )

    # --- V2 features: streaks and momentum ---

    # Current win/loss streak (shifted)
    def compute_streak(series):
        """Compute current streak length. Positive = winning, negative = losing."""
        shifted = series.shift(1)
        streaks = []
        current = 0
        for val in shifted:
            if pd.isna(val):
                streaks.append(0)
                continue
            if val == 1:
                current = current + 1 if current > 0 else 1
            else:
                current = current - 1 if current < 0 else -1
            streaks.append(current)
        return pd.Series(streaks, index=series.index)

    plog["win_streak"] = grouped["won"].transform(compute_streak)

    # Win rate momentum: last 5 minus last 20 (form vs baseline)
    plog["form_momentum"] = plog.get("roll5_won", 0) - plog.get("roll20_won", 0)

    # Upset rate: how often player beats higher-ranked opponents (shifted)
    # We'll compute this later when we have rank info

    return plog


def compute_surface_rolling(plog: pd.DataFrame) -> pd.DataFrame:
    """Compute surface-specific rolling stats."""
    grouped = plog.groupby(["player", "surface"])

    for window in [10, 20]:
        plog[f"surface_win_rate_{window}"] = grouped["won"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=3).mean()
        )

    plog["surface_career_win_rate"] = grouped["won"].transform(
        lambda x: x.shift(1).expanding(min_periods=3).mean()
    )

    return plog


def compute_fatigue_features(plog: pd.DataFrame) -> pd.DataFrame:
    """Compute fatigue: days since last match, recent match/set load."""
    grouped = plog.groupby("player")

    # Days since last match
    plog["prev_match_date"] = grouped["date"].shift(1)
    plog["days_since_last"] = (plog["date"] - plog["prev_match_date"]).dt.days

    # Matches and sets in last 7 and 14 days
    plog["matches_last_7d"] = 0
    plog["matches_last_14d"] = 0
    plog["sets_last_7d"] = 0.0
    plog["sets_last_14d"] = 0.0

    for player, group in grouped:
        dates = group["date"].values
        sets = group["n_sets"].values
        idxs = group.index.values

        for i in range(len(group)):
            d = dates[i]
            m7 = m14 = 0
            s7 = s14 = 0.0
            for j in range(i - 1, max(-1, i - 30), -1):
                diff = (d - dates[j]) / np.timedelta64(1, "D")
                if diff <= 0:
                    continue
                if diff <= 7:
                    m7 += 1
                    s7 += sets[j] if not np.isnan(sets[j]) else 0
                if diff <= 14:
                    m14 += 1
                    s14 += sets[j] if not np.isnan(sets[j]) else 0
                if diff > 14:
                    break

            plog.loc[idxs[i], "matches_last_7d"] = m7
            plog.loc[idxs[i], "matches_last_14d"] = m14
            plog.loc[idxs[i], "sets_last_7d"] = s7
            plog.loc[idxs[i], "sets_last_14d"] = s14

    return plog


def compute_h2h(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute head-to-head records before each match."""
    h2h_wins = defaultdict(lambda: defaultdict(int))
    h2h_surface_wins = defaultdict(lambda: defaultdict(int))

    h2h_records = {}

    for idx, row in matches.iterrows():
        w = row["winner_norm"]
        l = row["loser_norm"]
        surface = row.get("Surface", "Hard")
        if pd.isna(surface):
            surface = "Hard"
        pair = tuple(sorted([w, l]))

        # Record BEFORE this match
        h2h_records[idx] = {
            "h2h_w_wins": h2h_wins[pair][w],
            "h2h_l_wins": h2h_wins[pair][l],
            "h2h_total": h2h_wins[pair][w] + h2h_wins[pair][l],
            "h2h_surface_w_wins": h2h_surface_wins[(pair, surface)][w],
            "h2h_surface_l_wins": h2h_surface_wins[(pair, surface)][l],
        }

        # Update after
        h2h_wins[pair][w] += 1
        h2h_surface_wins[(pair, surface)][w] += 1

    return h2h_records


def map_round_number(round_str: str) -> int:
    """Map round string to numeric value (higher = deeper in tournament)."""
    if not round_str or not isinstance(round_str, str):
        return 0
    r = round_str.strip().lower()
    mapping = {
        "1st round": 1, "r128": 1, "round of 128": 1,
        "2nd round": 2, "r64": 2, "round of 64": 2,
        "3rd round": 3, "r32": 3, "round of 32": 3,
        "4th round": 4, "r16": 4, "round of 16": 4,
        "quarterfinals": 5, "qf": 5,
        "semifinals": 6, "sf": 6,
        "the final": 7, "final": 7, "f": 7,
        "round robin": 4, "rr": 4,
    }
    for key, val in mapping.items():
        if key in r:
            return val
    return 0


def map_tournament_level(series: str) -> int:
    """Map tournament series/level to ordinal."""
    if not series or not isinstance(series, str):
        return 1
    s = series.strip().lower()
    if "grand slam" in s:
        return 4
    if "masters" in s or "1000" in s:
        return 3
    if "500" in s or "international gold" in s:
        return 2
    return 1  # ATP 250 / International


def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # Load merged data
    merged_path = DATA_RAW / "matches_merged.csv"
    if not merged_path.exists():
        logger.error("Missing matches_merged.csv. Run scraper/merge_sources.py first.")
        return

    df = pd.read_csv(merged_path)
    logger.info(f"Loaded {len(df)} matches")

    # Parse date
    df["date"] = pd.to_datetime(df["Date"], dayfirst=False, errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # Filter: completed matches only, no walkovers
    if "Comment" in df.columns:
        df = df[~df["Comment"].str.contains("Walkover|W/O", case=False, na=False)]
    logger.info(f"After filtering walkovers: {len(df)} matches")

    # Normalize names
    df["winner_norm"] = df["Winner"].apply(normalize_player_name)
    df["loser_norm"] = df["Loser"].apply(normalize_player_name)

    # --- Compute ELO ratings ---
    logger.info("Computing ELO ratings...")
    elo_records = compute_elo_ratings(df)

    df["w_elo"] = df.index.map(lambda i: elo_records.get((i, "winner"), {}).get("elo_before", 1500))
    df["l_elo"] = df.index.map(lambda i: elo_records.get((i, "loser"), {}).get("elo_before", 1500))
    df["w_surface_elo"] = df.index.map(lambda i: elo_records.get((i, "winner"), {}).get("surface_elo_before", 1500))
    df["l_surface_elo"] = df.index.map(lambda i: elo_records.get((i, "loser"), {}).get("surface_elo_before", 1500))

    # --- Compute H2H ---
    logger.info("Computing head-to-head records...")
    h2h_records = compute_h2h(df)
    h2h_df = pd.DataFrame.from_dict(h2h_records, orient="index")
    df = df.join(h2h_df)

    # --- Build player match log + rolling stats ---
    logger.info("Building player match log...")
    plog = build_player_match_log(df)

    logger.info("Computing rolling features...")
    plog = compute_rolling_features(plog)

    logger.info("Computing surface-specific rolling features...")
    plog = compute_surface_rolling(plog)

    logger.info("Computing fatigue features...")
    plog = compute_fatigue_features(plog)

    # --- Merge rolling stats back into matches ---
    logger.info("Merging player stats into match records...")

    # Pivot: for each match, get winner's and loser's stats
    w_stats = plog[plog["won"] == 1].set_index("match_idx")
    l_stats = plog[plog["won"] == 0].set_index("match_idx")

    # Deduplicate (keep first if any duplicates)
    w_stats = w_stats[~w_stats.index.duplicated(keep="first")]
    l_stats = l_stats[~l_stats.index.duplicated(keep="first")]

    stat_cols_to_merge = [c for c in plog.columns if c.startswith(("roll", "surface_", "career_"))
                          or c in ("days_since_last", "matches_last_7d", "matches_last_14d",
                                   "sets_last_7d", "sets_last_14d")]

    for col in stat_cols_to_merge:
        if col in w_stats.columns:
            df[f"w_{col}"] = df.index.map(lambda i: w_stats.loc[i, col] if i in w_stats.index else np.nan)
            df[f"l_{col}"] = df.index.map(lambda i: l_stats.loc[i, col] if i in l_stats.index else np.nan)

    # --- Tournament context features ---
    logger.info("Adding tournament context features...")
    df["round_number"] = df["Round"].apply(map_round_number)
    df["tournament_level"] = df.get("Series", pd.Series(dtype=str)).apply(map_tournament_level)
    df["best_of"] = pd.to_numeric(df.get("Best of", 3), errors="coerce").fillna(3).astype(int)

    # Indoor/outdoor
    if "Court" in df.columns:
        df["is_indoor"] = (df["Court"].str.lower().str.contains("indoor", na=False)).astype(int)
    else:
        df["is_indoor"] = 0

    # Surface encoding
    df["surface_hard"] = (df["Surface"].str.lower() == "hard").astype(int)
    df["surface_clay"] = (df["Surface"].str.lower() == "clay").astype(int)
    df["surface_grass"] = (df["Surface"].str.lower() == "grass").astype(int)

    # --- Player bio features ---
    if "winner_age" in df.columns:
        df["w_age"] = pd.to_numeric(df["winner_age"], errors="coerce")
        df["l_age"] = pd.to_numeric(df["loser_age"], errors="coerce")
    if "winner_ht" in df.columns:
        df["w_height"] = pd.to_numeric(df["winner_ht"], errors="coerce")
        df["l_height"] = pd.to_numeric(df["loser_ht"], errors="coerce")

    # Hand encoding (R=0, L=1)
    if "winner_hand" in df.columns:
        df["w_hand_left"] = (df["winner_hand"] == "L").astype(int)
        df["l_hand_left"] = (df["loser_hand"] == "L").astype(int)

    # Ranking
    df["w_rank"] = pd.to_numeric(df.get("WRank", np.nan), errors="coerce")
    df["l_rank"] = pd.to_numeric(df.get("LRank", np.nan), errors="coerce")
    df["w_rank_pts"] = pd.to_numeric(df.get("WPts", np.nan), errors="coerce")
    df["l_rank_pts"] = pd.to_numeric(df.get("LPts", np.nan), errors="coerce")

    # Seed
    df["w_seed"] = pd.to_numeric(df.get("winner_seed", np.nan), errors="coerce")
    df["l_seed"] = pd.to_numeric(df.get("loser_seed", np.nan), errors="coerce")

    # --- Distance to home ---
    logger.info("Computing distance to home...")
    tournament_col = "Tournament" if "Tournament" in df.columns else None
    location_col = "Location" if "Location" in df.columns else ""

    if tournament_col and "winner_ioc" in df.columns:
        df["w_distance_home"] = df.apply(
            lambda r: compute_distance_to_home(
                r.get("winner_ioc", ""),
                r.get(tournament_col, ""),
                r.get(location_col, "") if location_col else "",
            ),
            axis=1,
        )
        df["l_distance_home"] = df.apply(
            lambda r: compute_distance_to_home(
                r.get("loser_ioc", ""),
                r.get(tournament_col, ""),
                r.get(location_col, "") if location_col else "",
            ),
            axis=1,
        )
        df["w_is_home"] = (df.get("winner_ioc", "") == "").astype(int)  # placeholder
        df["l_is_home"] = (df.get("loser_ioc", "") == "").astype(int)

    # --- Balance: randomly assign P1/P2 ---
    logger.info("Balancing rows (random P1/P2 assignment)...")
    np.random.seed(42)
    flip = np.random.random(len(df)) < 0.5

    # Identify all w_ and l_ column pairs — EXCLUDE raw in-match stats (data leakage!)
    raw_match_stats = {
        "ace", "df", "svpt", "1stIn", "1stWon", "2ndWon",
        "SvGms", "bpSaved", "bpFaced",
    }
    w_cols = [c for c in df.columns
              if c.startswith("w_") and f"l_{c[2:]}" in df.columns
              and c[2:] not in raw_match_stats]

    # Create P1/P2 columns
    for w_col in w_cols:
        l_col = f"l_{w_col[2:]}"
        p1_col = f"p1_{w_col[2:]}"
        p2_col = f"p2_{w_col[2:]}"

        df[p1_col] = np.where(flip, df[l_col], df[w_col])
        df[p2_col] = np.where(flip, df[w_col], df[l_col])

    # H2H columns
    df["p1_h2h_wins"] = np.where(flip, df["h2h_l_wins"], df["h2h_w_wins"])
    df["p2_h2h_wins"] = np.where(flip, df["h2h_w_wins"], df["h2h_l_wins"])
    df["p1_h2h_surface_wins"] = np.where(flip, df["h2h_surface_l_wins"], df["h2h_surface_w_wins"])
    df["p2_h2h_surface_wins"] = np.where(flip, df["h2h_surface_w_wins"], df["h2h_surface_l_wins"])
    df["h2h_total_matches"] = df["h2h_total"]

    # Target
    df["p1_win"] = np.where(flip, 0, 1)

    # Player names for reference
    df["p1_name"] = np.where(flip, df["Loser"], df["Winner"])
    df["p2_name"] = np.where(flip, df["Winner"], df["Loser"])

    # Odds (keep in decimal format from tennis-data.co.uk)
    for odds_col in ["B365W", "B365L", "PSW", "PSL", "MaxW", "MaxL", "AvgW", "AvgL"]:
        if odds_col in df.columns:
            df[odds_col] = pd.to_numeric(df[odds_col], errors="coerce")

    # Map odds to P1/P2
    if "PSW" in df.columns:
        df["p1_odds_ps"] = np.where(flip, df["PSL"], df["PSW"])
        df["p2_odds_ps"] = np.where(flip, df["PSW"], df["PSL"])
    if "MaxW" in df.columns:
        df["p1_odds_max"] = np.where(flip, df["MaxL"], df["MaxW"])
        df["p2_odds_max"] = np.where(flip, df["MaxW"], df["MaxL"])
    if "AvgW" in df.columns:
        df["p1_odds_avg"] = np.where(flip, df["AvgL"], df["AvgW"])
        df["p2_odds_avg"] = np.where(flip, df["AvgW"], df["AvgL"])
    if "B365W" in df.columns:
        df["p1_odds_b365"] = np.where(flip, df["B365L"], df["B365W"])
        df["p2_odds_b365"] = np.where(flip, df["B365W"], df["B365L"])

    # Cap odds outliers at 50.0 decimal
    for col in [c for c in df.columns if "odds" in c.lower()]:
        df[col] = df[col].clip(upper=50.0)

    # --- V2: Market-aware features (Pinnacle implied probability) ---
    logger.info("Computing market-aware features...")
    if "p1_odds_ps" in df.columns:
        p1_ip = 1 / df["p1_odds_ps"]
        p2_ip = 1 / df["p2_odds_ps"]
        total_ip = p1_ip + p2_ip
        # No-vig implied probability (what Pinnacle really thinks)
        df["pinnacle_p1_prob"] = p1_ip / total_ip
        df["pinnacle_p1_prob"] = df["pinnacle_p1_prob"].fillna(0.5)
    else:
        df["pinnacle_p1_prob"] = 0.5

    # Avg market implied probability (captures consensus)
    if "p1_odds_avg" in df.columns:
        p1_avg_ip = 1 / df["p1_odds_avg"]
        p2_avg_ip = 1 / df["p2_odds_avg"]
        total_avg = p1_avg_ip + p2_avg_ip
        df["market_avg_p1_prob"] = p1_avg_ip / total_avg
        df["market_avg_p1_prob"] = df["market_avg_p1_prob"].fillna(0.5)
    else:
        df["market_avg_p1_prob"] = 0.5

    # Line disagreement: Pinnacle vs market average (captures sharp vs public)
    df["line_disagreement"] = df["pinnacle_p1_prob"] - df.get("market_avg_p1_prob", 0.5)

    # --- V2: Ranking momentum (rank change over recent period) ---
    # Approximate from rolling win rate change
    # Already have form_momentum from player log, but we need rank-based momentum
    # Use WRank/LRank from TDU data for ranking trajectory
    logger.info("Computing ranking momentum...")
    if "WRank" in df.columns:
        df["w_rank_num"] = pd.to_numeric(df["WRank"], errors="coerce")
        df["l_rank_num"] = pd.to_numeric(df["LRank"], errors="coerce")

        # Per-player rolling rank (lower = better, so negative change = improvement)
        for prefix, rank_col in [("w_", "w_rank_num"), ("l_", "l_rank_num")]:
            # We need to track rank changes per player over time
            # Simple approach: compute rank at this match vs avg rank over last 10
            pass  # Handled via diff_rank already

    # --- Build diff/ratio matchup features ---
    logger.info("Building diff/ratio matchup features...")
    p1_cols = [c for c in df.columns if c.startswith("p1_") and c not in
               ("p1_win", "p1_name", "p1_odds_ps", "p1_odds_max", "p1_odds_avg", "p1_odds_b365")]
    excluded_suffixes = {"name", "win"}

    for p1_col in p1_cols:
        suffix = p1_col[3:]
        if suffix in excluded_suffixes:
            continue
        p2_col = f"p2_{suffix}"
        if p2_col not in df.columns:
            continue

        p1_vals = pd.to_numeric(df[p1_col], errors="coerce")
        p2_vals = pd.to_numeric(df[p2_col], errors="coerce")

        df[f"diff_{suffix}"] = p1_vals - p2_vals
        df[f"ratio_{suffix}"] = np.where(
            (p2_vals != 0) & p2_vals.notna(),
            p1_vals / p2_vals,
            1.0,
        )

    # Also diff/ratio for H2H
    df["diff_h2h_wins"] = df["p1_h2h_wins"] - df["p2_h2h_wins"]
    df["diff_h2h_surface_wins"] = df["p1_h2h_surface_wins"] - df["p2_h2h_surface_wins"]

    # Calendar features
    df["month"] = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek

    # --- Save master.csv ---
    master_path = DATA_PROCESSED / "master.csv"
    df.to_csv(master_path, index=False)
    logger.info(f"Saved master.csv: {len(df)} rows x {len(df.columns)} cols")

    # --- Create ml_ready.csv ---
    logger.info("Creating ml_ready.csv...")

    feature_cols = [c for c in df.columns if c.startswith(("diff_", "ratio_"))]
    # V2: market-aware features as training inputs
    market_cols = [
        "pinnacle_p1_prob", "market_avg_p1_prob", "line_disagreement",
    ]
    context_cols = [
        "round_number", "tournament_level", "best_of", "is_indoor",
        "surface_hard", "surface_clay", "surface_grass",
        "month", "day_of_week", "h2h_total_matches",
    ]
    id_cols = ["date", "p1_name", "p2_name", "p1_win"]
    odds_cols = [c for c in df.columns if c.startswith(("p1_odds_", "p2_odds_"))]
    extra = ["n_books"] if "n_books" in df.columns else []

    all_cols = (id_cols + feature_cols
                + [c for c in market_cols if c in df.columns]
                + [c for c in context_cols if c in df.columns]
                + odds_cols + extra)

    ml_df = df[[c for c in all_cols if c in df.columns]].copy()

    # Drop rows with too many missing features
    feature_set = feature_cols + [c for c in context_cols if c in ml_df.columns]
    available = [c for c in feature_set if c in ml_df.columns]
    thresh = int(len(available) * 0.6)  # need at least 60% of features
    ml_df = ml_df.dropna(subset=available, thresh=thresh)

    ml_df.to_csv(DATA_PROCESSED / "ml_ready.csv", index=False)
    logger.info(f"Saved ml_ready.csv: {len(ml_df)} rows x {len(ml_df.columns)} cols")
    logger.info(f"Features: {len(feature_cols)} diff/ratio + {len([c for c in context_cols if c in ml_df.columns])} context")
    logger.info(f"Date range: {ml_df['date'].min()} to {ml_df['date'].max()}")
    logger.info(f"P1 win rate: {ml_df['p1_win'].mean():.3f} (should be ~0.50)")


if __name__ == "__main__":
    main()
