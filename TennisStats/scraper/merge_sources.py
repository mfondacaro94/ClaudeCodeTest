"""Merge tennis-data.co.uk (odds) with Sackmann (stats) data sources.

Strategy: TDU is the primary source (has real match dates + odds). We enrich it
with Sackmann's match-level stats (serve, return, etc.) and player bio data.

Matching: normalize player names, then join on (year, tournament_name_norm, round_norm, sorted_players).
Sackmann only has tourney_date (tournament start), not individual match dates, so we
cannot match on exact date.
"""

import sys
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from utils.helpers import get_logger, DATA_RAW
from utils.data_cleaning import normalize_player_name, strip_accents

logger = get_logger("merge_sources")


def norm_tournament(name: str) -> str:
    """Normalize tournament name for cross-source matching."""
    if not name or not isinstance(name, str):
        return ""
    s = strip_accents(name.strip().lower())
    # Remove common suffixes/prefixes
    s = re.sub(r"\b(atp|wta|masters|open|international|championship|cup)\b", "", s)
    s = re.sub(r"[^a-z0-9]", "", s)
    return s


def norm_round(r: str) -> str:
    """Normalize round name for cross-source matching."""
    if not r or not isinstance(r, str):
        return ""
    r = r.strip().lower()
    mapping = {
        "1st round": "r32", "2nd round": "r16", "3rd round": "r8",
        "4th round": "r4", "quarterfinals": "qf", "semifinals": "sf",
        "the final": "f", "final": "f", "round robin": "rr",
        "r128": "r128", "r64": "r64", "r32": "r32", "r16": "r16",
        "qf": "qf", "sf": "sf", "f": "f", "rr": "rr",
    }
    for key, val in mapping.items():
        if key in r:
            return val
    return r.replace(" ", "")


def build_player_key(name1: str, name2: str) -> str:
    """Sorted normalized player pair key."""
    n1 = normalize_player_name(name1)
    n2 = normalize_player_name(name2)
    return "|".join(sorted([n1, n2]))


def merge_datasets():
    """Merge tennis-data.co.uk odds with Sackmann match stats."""
    tdu_path = DATA_RAW / "matches_odds.csv"
    sack_path = DATA_RAW / "matches_stats.csv"

    if not tdu_path.exists():
        logger.error(f"Missing {tdu_path}. Run download_tennis_data_uk.py first.")
        return
    if not sack_path.exists():
        logger.error(f"Missing {sack_path}. Run download_sackmann.py first.")
        return

    tdu = pd.read_csv(tdu_path, low_memory=False)
    sack = pd.read_csv(sack_path, low_memory=False)
    logger.info(f"Loaded TDU: {len(tdu)} rows, Sackmann: {len(sack)} rows")

    # Parse dates
    tdu["date"] = pd.to_datetime(tdu["Date"], dayfirst=False, errors="coerce")
    tdu = tdu.dropna(subset=["date", "Winner", "Loser"])
    tdu["year"] = tdu["date"].dt.year

    sack["tourney_start"] = pd.to_datetime(sack["tourney_date"].astype(str), format="%Y%m%d", errors="coerce")
    sack = sack.dropna(subset=["tourney_start", "winner_name", "loser_name"])
    sack["year"] = sack["tourney_start"].dt.year

    # Build match keys
    # Primary key: year + player pair (unique enough for most matches)
    tdu["player_key"] = tdu.apply(lambda r: build_player_key(r["Winner"], r["Loser"]), axis=1)
    sack["player_key"] = sack.apply(lambda r: build_player_key(r["winner_name"], r["loser_name"]), axis=1)

    # Secondary disambiguation: tournament name + round
    tdu["tourney_norm"] = tdu["Tournament"].apply(norm_tournament)
    sack["tourney_norm"] = sack["tourney_name"].apply(norm_tournament)

    tdu["round_norm"] = tdu["Round"].apply(norm_round)
    sack["round_norm"] = sack["round"].apply(norm_round)

    # Composite key: year + player_key (most unique)
    tdu["merge_key"] = tdu["year"].astype(str) + "|" + tdu["player_key"]
    sack["merge_key"] = sack["year"].astype(str) + "|" + sack["player_key"]

    # Check for duplicates in merge keys
    tdu_dupes = tdu["merge_key"].duplicated(keep=False).sum()
    sack_dupes = sack["merge_key"].duplicated(keep=False).sum()
    logger.info(f"TDU duplicate merge keys: {tdu_dupes}/{len(tdu)}")
    logger.info(f"Sackmann duplicate merge keys: {sack_dupes}/{len(sack)}")

    # For matches with duplicate merge keys, add round to disambiguate
    tdu["merge_key_full"] = tdu["merge_key"] + "|" + tdu["round_norm"]
    sack["merge_key_full"] = sack["merge_key"] + "|" + sack["round_norm"]

    # Sackmann stat columns to bring over
    stat_cols = [
        "winner_id", "loser_id", "winner_hand", "loser_hand",
        "winner_ht", "loser_ht", "winner_age", "loser_age",
        "winner_ioc", "loser_ioc",
        "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
        "w_SvGms", "w_bpSaved", "w_bpFaced",
        "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
        "l_SvGms", "l_bpSaved", "l_bpFaced",
        "minutes", "tourney_level", "draw_size",
        "winner_seed", "loser_seed",
    ]
    available_stats = [c for c in stat_cols if c in sack.columns]

    # Step 1: Try full key match (year + players + round)
    sack_full = sack.drop_duplicates(subset="merge_key_full", keep="first")
    sack_lookup_full = sack_full.set_index("merge_key_full")[available_stats]

    matched_full = tdu["merge_key_full"].isin(sack_lookup_full.index)
    logger.info(f"Full key match (year+players+round): {matched_full.sum()}/{len(tdu)}")

    # Step 2: For unmatched, try simple key (year + players only)
    sack_simple = sack.drop_duplicates(subset="merge_key", keep="first")
    sack_lookup_simple = sack_simple.set_index("merge_key")[available_stats]

    # Merge: prefer full key, fall back to simple key
    for col in available_stats:
        tdu[col] = np.nan

    # Full key matches
    full_matches = tdu.loc[matched_full, "merge_key_full"]
    for col in available_stats:
        tdu.loc[matched_full, col] = full_matches.map(sack_lookup_full[col])

    # Simple key matches for remaining
    unmatched = tdu[col].isna()
    simple_matches = tdu.loc[unmatched, "merge_key"]
    matched_simple = simple_matches.isin(sack_lookup_simple.index)
    for col in available_stats:
        mask = unmatched & tdu["merge_key"].isin(sack_lookup_simple.index)
        tdu.loc[mask, col] = tdu.loc[mask, "merge_key"].map(sack_lookup_simple[col])

    total_matched = tdu["winner_id"].notna().sum()
    logger.info(f"Total matched: {total_matched}/{len(tdu)} ({total_matched/len(tdu)*100:.1f}%)")

    # Count bookmaker odds per match
    book_pairs = [("B365W", "B365L"), ("PSW", "PSL"), ("EXW", "EXL"),
                  ("LBW", "LBL"), ("SJW", "SJL")]
    tdu["n_books"] = 0
    for w_col, l_col in book_pairs:
        if w_col in tdu.columns and l_col in tdu.columns:
            tdu["n_books"] += (tdu[w_col].notna() & tdu[l_col].notna()).astype(int)

    # Clean up temp columns
    drop_cols = ["player_key", "tourney_norm", "round_norm", "merge_key", "merge_key_full", "year"]
    tdu = tdu.drop(columns=[c for c in drop_cols if c in tdu.columns])

    # Save
    out_path = DATA_RAW / "matches_merged.csv"
    tdu.to_csv(out_path, index=False)
    logger.info(f"Saved merged data: {len(tdu)} rows -> {out_path.name}")

    # Summary stats
    logger.info(f"Date range: {tdu['date'].min()} to {tdu['date'].max()}")
    logger.info(f"Matches with Pinnacle odds: {tdu['PSW'].notna().sum()}")
    logger.info(f"Matches with 3+ books: {(tdu['n_books'] >= 3).sum()}")
    logger.info(f"Matches with serve stats: {tdu['w_ace'].notna().sum()}")
    logger.info(f"Matches with player bio: {tdu['winner_ioc'].notna().sum()}")


def main():
    merge_datasets()


if __name__ == "__main__":
    main()
