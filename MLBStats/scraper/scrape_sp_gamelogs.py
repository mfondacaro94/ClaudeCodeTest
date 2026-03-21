"""Scrape starting pitcher game logs from baseball-reference.com.

For each pitcher with at least 1 game started (GS > 0), scrape their
game log page to get per-game pitching stats. This lets us match
specific starting pitchers to specific games.

Output: data/raw/sp_gamelogs.csv
"""

import sys
import re
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import parse_br_table, safe_float, normalize_team_name

logger = get_logger("scrape_sp_gamelogs")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025


def get_starters(pitchers_path: Path) -> list[dict]:
    """Get list of pitchers who started at least 1 game."""
    df = pd.read_csv(pitchers_path)
    df["p_gs"] = pd.to_numeric(df["p_gs"], errors="coerce").fillna(0)
    starters = df[df["p_gs"] > 0][["player_id", "player_url", "year_id", "team_name_abbr", "p_gs"]].copy()
    starters = starters.drop_duplicates(subset=["player_id", "year_id"])

    result = []
    for _, row in starters.iterrows():
        result.append({
            "player_id": row["player_id"],
            "player_url": row["player_url"],
            "year": int(row["year_id"]),
        })
    return result


def scrape_pitcher_gamelog(session: RateLimitedSession, player_id: str,
                           player_url: str, year: int) -> list[dict]:
    """Scrape a pitcher's game log for a specific year."""
    url = f"{BASE_URL}/players/gl.fcgi?id={player_id}&t=p&year={year}"

    try:
        resp = session.get(url)
    except Exception as e:
        logger.warning(f"Failed to fetch game log {player_id} {year}: {e}")
        return []

    df = parse_br_table(resp.text, "players_standard_pitching")
    if df.empty:
        return []

    rows = []
    for _, row in df.iterrows():
        # Only keep game starts (GS indicated by "GS" in p_player_game_span)
        game_span = str(row.get("p_player_game_span", ""))
        if not game_span.startswith("GS"):
            continue

        date_str = row.get("date", "")
        if not date_str:
            continue

        # Extract the boxscore URL to get the date reliably
        date_url = row.get("date_url", "")
        # Parse date from the text (format: "2024-06-19")
        date_clean = str(date_str).strip()

        # Determine home/away
        location = row.get("game_location", "")
        is_home = location != "@"

        opp = normalize_team_name(str(row.get("opp_name_abbr", "")).strip())
        team = normalize_team_name(str(row.get("team_name_abbr", "")).strip())

        rows.append({
            "date": date_clean,
            "year": year,
            "player_id": player_id,
            "team": team,
            "opp": opp,
            "is_home": is_home,
            "home_team": team if is_home else opp,
            "away_team": opp if is_home else team,
            "ip": safe_float(row.get("p_ip"), 0),
            "h": safe_float(row.get("p_h"), 0),
            "r": safe_float(row.get("p_r"), 0),
            "er": safe_float(row.get("p_er"), 0),
            "bb": safe_float(row.get("p_bb"), 0),
            "so": safe_float(row.get("p_so"), 0),
            "hr": safe_float(row.get("p_hr"), 0),
            "pitches": safe_float(row.get("p_pitches"), 0),
            "game_score": safe_float(row.get("p_game_score"), 0),
            "days_rest": safe_float(row.get("p_days_rest"), 0),
            "era_cume": safe_float(row.get("p_earned_run_avg_cume")),
            "fip_cume": safe_float(row.get("p_fip_cume")),
            "decision": str(row.get("p_game_decision", "")).strip(),
        })

    return rows


def main():
    session = RateLimitedSession(delay=6.0)
    progress_path = DATA_RAW / "sp_gamelogs_progress.ndjson"
    progress = load_progress(progress_path)
    scraped_keys = {f"{r['player_id']}_{r['year']}" for r in progress}

    # Get all starters from pitcher data
    pitchers_path = DATA_RAW / "pitchers.csv"
    starters = get_starters(pitchers_path)

    # Deduplicate by (player_id, year)
    unique = {}
    for s in starters:
        key = f"{s['player_id']}_{s['year']}"
        if key not in unique and key not in scraped_keys:
            unique[key] = s

    logger.info(f"Found {len(unique)} pitcher-years to scrape ({len(scraped_keys)} already done)")

    # Scrape
    output_path = DATA_RAW / "sp_gamelogs.csv"
    batch = []
    count = 0

    for key, info in unique.items():
        rows = scrape_pitcher_gamelog(session, info["player_id"], info.get("player_url", ""), info["year"])
        batch.extend(rows)
        count += 1

        progress.append({"player_id": info["player_id"], "year": info["year"], "starts": len(rows)})

        # Save every 50 pitchers
        if count % 50 == 0:
            save_progress(progress_path, progress)
            if batch:
                batch_df = pd.DataFrame(batch)
                if output_path.exists() and output_path.stat().st_size > 0:
                    existing = pd.read_csv(output_path)
                    batch_df = pd.concat([existing, batch_df], ignore_index=True)
                    batch_df = batch_df.drop_duplicates(subset=["date", "player_id", "home_team"], keep="last")
                batch_df.to_csv(output_path, index=False)
                logger.info(f"Scraped {count}/{len(unique)} pitcher-years — saved {len(batch_df)} game starts")
                batch = []

    # Final save
    save_progress(progress_path, progress)
    if batch:
        batch_df = pd.DataFrame(batch)
        if output_path.exists() and output_path.stat().st_size > 0:
            existing = pd.read_csv(output_path)
            batch_df = pd.concat([existing, batch_df], ignore_index=True)
            batch_df = batch_df.drop_duplicates(subset=["date", "player_id", "home_team"], keep="last")
        batch_df.to_csv(output_path, index=False)

    total = len(pd.read_csv(output_path)) if output_path.exists() and output_path.stat().st_size > 0 else 0
    logger.info(f"Saved {total} SP game log rows to {output_path}")


if __name__ == "__main__":
    main()
