"""Scrape ALL pitcher game logs from baseball-reference.com.

Scrapes every pitcher's game log (starters AND relievers) to get:
- Starting pitcher identification per game (#1: SP matchup)
- Reliever usage per game (#2: bullpen availability)
- Pitcher handedness from name suffix (* = lefty) (#3: platoon splits)

Saves every 25 pitcher-years to CSV to avoid data loss on interruption.
Output: data/raw/pitcher_gamelogs.csv
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import parse_br_table, safe_float, normalize_team_name

logger = get_logger("scrape_pitcher_gamelogs")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025


def get_all_pitchers(pitchers_path: Path) -> list[dict]:
    """Get all pitcher-year combinations to scrape."""
    df = pd.read_csv(pitchers_path)
    # Filter out multi-team aggregates (2TM, 3TM, etc.)
    df = df[~df["team_name_abbr"].str.contains("TM", na=False)]
    df = df.drop_duplicates(subset=["player_id", "year_id"])

    result = []
    for _, row in df.iterrows():
        result.append({
            "player_id": row["player_id"],
            "player_url": row.get("player_url", ""),
            "year": int(row["year_id"]),
        })
    return result


def scrape_pitcher_gamelog(session: RateLimitedSession, player_id: str, year: int) -> list[dict]:
    """Scrape a pitcher's full game log for a specific year.

    Returns ALL appearances (starts and relief), not just starts.
    """
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
        date_str = str(row.get("date", "")).strip()
        if not date_str or len(date_str) < 8:
            continue

        game_span = str(row.get("p_player_game_span", ""))
        is_start = game_span.startswith("GS")

        # Determine home/away
        location = row.get("game_location", "")
        is_home = location != "@"

        opp = normalize_team_name(str(row.get("opp_name_abbr", "")).strip())
        team = normalize_team_name(str(row.get("team_name_abbr", "")).strip())

        # Extract player name (for handedness — * suffix = lefty)
        player_name = str(row.get("player", "")).strip()

        rows.append({
            "date": date_str,
            "year": year,
            "player_id": player_id,
            "team": team,
            "opp": opp,
            "is_home": is_home,
            "home_team": team if is_home else opp,
            "away_team": opp if is_home else team,
            "is_start": is_start,
            "game_span": game_span,
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


def flush_batch(batch: list[dict], output_path: Path, progress: list[dict], progress_path: Path, count: int, total: int):
    """Save current batch to CSV and progress to ndjson."""
    save_progress(progress_path, progress)
    if not batch:
        return
    batch_df = pd.DataFrame(batch)
    if output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path)
        batch_df = pd.concat([existing, batch_df], ignore_index=True)
        batch_df = batch_df.drop_duplicates(subset=["date", "player_id", "home_team", "is_start"], keep="last")
    batch_df.to_csv(output_path, index=False)
    logger.info(f"Scraped {count}/{total} pitcher-years — saved {len(batch_df)} appearances to CSV")


def main():
    session = RateLimitedSession(delay=6.0)
    progress_path = DATA_RAW / "pitcher_gamelogs_progress.ndjson"
    progress = load_progress(progress_path)
    scraped_keys = {f"{r['player_id']}_{r['year']}" for r in progress}

    # Get ALL pitchers (not just starters)
    pitchers_path = DATA_RAW / "pitchers.csv"
    all_pitchers = get_all_pitchers(pitchers_path)

    # Deduplicate and filter already-scraped
    unique = {}
    for p in all_pitchers:
        key = f"{p['player_id']}_{p['year']}"
        if key not in unique and key not in scraped_keys:
            unique[key] = p

    logger.info(f"Found {len(unique)} pitcher-years to scrape ({len(scraped_keys)} already done)")

    # Scrape with frequent saves
    output_path = DATA_RAW / "pitcher_gamelogs.csv"
    batch = []
    count = 0

    for key, info in unique.items():
        rows = scrape_pitcher_gamelog(session, info["player_id"], info["year"])
        batch.extend(rows)
        count += 1

        progress.append({
            "player_id": info["player_id"],
            "year": info["year"],
            "appearances": len(rows),
        })

        # Save every 25 pitcher-years (conservative to avoid data loss)
        if count % 25 == 0:
            flush_batch(batch, output_path, progress, progress_path, count, len(unique))
            batch = []

    # Final save
    flush_batch(batch, output_path, progress, progress_path, count, len(unique))

    total = len(pd.read_csv(output_path)) if output_path.exists() and output_path.stat().st_size > 0 else 0
    logger.info(f"Complete. {total} total pitcher appearances saved to {output_path}")


if __name__ == "__main__":
    main()
