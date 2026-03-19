"""Scrape individual pitcher stats from baseball-reference.com.

Discovery: finds pitchers via team roster pages, then scrapes each
pitcher's career/season stats from their player page.
Output: data/raw/pitchers.csv
"""

import sys
import re
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import parse_br_table, safe_float, normalize_team_name

logger = get_logger("scrape_pitchers")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025

TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

PITCHING_COLS = [
    "year_id", "age", "team_name_abbr", "comp_name_abbr",
    "p_war", "p_w", "p_l", "p_win_loss_perc", "p_earned_run_avg",
    "p_g", "p_gs", "p_gf", "p_cg", "p_sho", "p_sv", "p_ip",
    "p_h", "p_r", "p_er", "p_hr", "p_bb", "p_ibb", "p_so",
    "p_hbp", "p_bk", "p_wp", "p_bfp",
    "p_earned_run_avg_plus", "p_fip", "p_whip",
    "p_hits_per_nine", "p_hr_per_nine", "p_bb_per_nine",
    "p_so_per_nine", "p_strikeouts_per_base_on_balls",
]


def discover_pitcher_urls(session: RateLimitedSession, team: str, year: int) -> list[dict]:
    """Find pitcher URLs from a team's roster page."""
    url = f"{BASE_URL}/teams/{team}/{year}.shtml"
    try:
        resp = session.get(url)
    except Exception as e:
        logger.warning(f"Failed to fetch roster {team} {year}: {e}")
        return []

    # Parse the pitching table to find player links
    df = parse_br_table(resp.text, "team_pitching")
    if df.empty:
        df = parse_br_table(resp.text, "players_standard_pitching")

    pitchers = []
    if not df.empty:
        for _, row in df.iterrows():
            name = row.get("player", row.get("name_display", ""))
            player_url = row.get("player_url", row.get("name_display_url", ""))
            if player_url and "/players/" in str(player_url):
                # Extract player ID from URL like /players/o/ohtansh01.shtml
                match = re.search(r"/players/\w/(\w+)\.shtml", str(player_url))
                player_id = match.group(1) if match else ""
                pitchers.append({
                    "name": str(name).strip(),
                    "player_id": player_id,
                    "url": player_url,
                    "team": team,
                    "year": year,
                })

    return pitchers


def scrape_pitcher_stats(session: RateLimitedSession, player_url: str, player_id: str) -> list[dict]:
    """Scrape a pitcher's season-by-season stats from their player page."""
    full_url = f"{BASE_URL}{player_url}" if player_url.startswith("/") else player_url

    try:
        resp = session.get(full_url)
    except Exception as e:
        logger.warning(f"Failed to fetch pitcher {player_id}: {e}")
        return []

    # Parse standard pitching table
    df = parse_br_table(resp.text, "players_standard_pitching")
    if df.empty:
        return []

    rows = []
    for _, row in df.iterrows():
        year = row.get("year_id", "")
        if not year or not str(year).strip().isdigit():
            continue

        year_int = int(year)
        if year_int < START_YEAR or year_int > END_YEAR:
            continue

        record = {"player_id": player_id, "player_url": player_url}
        for col in PITCHING_COLS:
            val = row.get(col, None)
            if col in ("year_id", "team_name_abbr", "comp_name_abbr"):
                record[col] = str(val).strip() if val else ""
            elif col == "p_ip":
                record[col] = safe_float(val, 0)
            else:
                record[col] = safe_float(val)

        if record.get("team_name_abbr"):
            record["team_name_abbr"] = normalize_team_name(record["team_name_abbr"])

        rows.append(record)

    return rows


def main():
    session = RateLimitedSession(delay=4.0)
    progress_path = DATA_RAW / "pitchers_progress.ndjson"
    progress = load_progress(progress_path)
    scraped_ids = {r["player_id"] for r in progress if "player_id" in r}
    scraped_rosters = {f"{r['team']}_{r['year']}" for r in progress if "team" in r and "year" in r}

    # Phase 1: Discover all pitcher URLs from team rosters
    all_pitcher_info = []
    for year in range(START_YEAR, END_YEAR + 1):
        for team in TEAMS:
            key = f"{team}_{year}"
            if key in scraped_rosters:
                continue

            pitchers = discover_pitcher_urls(session, team, year)
            all_pitcher_info.extend(pitchers)
            progress.append({"team": team, "year": year, "type": "roster", "count": len(pitchers)})

            if len(progress) % 20 == 0:
                save_progress(progress_path, progress)

        logger.info(f"Discovered pitchers for {year}")

    # Deduplicate pitcher URLs
    unique_pitchers = {}
    for p in all_pitcher_info:
        pid = p["player_id"]
        if pid and pid not in unique_pitchers:
            unique_pitchers[pid] = p

    logger.info(f"Found {len(unique_pitchers)} unique pitchers to scrape")

    # Phase 2: Scrape each pitcher's stats
    all_stats = []
    for i, (pid, info) in enumerate(unique_pitchers.items()):
        if pid in scraped_ids:
            continue

        stats = scrape_pitcher_stats(session, info["url"], pid)
        all_stats.extend(stats)

        progress.append({"player_id": pid, "type": "stats", "seasons": len(stats)})
        if (i + 1) % 50 == 0:
            save_progress(progress_path, progress)
            logger.info(f"Scraped {i + 1}/{len(unique_pitchers)} pitchers ({len(all_stats)} season rows)")

    save_progress(progress_path, progress)

    # Save
    pitchers_path = DATA_RAW / "pitchers.csv"
    stats_df = pd.DataFrame(all_stats)

    if pitchers_path.exists():
        existing = pd.read_csv(pitchers_path)
        stats_df = pd.concat([existing, stats_df], ignore_index=True)
        stats_df = stats_df.drop_duplicates(subset=["player_id", "year_id", "team_name_abbr"], keep="last")

    stats_df.to_csv(pitchers_path, index=False)
    logger.info(f"Saved {len(stats_df)} pitcher-season rows to {pitchers_path}")


if __name__ == "__main__":
    main()
