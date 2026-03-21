"""Scrape individual batter (position player) stats from baseball-reference.com.

Discovery: finds batters via team roster pages, then scrapes each
player's career/season batting stats from their player page.
Output: data/raw/batters.csv
"""

import sys
import re
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import parse_br_table, safe_float, normalize_team_name

logger = get_logger("scrape_batters")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025

TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]

BATTING_COLS = [
    "year_id", "age", "team_name_abbr", "comp_name_abbr",
    "b_war", "b_games", "b_pa", "b_ab", "b_r", "b_h",
    "b_doubles", "b_triples", "b_hr", "b_rbi", "b_sb", "b_cs",
    "b_bb", "b_so", "b_batting_avg", "b_onbase_perc",
    "b_slugging_perc", "b_onbase_plus_slugging",
    "b_onbase_plus_slugging_plus", "b_roba", "b_rbat_plus",
    "b_tb", "b_gidp", "b_hbp", "b_sf", "b_ibb",
    "pos",
]


def discover_batter_urls(session: RateLimitedSession, team: str, year: int) -> list[dict]:
    """Find batter URLs from a team's roster page."""
    url = f"{BASE_URL}/teams/{team}/{year}.shtml"
    try:
        resp = session.get(url)
    except Exception as e:
        logger.warning(f"Failed to fetch roster {team} {year}: {e}")
        return []

    df = parse_br_table(resp.text, "team_batting")
    if df.empty:
        df = parse_br_table(resp.text, "players_standard_batting")

    batters = []
    if not df.empty:
        for _, row in df.iterrows():
            name = row.get("player", row.get("name_display", ""))
            player_url = row.get("player_url", row.get("name_display_url", ""))
            if player_url and "/players/" in str(player_url):
                match = re.search(r"/players/\w/(\w+)\.shtml", str(player_url))
                player_id = match.group(1) if match else ""
                batters.append({
                    "name": str(name).strip(),
                    "player_id": player_id,
                    "url": player_url,
                    "team": team,
                    "year": year,
                })

    return batters


def scrape_batter_stats(session: RateLimitedSession, player_url: str, player_id: str) -> list[dict]:
    """Scrape a batter's season-by-season stats from their player page."""
    full_url = f"{BASE_URL}{player_url}" if player_url.startswith("/") else player_url

    try:
        resp = session.get(full_url)
    except Exception as e:
        logger.warning(f"Failed to fetch batter {player_id}: {e}")
        return []

    df = parse_br_table(resp.text, "players_standard_batting")
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
        for col in BATTING_COLS:
            val = row.get(col, None)
            if col in ("year_id", "team_name_abbr", "comp_name_abbr", "pos"):
                record[col] = str(val).strip() if val else ""
            else:
                record[col] = safe_float(val)

        if record.get("team_name_abbr"):
            record["team_name_abbr"] = normalize_team_name(record["team_name_abbr"])

        rows.append(record)

    return rows


def main():
    session = RateLimitedSession(delay=6.0)
    progress_path = DATA_RAW / "batters_progress.ndjson"
    progress = load_progress(progress_path)
    scraped_ids = {r["player_id"] for r in progress if r.get("type") == "stats"}
    scraped_rosters = {f"{r['team']}_{r['year']}" for r in progress if r.get("type") == "roster"}

    # Phase 1: Discover batter URLs from team rosters
    all_batter_info = []
    for year in range(START_YEAR, END_YEAR + 1):
        for team in TEAMS:
            key = f"{team}_{year}"
            if key in scraped_rosters:
                continue

            batters = discover_batter_urls(session, team, year)
            all_batter_info.extend(batters)
            progress.append({"team": team, "year": year, "type": "roster", "count": len(batters)})

            if len(progress) % 20 == 0:
                save_progress(progress_path, progress)

        logger.info(f"Discovered batters for {year}")

    # If all rosters were already in progress, re-discover batter URLs
    # so we can find any players whose stats haven't been scraped yet
    if not all_batter_info:
        logger.info("Re-discovering batter URLs from previously scraped rosters...")
        for year in range(START_YEAR, END_YEAR + 1):
            for team in TEAMS:
                batters = discover_batter_urls(session, team, year)
                all_batter_info.extend(batters)
            logger.info(f"Re-discovered batters for {year}")

    # Deduplicate
    unique_batters = {}
    for b in all_batter_info:
        bid = b["player_id"]
        if bid and bid not in unique_batters:
            unique_batters[bid] = b

    logger.info(f"Found {len(unique_batters)} unique batters to scrape")

    # Phase 2: Scrape each batter's stats (with incremental CSV saves)
    batters_path = DATA_RAW / "batters.csv"
    batch_stats = []
    scraped_count = 0

    for i, (bid, info) in enumerate(unique_batters.items()):
        if bid in scraped_ids:
            continue

        stats = scrape_batter_stats(session, info["url"], bid)
        batch_stats.extend(stats)
        scraped_count += 1

        progress.append({"player_id": bid, "type": "stats", "seasons": len(stats)})

        # Save to CSV every 50 batters so we never lose more than 50
        if scraped_count % 50 == 0:
            save_progress(progress_path, progress)
            batch_df = pd.DataFrame(batch_stats)
            if not batch_df.empty:
                if batters_path.exists() and batters_path.stat().st_size > 0:
                    existing = pd.read_csv(batters_path)
                    batch_df = pd.concat([existing, batch_df], ignore_index=True)
                    batch_df = batch_df.drop_duplicates(subset=["player_id", "year_id", "team_name_abbr"], keep="last")
                batch_df.to_csv(batters_path, index=False)
                logger.info(f"Scraped {scraped_count}/{len(unique_batters)} batters — saved {len(batch_df)} rows to CSV")
                batch_stats = []

    # Final save for any remaining batch
    save_progress(progress_path, progress)
    if batch_stats:
        batch_df = pd.DataFrame(batch_stats)
        if batters_path.exists() and batters_path.stat().st_size > 0:
            existing = pd.read_csv(batters_path)
            batch_df = pd.concat([existing, batch_df], ignore_index=True)
            batch_df = batch_df.drop_duplicates(subset=["player_id", "year_id", "team_name_abbr"], keep="last")
        batch_df.to_csv(batters_path, index=False)

    total = len(pd.read_csv(batters_path)) if batters_path.exists() and batters_path.stat().st_size > 0 else 0
    logger.info(f"Saved {total} batter-season rows to {batters_path}")


if __name__ == "__main__":
    main()
