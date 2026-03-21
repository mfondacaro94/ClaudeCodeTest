"""Scrape team-level batting and pitching stats from baseball-reference.com.

Crawls season pages for each year and extracts team aggregate stats.
Output: data/raw/teams_batting.csv, data/raw/teams_pitching.csv
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import parse_br_table, normalize_team_name, safe_float

logger = get_logger("scrape_teams")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025

BATTING_STATS = [
    "team_name", "batting_avg", "onbase_perc", "slugging_perc",
    "onbase_plus_slugging", "R", "H", "2B", "3B", "HR", "SB", "CS",
    "BB", "SO", "PA", "AB", "HBP", "SF", "SH", "GIDP", "TB",
]

PITCHING_STATS = [
    "team_name", "earned_run_avg", "R", "ER", "H", "HR", "BB", "SO",
    "IP", "whip", "hits_per_nine", "hr_per_nine", "bb_per_nine",
    "so_per_nine", "strikeouts_per_base_on_balls", "W", "L",
]


def scrape_season_batting(session: RateLimitedSession, year: int) -> list[dict]:
    """Scrape team batting stats for a given season."""
    url = f"{BASE_URL}/leagues/majors/{year}-standard-batting.shtml"
    logger.info(f"Scraping team batting: {year}")
    resp = session.get(url)
    df = parse_br_table(resp.text, "teams_standard_batting")

    if df.empty:
        # Fallback: try the main season page
        url = f"{BASE_URL}/leagues/majors/{year}.shtml"
        resp = session.get(url)
        df = parse_br_table(resp.text, "teams_standard_batting")

    if df.empty:
        logger.warning(f"No batting data found for {year}")
        return []

    rows = []
    for _, row in df.iterrows():
        team = row.get("team_name_abbr", row.get("team_name", ""))
        if not team or team in ("Lg Avg", "LgAvg", "") or "ZZZZ" in str(team).upper():
            continue
        record = {"year": year, "team": normalize_team_name(str(team))}
        for stat in BATTING_STATS:
            if stat == "team_name":
                continue
            record[f"bat_{stat}"] = safe_float(row.get(stat))
        rows.append(record)
    return rows


def scrape_season_pitching(session: RateLimitedSession, year: int) -> list[dict]:
    """Scrape team pitching stats for a given season."""
    url = f"{BASE_URL}/leagues/majors/{year}-standard-pitching.shtml"
    logger.info(f"Scraping team pitching: {year}")
    resp = session.get(url)
    df = parse_br_table(resp.text, "teams_standard_pitching")

    if df.empty:
        url = f"{BASE_URL}/leagues/majors/{year}.shtml"
        resp = session.get(url)
        df = parse_br_table(resp.text, "teams_standard_pitching")

    if df.empty:
        logger.warning(f"No pitching data found for {year}")
        return []

    rows = []
    for _, row in df.iterrows():
        team = row.get("team_name_abbr", row.get("team_name", ""))
        if not team or team in ("Lg Avg", "LgAvg", "") or "ZZZZ" in str(team).upper():
            continue
        record = {"year": year, "team": normalize_team_name(str(team))}
        for stat in PITCHING_STATS:
            if stat == "team_name":
                continue
            record[f"pitch_{stat}"] = safe_float(row.get(stat))
        rows.append(record)
    return rows


def main():
    session = RateLimitedSession(delay=6.0)
    progress_path = DATA_RAW / "teams_progress.ndjson"
    progress = load_progress(progress_path)
    scraped_years = {r["year"] for r in progress}

    all_batting = []
    all_pitching = []

    for year in range(START_YEAR, END_YEAR + 1):
        if year in scraped_years:
            logger.info(f"Skipping {year} (already scraped)")
            continue

        batting = scrape_season_batting(session, year)
        pitching = scrape_season_pitching(session, year)

        all_batting.extend(batting)
        all_pitching.extend(pitching)

        progress.append({"year": year, "batting_count": len(batting), "pitching_count": len(pitching)})
        save_progress(progress_path, progress)
        logger.info(f"  {year}: {len(batting)} batting, {len(pitching)} pitching rows")

    # Merge with any previously scraped data
    bat_path = DATA_RAW / "teams_batting.csv"
    pitch_path = DATA_RAW / "teams_pitching.csv"

    if bat_path.exists():
        existing = pd.read_csv(bat_path)
        all_batting = pd.concat([existing, pd.DataFrame(all_batting)], ignore_index=True)
        all_batting = all_batting.drop_duplicates(subset=["year", "team"], keep="last")
    else:
        all_batting = pd.DataFrame(all_batting)

    if pitch_path.exists():
        existing = pd.read_csv(pitch_path)
        all_pitching = pd.concat([existing, pd.DataFrame(all_pitching)], ignore_index=True)
        all_pitching = all_pitching.drop_duplicates(subset=["year", "team"], keep="last")
    else:
        all_pitching = pd.DataFrame(all_pitching)

    all_batting.to_csv(bat_path, index=False)
    all_pitching.to_csv(pitch_path, index=False)
    logger.info(f"Saved {len(all_batting)} batting rows, {len(all_pitching)} pitching rows")


if __name__ == "__main__":
    main()
