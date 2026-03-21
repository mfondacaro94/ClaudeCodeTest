"""Scrape park factors from baseball-reference.com.

Uses the teams_miscellaneous table from the league misc page.
Only 11 pages (one per year 2015-2025), very fast.
Output: data/raw/park_factors.csv
"""

import sys
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW
from utils.data_cleaning import parse_br_table, safe_float, normalize_team_name

logger = get_logger("scrape_park_factors")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025

# Map full team names to abbreviations
TEAM_NAME_MAP = {
    "Arizona Diamondbacks": "ARI", "Atlanta Braves": "ATL",
    "Baltimore Orioles": "BAL", "Boston Red Sox": "BOS",
    "Chicago Cubs": "CHC", "Chicago White Sox": "CHW",
    "Cincinnati Reds": "CIN", "Cleveland Guardians": "CLE",
    "Cleveland Indians": "CLE", "Colorado Rockies": "COL",
    "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KCR", "Los Angeles Angels": "LAA",
    "Los Angeles Angels of Anaheim": "LAA", "Los Angeles Dodgers": "LAD",
    "Miami Marlins": "MIA", "Milwaukee Brewers": "MIL",
    "Minnesota Twins": "MIN", "New York Mets": "NYM",
    "New York Yankees": "NYY", "Oakland Athletics": "OAK",
    "Athletics": "OAK", "Philadelphia Phillies": "PHI",
    "Pittsburgh Pirates": "PIT", "San Diego Padres": "SDP",
    "San Francisco Giants": "SFG", "Seattle Mariners": "SEA",
    "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TBR",
    "Texas Rangers": "TEX", "Toronto Blue Jays": "TOR",
    "Washington Nationals": "WSN",
}


def scrape_year(session: RateLimitedSession, year: int) -> list[dict]:
    url = f"{BASE_URL}/leagues/majors/{year}-misc.shtml"
    logger.info(f"Scraping park factors: {year}")

    try:
        resp = session.get(url)
    except Exception as e:
        logger.warning(f"Failed to fetch {year}: {e}")
        return []

    df = parse_br_table(resp.text, "teams_miscellaneous")
    if df.empty:
        return []

    rows = []
    for _, row in df.iterrows():
        team_name = str(row.get("team_name", "")).strip()
        team = TEAM_NAME_MAP.get(team_name, "")
        if not team:
            continue

        rows.append({
            "year": year,
            "team": team,
            "bpf": safe_float(row.get("bpf"), 100),
            "ppf": safe_float(row.get("ppf"), 100),
        })

    return rows


def main():
    session = RateLimitedSession(delay=6.0)
    all_rows = []

    for year in range(START_YEAR, END_YEAR + 1):
        rows = scrape_year(session, year)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    path = DATA_RAW / "park_factors.csv"
    df.to_csv(path, index=False)
    logger.info(f"Saved {len(df)} park factor rows to {path}")


if __name__ == "__main__":
    main()
