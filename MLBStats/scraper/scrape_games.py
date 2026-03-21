"""Scrape game-by-game results from baseball-reference.com.

Crawls team schedule pages to extract game results, starting pitchers,
scores, and other game-level data.
Output: data/raw/games.csv
"""

import sys
import re
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import parse_br_table, normalize_team_name, safe_float

logger = get_logger("scrape_games")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025

# All 30 MLB teams (current abbreviations)
TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]


def scrape_team_schedule(session: RateLimitedSession, team: str, year: int) -> list[dict]:
    """Scrape a team's full season schedule and results."""
    url = f"{BASE_URL}/teams/{team}/{year}-schedule-scores.shtml"
    logger.info(f"Scraping schedule: {team} {year}")

    try:
        resp = session.get(url)
    except Exception as e:
        logger.warning(f"Failed to fetch {team} {year}: {e}")
        return []

    df = parse_br_table(resp.text, "team_schedule")
    if df.empty:
        # Try alternate table id
        df = parse_br_table(resp.text, "results")
    if df.empty:
        logger.warning(f"No schedule data for {team} {year}")
        return []

    games = []
    for _, row in df.iterrows():
        # Skip header rows, postponed games
        gm_num = row.get("team_game", row.get("Gm#", ""))
        if not gm_num or str(gm_num).strip() in ("", "Gm#"):
            continue

        date_str = row.get("date_game", row.get("Date", ""))
        if not date_str:
            continue

        # Determine home/away
        homeaway = row.get("homeORvis", row.get("", ""))
        is_home = homeaway != "@"

        opp = row.get("opp_ID", row.get("Opp", ""))
        opp = normalize_team_name(str(opp).replace("@", "").strip())

        # Scores
        runs_scored = safe_float(row.get("R", row.get("runs_scored")), 0)
        runs_allowed = safe_float(row.get("RA", row.get("runs_allowed")), 0)

        # Result
        result = row.get("win_loss_result", row.get("W/L", ""))
        if not result or str(result).strip() == "":
            continue  # Game not yet played

        won = str(result).startswith("W")

        # Winning/losing/saving pitcher (BR schedule doesn't have starting pitcher)
        wp_name = row.get("winning_pitcher", "")
        wp_url = row.get("winning_pitcher_url", "")
        lp_name = row.get("losing_pitcher", "")
        lp_url = row.get("losing_pitcher_url", "")
        sv_name = row.get("saving_pitcher", "")
        sv_url = row.get("saving_pitcher_url", "")

        # Extra innings, day/night, attendance
        extra_innings = row.get("extra_innings", "")
        day_night = row.get("day_or_night", "")
        attendance = safe_float(str(row.get("attendance", "")).replace(",", ""))

        # Build game record from the home team's perspective
        if is_home:
            home_team = normalize_team_name(team)
            away_team = opp
            home_runs = runs_scored
            away_runs = runs_allowed
            home_win = 1 if won else 0
        else:
            home_team = opp
            away_team = normalize_team_name(team)
            home_runs = runs_allowed
            away_runs = runs_scored
            home_win = 0 if won else 1

        game = {
            "date": date_str,
            "year": year,
            "home_team": home_team,
            "away_team": away_team,
            "home_runs": home_runs,
            "away_runs": away_runs,
            "home_win": home_win,
            "scrape_team": normalize_team_name(team),
            "is_home": is_home,
            "wp_name": wp_name,
            "wp_url": wp_url,
            "lp_name": lp_name,
            "lp_url": lp_url,
            "sv_name": sv_name,
            "sv_url": sv_url,
            "extra_innings": extra_innings,
            "day_night": day_night,
            "attendance": attendance,
        }
        games.append(game)

    return games


def deduplicate_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Each game appears twice (once per team). Keep one canonical row.

    Strategy: Keep the row scraped from the home team's perspective
    (which has correct home_win, attendance, etc.). Drop the duplicate
    away-perspective rows.
    """
    if games_df.empty:
        return games_df

    # Prefer rows scraped from the home team's perspective
    home_rows = games_df[games_df["is_home"] == True].copy()

    # Drop helper columns
    home_rows = home_rows.drop(columns=["scrape_team", "is_home"], errors="ignore")

    # Deduplicate (some games may have multiple entries)
    home_rows = home_rows.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

    return home_rows


def main():
    session = RateLimitedSession(delay=6.0)
    progress_path = DATA_RAW / "games_progress.ndjson"
    progress = load_progress(progress_path)
    scraped_keys = {f"{r['team']}_{r['year']}" for r in progress}

    all_games = []

    for year in range(START_YEAR, END_YEAR + 1):
        for team in TEAMS:
            key = f"{team}_{year}"
            if key in scraped_keys:
                continue

            games = scrape_team_schedule(session, team, year)
            all_games.extend(games)

            progress.append({"team": team, "year": year, "games": len(games)})
            if len(progress) % 10 == 0:
                save_progress(progress_path, progress)

        save_progress(progress_path, progress)
        logger.info(f"Completed year {year}")

    # Convert to DataFrame and deduplicate new data
    games_path = DATA_RAW / "games.csv"
    new_df = pd.DataFrame(all_games)

    if not new_df.empty:
        new_df = deduplicate_games(new_df)

    # Merge with any previously scraped data (already deduped)
    if games_path.exists():
        existing = pd.read_csv(games_path)
        games_df = pd.concat([existing, new_df], ignore_index=True)
        # Drop duplicates in case new data overlaps with existing
        games_df = games_df.drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
    else:
        games_df = new_df

    if not games_df.empty:
        games_df = games_df.sort_values(["date", "home_team"]).reset_index(drop=True)
        games_df.to_csv(games_path, index=False)
        logger.info(f"Saved {len(games_df)} unique games to {games_path}")
    else:
        logger.warning("No games scraped")


if __name__ == "__main__":
    main()
