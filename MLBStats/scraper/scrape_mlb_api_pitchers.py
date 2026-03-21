"""Scrape pitcher game-level data from the official MLB Stats API.

Uses statsapi.mlb.com (free, no API key, no aggressive rate limits).
Pulls schedule + boxscore data to get:
- Starting pitcher per game
- All reliever appearances per game (IP, K, ER, etc.)
- Pitcher handedness (L/R)

One API call per game day = ~2,200 requests total for 2015-2025.
Saves every 50 days to CSV for crash safety.

Output: data/raw/pitcher_gamelogs.csv (overwrites BR version with better data)
"""

import sys
import time
import json
import requests
import pandas as pd
from pathlib import Path
from datetime import date, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import normalize_team_name

logger = get_logger("scrape_mlb_api")

# MLB team ID -> abbreviation mapping
TEAM_ID_MAP = {}  # Built dynamically from API


def build_team_map(session: requests.Session) -> dict:
    """Fetch MLB team ID to abbreviation mapping."""
    resp = session.get("https://statsapi.mlb.com/api/v1/teams?sportId=1")
    teams = resp.json().get("teams", [])
    mapping = {}
    for t in teams:
        abbr = normalize_team_name(t.get("abbreviation", ""))
        mapping[t["id"]] = abbr
    return mapping


def get_pitcher_hand(session: requests.Session, player_id: int, cache: dict) -> str:
    """Get pitcher handedness (L/R) with caching."""
    if player_id in cache:
        return cache[player_id]
    try:
        resp = session.get(f"https://statsapi.mlb.com/api/v1/people/{player_id}")
        people = resp.json().get("people", [])
        if people:
            hand = people[0].get("pitchHand", {}).get("code", "R")
            cache[player_id] = hand
            return hand
    except Exception:
        pass
    cache[player_id] = "R"
    return "R"


def scrape_day(session: requests.Session, game_date: date, team_map: dict,
               hand_cache: dict) -> list[dict]:
    """Scrape all games for a single day."""
    date_str = game_date.strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?date={date_str}&sportId=1&hydrate=probablePitcher"

    try:
        resp = session.get(url, timeout=15)
        data = resp.json()
    except Exception as e:
        logger.warning(f"Failed to fetch schedule for {date_str}: {e}")
        return []

    dates = data.get("dates", [])
    if not dates:
        return []

    games = dates[0].get("games", [])
    rows = []

    for game in games:
        status = game.get("status", {}).get("abstractGameState", "")
        if status != "Final":
            continue

        game_pk = game["gamePk"]
        home_team_id = game["teams"]["home"]["team"]["id"]
        away_team_id = game["teams"]["away"]["team"]["id"]
        home_team = team_map.get(home_team_id, "UNK")
        away_team = team_map.get(away_team_id, "UNK")

        # Get probable pitchers from schedule (backup)
        home_sp_name = game["teams"]["home"].get("probablePitcher", {}).get("fullName", "")
        away_sp_name = game["teams"]["away"].get("probablePitcher", {}).get("fullName", "")
        home_sp_id = game["teams"]["home"].get("probablePitcher", {}).get("id", 0)
        away_sp_id = game["teams"]["away"].get("probablePitcher", {}).get("id", 0)

        # Get boxscore for detailed pitcher stats
        try:
            box_resp = session.get(
                f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore", timeout=15
            )
            box = box_resp.json()
        except Exception as e:
            logger.warning(f"Failed to fetch boxscore {game_pk}: {e}")
            continue

        # Brief delay between boxscore calls
        time.sleep(0.3)

        for side, team_abbr in [("home", home_team), ("away", away_team)]:
            team_data = box.get("teams", {}).get(side, {})
            pitcher_ids = team_data.get("pitchers", [])
            players = team_data.get("players", {})

            for idx, pid in enumerate(pitcher_ids):
                player_data = players.get(f"ID{pid}", {})
                person = player_data.get("person", {})
                stats = player_data.get("stats", {}).get("pitching", {})

                if not stats:
                    continue

                is_starter = idx == 0  # First pitcher listed is the starter
                name = person.get("fullName", "")
                hand = get_pitcher_hand(session, pid, hand_cache)

                ip_str = stats.get("inningsPitched", "0")
                try:
                    ip = float(ip_str)
                except (ValueError, TypeError):
                    ip = 0

                rows.append({
                    "date": date_str,
                    "year": game_date.year,
                    "game_pk": game_pk,
                    "player_id": pid,
                    "player_name": name,
                    "throws": hand,
                    "team": team_abbr,
                    "opp": away_team if side == "home" else home_team,
                    "is_home": side == "home",
                    "home_team": home_team,
                    "away_team": away_team,
                    "is_start": is_starter,
                    "ip": ip,
                    "h": stats.get("hits", 0),
                    "r": stats.get("runs", 0),
                    "er": stats.get("earnedRuns", 0),
                    "bb": stats.get("baseOnBalls", 0),
                    "so": stats.get("strikeOuts", 0),
                    "hr": stats.get("homeRuns", 0),
                    "pitches": stats.get("numberOfPitches", 0),
                    "strikes": stats.get("strikes", 0),
                    "decision": stats.get("note", ""),
                })

    return rows


def flush_batch(batch, output_path, progress, progress_path, days_done, total_days):
    """Save batch to CSV and progress."""
    save_progress(progress_path, progress)
    if not batch:
        return
    batch_df = pd.DataFrame(batch)
    if output_path.exists() and output_path.stat().st_size > 0:
        existing = pd.read_csv(output_path)
        batch_df = pd.concat([existing, batch_df], ignore_index=True)
        batch_df = batch_df.drop_duplicates(
            subset=["date", "player_id", "game_pk"], keep="last"
        )
    batch_df.to_csv(output_path, index=False)
    logger.info(f"Day {days_done}/{total_days} — saved {len(batch_df)} total appearances")


def main():
    session = requests.Session()
    session.headers.update({"User-Agent": "MLBStatsResearch/1.0"})

    # Build team mapping
    team_map = build_team_map(session)
    logger.info(f"Loaded {len(team_map)} MLB teams")

    # Pitcher handedness cache
    hand_cache = {}

    # Progress tracking
    progress_path = DATA_RAW / "mlb_api_progress.ndjson"
    progress = load_progress(progress_path)
    scraped_dates = {r["date"] for r in progress}

    # Generate all game dates (March 20 - Oct 5 each year)
    all_dates = []
    for year in range(2015, 2026):
        d = date(year, 3, 20)
        end = date(year, 10, 5)
        while d <= end:
            if d.strftime("%Y-%m-%d") not in scraped_dates:
                all_dates.append(d)
            d += timedelta(days=1)

    logger.info(f"Found {len(all_dates)} days to scrape ({len(scraped_dates)} already done)")

    output_path = DATA_RAW / "pitcher_gamelogs.csv"
    batch = []
    days_done = len(scraped_dates)
    total_days = len(all_dates) + len(scraped_dates)

    for game_date in all_dates:
        rows = scrape_day(session, game_date, team_map, hand_cache)
        batch.extend(rows)
        days_done += 1

        date_str = game_date.strftime("%Y-%m-%d")
        progress.append({"date": date_str, "games": len(rows)})

        # Save every 50 days
        if days_done % 50 == 0:
            flush_batch(batch, output_path, progress, progress_path, days_done, total_days)
            batch = []

        # Gentle delay between days (0.5s + boxscore delays)
        time.sleep(0.5)

    # Final save
    flush_batch(batch, output_path, progress, progress_path, days_done, total_days)

    total = len(pd.read_csv(output_path)) if output_path.exists() else 0
    logger.info(f"Complete. {total} pitcher appearances saved. {len(hand_cache)} pitcher handedness cached.")


if __name__ == "__main__":
    main()
