"""Fetch live MLB odds from TheOddsAPI.

Output: data/odds/upcoming_cache.json
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from utils.helpers import get_logger, DATA_ODDS, save_json

load_dotenv()

logger = get_logger("scrape_odds")

ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "baseball_mlb"
REGIONS = "us"
MARKETS = "h2h,totals"
ODDS_FORMAT = "american"


def fetch_live_odds(api_key: str = None) -> dict:
    """Fetch current MLB game odds from TheOddsAPI."""
    import requests

    api_key = api_key or os.getenv("ODDS_API_KEY")
    if not api_key:
        logger.error("No ODDS_API_KEY found. Set it in .env or environment.")
        return {}

    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": REGIONS,
        "markets": MARKETS,
        "oddsFormat": ODDS_FORMAT,
        "dateFormat": "iso",
    }

    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()

    data = resp.json()
    remaining = resp.headers.get("x-requests-remaining", "?")
    logger.info(f"Fetched {len(data)} games. API requests remaining: {remaining}")

    # Parse into a simpler structure
    games = []
    for event in data:
        game = {
            "id": event.get("id"),
            "sport": event.get("sport_key"),
            "commence_time": event.get("commence_time"),
            "home_team": event.get("home_team"),
            "away_team": event.get("away_team"),
            "bookmakers": [],
        }

        for book in event.get("bookmakers", []):
            book_data = {
                "key": book.get("key"),
                "title": book.get("title"),
                "markets": {},
            }
            for market in book.get("markets", []):
                mkt_key = market.get("key")
                outcomes = {}
                for outcome in market.get("outcomes", []):
                    name = outcome.get("name")
                    price = outcome.get("price")
                    point = outcome.get("point")
                    outcomes[name] = {"price": price}
                    if point is not None:
                        outcomes[name]["point"] = point
                book_data["markets"][mkt_key] = outcomes
            game["bookmakers"].append(book_data)

        games.append(game)

    result = {
        "fetched_at": datetime.utcnow().isoformat(),
        "games": games,
    }

    # Cache to disk
    cache_path = DATA_ODDS / "upcoming_cache.json"
    save_json(cache_path, result)
    logger.info(f"Cached {len(games)} games to {cache_path}")

    return result


def get_consensus_odds(cache: dict) -> list[dict]:
    """Extract consensus (average) moneyline odds across bookmakers."""
    from utils.odds_math import american_to_implied_prob, remove_vig

    games = []
    for event in cache.get("games", []):
        home = event["home_team"]
        away = event["away_team"]

        home_prices = []
        away_prices = []

        for book in event.get("bookmakers", []):
            h2h = book.get("markets", {}).get("h2h", {})
            if home in h2h:
                home_prices.append(h2h[home]["price"])
            if away in h2h:
                away_prices.append(h2h[away]["price"])

        if home_prices and away_prices:
            avg_home = sum(home_prices) / len(home_prices)
            avg_away = sum(away_prices) / len(away_prices)
            home_impl = american_to_implied_prob(avg_home)
            away_impl = american_to_implied_prob(avg_away)
            home_fair, away_fair = remove_vig(home_impl, away_impl)

            games.append({
                "commence_time": event.get("commence_time"),
                "home_team": home,
                "away_team": away,
                "home_odds": round(avg_home),
                "away_odds": round(avg_away),
                "home_implied_prob": round(home_fair, 4),
                "away_implied_prob": round(away_fair, 4),
                "num_books": len(home_prices),
            })

    return games


if __name__ == "__main__":
    fetch_live_odds()
