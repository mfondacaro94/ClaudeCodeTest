"""Fetch historical and forecast weather data for MLB game locations.

Uses the Open-Meteo API (free, no key required) for:
- Historical weather at game time for past games
- Forecast weather for upcoming games

Weather factors that affect MLB games:
- Temperature: ball carries further in heat, offense up
- Wind speed/direction: impacts home runs and fly balls
- Humidity: marginal effect on ball flight
- Precipitation: game delays, slippery conditions
- Barometric pressure: lower pressure = more HRs

Output: data/raw/weather.csv
"""

import sys
import time
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, DATA_RAW, save_progress, load_progress
from utils.stadium_data import STADIUMS

logger = get_logger("scrape_weather")

OPEN_METEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

HOURLY_VARS = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_direction_10m",
    "surface_pressure",
]


def fetch_weather_for_game(lat: float, lon: float, date: str,
                           game_hour: int = 19, is_forecast: bool = False) -> dict:
    """Fetch weather data for a specific location and date.

    Args:
        lat: Stadium latitude
        lon: Stadium longitude
        date: Game date (YYYY-MM-DD)
        game_hour: Approximate game start hour (24h, local time)
        is_forecast: Use forecast API instead of archive

    Returns:
        Dict with weather variables at game time.
    """
    api_url = OPEN_METEO_FORECAST if is_forecast else OPEN_METEO_ARCHIVE

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date,
        "end_date": date,
        "hourly": ",".join(HOURLY_VARS),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "America/New_York",
    }

    try:
        resp = requests.get(api_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning(f"Weather fetch failed for {date} at ({lat},{lon}): {e}")
        return {}

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])

    # Find the index closest to game time
    target_idx = None
    for i, t in enumerate(times):
        hour = int(t.split("T")[1].split(":")[0])
        if hour == game_hour:
            target_idx = i
            break

    if target_idx is None and times:
        # Default to 7 PM slot or last available
        target_idx = min(19, len(times) - 1)

    if target_idx is None:
        return {}

    result = {"date": date, "lat": lat, "lon": lon, "game_hour": game_hour}
    for var in HOURLY_VARS:
        values = hourly.get(var, [])
        result[var] = values[target_idx] if target_idx < len(values) else None

    return result


def get_stadium_for_team(team: str) -> dict:
    """Look up stadium coordinates for a team."""
    return STADIUMS.get(team, {})


def fetch_weather_for_games(games_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch weather for all games in the DataFrame.

    Args:
        games_df: Must have columns: date, home_team

    Returns:
        DataFrame with weather columns added.
    """
    progress_path = DATA_RAW / "weather_progress.ndjson"
    progress = load_progress(progress_path)
    fetched_keys = {f"{r['date']}_{r['home_team']}" for r in progress}

    weather_rows = []
    total = len(games_df)

    for i, (_, game) in enumerate(games_df.iterrows()):
        date_str = str(game["date"])[:10]
        home = game["home_team"]
        key = f"{date_str}_{home}"

        if key in fetched_keys:
            continue

        stadium = get_stadium_for_team(home)
        if not stadium:
            logger.warning(f"No stadium data for {home}")
            continue

        # Determine if this is historical or forecast
        game_date = pd.to_datetime(date_str)
        is_forecast = game_date > datetime.now() - timedelta(days=5)

        # Estimate game hour (most games at 7 PM local, day games at 1 PM)
        game_hour = 19  # default evening

        weather = fetch_weather_for_game(
            stadium["lat"], stadium["lon"], date_str,
            game_hour=game_hour, is_forecast=is_forecast
        )

        if weather:
            weather["home_team"] = home
            weather_rows.append(weather)
            progress.append({"date": date_str, "home_team": home})

        # Rate limit: Open-Meteo allows ~600 req/min for free
        if (i + 1) % 500 == 0:
            save_progress(progress_path, progress)
            logger.info(f"Weather: {i + 1}/{total} games fetched")
            time.sleep(1)

    save_progress(progress_path, progress)

    weather_df = pd.DataFrame(weather_rows)

    # Save / append
    weather_path = DATA_RAW / "weather.csv"
    if weather_path.exists():
        existing = pd.read_csv(weather_path)
        weather_df = pd.concat([existing, weather_df], ignore_index=True)
        weather_df = weather_df.drop_duplicates(subset=["date", "home_team"], keep="last")

    weather_df.to_csv(weather_path, index=False)
    logger.info(f"Saved {len(weather_df)} weather records to {weather_path}")
    return weather_df


def main():
    """Fetch weather for all scraped games."""
    games_path = DATA_RAW / "games.csv"
    if not games_path.exists():
        logger.error("No games.csv found. Run scrape_games.py first.")
        return

    games = pd.read_csv(games_path)
    games["date"] = pd.to_datetime(games["date"])

    # Only outdoor stadiums matter most, but temperature affects all
    logger.info(f"Fetching weather for {len(games)} games...")
    fetch_weather_for_games(games)


if __name__ == "__main__":
    main()
