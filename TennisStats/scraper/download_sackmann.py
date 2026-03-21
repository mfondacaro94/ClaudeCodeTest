"""Download ATP match stats, player bios, and rankings from Jeff Sackmann's GitHub (resumable)."""

import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from utils.helpers import get_logger, DATA_RAW, RateLimitedSession, load_progress, save_progress

logger = get_logger("download_sackmann")

YEARS = list(range(2010, 2026))
BASE_URL = "https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master"


def download_matches(session: RateLimitedSession):
    """Download match files year by year with progress saves."""
    progress = load_progress(DATA_RAW / "sackmann_matches_progress.ndjson")
    completed_years = {p["year"] for p in progress}

    all_dfs = []
    for year in YEARS:
        csv_path = DATA_RAW / f"matches_sackmann_{year}.csv"

        if year in completed_years and csv_path.exists():
            logger.info(f"Skipping matches {year} (already downloaded)")
            all_dfs.append(pd.read_csv(csv_path))
            continue

        url = f"{BASE_URL}/atp_matches_{year}.csv"
        try:
            logger.info(f"Downloading matches {year}...")
            resp = session.get(url)
            df = pd.read_csv(io.StringIO(resp.text))
            df.to_csv(csv_path, index=False)
            all_dfs.append(df)

            progress.append({"year": year, "rows": len(df), "status": "ok"})
            save_progress(DATA_RAW / "sackmann_matches_progress.ndjson", progress)
            logger.info(f"  -> {len(df)} matches saved")
        except Exception as e:
            logger.warning(f"  -> Failed {year}: {e}")
            progress.append({"year": year, "rows": 0, "status": "failed"})
            save_progress(DATA_RAW / "sackmann_matches_progress.ndjson", progress)

    # Combine all years
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(DATA_RAW / "matches_stats.csv", index=False)
        logger.info(f"Combined: {len(combined)} total matches -> matches_stats.csv")

    return all_dfs


def download_players(session: RateLimitedSession):
    """Download player biographical data."""
    out_path = DATA_RAW / "players.csv"
    if out_path.exists():
        logger.info("Players file already exists, skipping")
        return

    url = f"{BASE_URL}/atp_players.csv"
    logger.info("Downloading player data...")
    resp = session.get(url)
    df = pd.read_csv(io.StringIO(resp.text))
    df.to_csv(out_path, index=False)
    logger.info(f"Saved {len(df)} players -> players.csv")


def download_rankings(session: RateLimitedSession):
    """Download rankings history files."""
    out_path = DATA_RAW / "rankings_history.csv"
    if out_path.exists():
        logger.info("Rankings file already exists, skipping")
        return

    decades = ["00s", "10s", "20s", "current"]
    all_dfs = []

    for decade in decades:
        url = f"{BASE_URL}/atp_rankings_{decade}.csv"
        try:
            logger.info(f"Downloading rankings {decade}...")
            resp = session.get(url)
            df = pd.read_csv(io.StringIO(resp.text))
            all_dfs.append(df)
            logger.info(f"  -> {len(df)} ranking entries")
        except Exception as e:
            logger.warning(f"  -> Failed {decade}: {e}")

    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(out_path, index=False)
        logger.info(f"Combined: {len(combined)} ranking entries -> rankings_history.csv")


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    session = RateLimitedSession(delay=1.0, max_retries=3)

    download_matches(session)
    download_players(session)
    download_rankings(session)
    logger.info("All Sackmann data downloaded.")


if __name__ == "__main__":
    main()
