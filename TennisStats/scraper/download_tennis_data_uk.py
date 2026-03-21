"""Download ATP match results + odds from tennis-data.co.uk (year by year, resumable)."""

import sys
import io
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from utils.helpers import get_logger, DATA_RAW, RateLimitedSession, load_progress, save_progress

logger = get_logger("download_tdu")

YEARS = list(range(2010, 2026))

# tennis-data.co.uk URL patterns (they vary by year)
def get_url(year: int) -> str:
    """Get the download URL for a given year."""
    return f"http://www.tennis-data.co.uk/{year}/{year}.xlsx"


def get_alt_urls(year: int) -> list[str]:
    """Alternative URL patterns tennis-data.co.uk has used."""
    return [
        f"http://www.tennis-data.co.uk/{year}/{year}.xlsx",
        f"http://www.tennis-data.co.uk/{year}w/{year}.xlsx",
        f"http://www.tennis-data.co.uk/{year}/{year}.xls",
    ]


def download_year(year: int, session: RateLimitedSession) -> pd.DataFrame:
    """Download one year of ATP data. Returns DataFrame or None on failure."""
    urls = get_alt_urls(year)

    for url in urls:
        try:
            logger.info(f"Trying {url}")
            resp = session.get(url)
            if resp.status_code == 200:
                df = pd.read_excel(io.BytesIO(resp.content))
                if len(df) > 0:
                    logger.info(f"  -> {len(df)} matches")
                    return df
        except Exception as e:
            logger.warning(f"  -> Failed: {e}")
            continue

    logger.warning(f"Could not download {year} from any URL")
    return None


def main():
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    session = RateLimitedSession(delay=3.0, max_retries=3)

    progress = load_progress(DATA_RAW / "tennis_data_uk_progress.ndjson")
    completed_years = {p["year"] for p in progress}

    all_dfs = []
    for year in YEARS:
        csv_path = DATA_RAW / f"matches_tdu_{year}.csv"

        if year in completed_years and csv_path.exists():
            logger.info(f"Skipping {year} (already downloaded)")
            all_dfs.append(pd.read_csv(csv_path))
            continue

        df = download_year(year, session)
        if df is not None and len(df) > 0:
            df.to_csv(csv_path, index=False)
            all_dfs.append(df)

            progress.append({"year": year, "rows": len(df), "status": "ok"})
            save_progress(DATA_RAW / "tennis_data_uk_progress.ndjson", progress)
            logger.info(f"Saved {year}: {len(df)} matches -> {csv_path.name}")
        else:
            progress.append({"year": year, "rows": 0, "status": "failed"})
            save_progress(DATA_RAW / "tennis_data_uk_progress.ndjson", progress)

    # Combine all years
    if all_dfs:
        combined = pd.concat(all_dfs, ignore_index=True)
        combined.to_csv(DATA_RAW / "matches_odds.csv", index=False)
        logger.info(f"Combined: {len(combined)} total matches -> matches_odds.csv")
    else:
        logger.error("No data downloaded!")


if __name__ == "__main__":
    main()
