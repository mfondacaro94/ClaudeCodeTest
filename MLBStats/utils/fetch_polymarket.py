"""Fetch resolved MLB prediction market data from Polymarket."""

import sys
import requests
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, DATA_RAW

logger = get_logger("fetch_polymarket")

GAMMA_API = "https://gamma-api.polymarket.com"


def fetch_mlb_markets() -> pd.DataFrame:
    """Fetch resolved MLB game markets from Polymarket."""
    markets = []
    offset = 0
    limit = 100

    while True:
        params = {
            "closed": "true",
            "tag": "mlb",
            "limit": limit,
            "offset": offset,
            "order": "end_date_time",
            "ascending": "false",
        }

        try:
            resp = requests.get(f"{GAMMA_API}/markets", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning(f"Failed at offset {offset}: {e}")
            break

        if not data:
            break

        for market in data:
            question = market.get("question", "")
            if "MLB" not in question and "baseball" not in question.lower():
                continue

            markets.append({
                "id": market.get("id"),
                "question": question,
                "end_date": market.get("end_date_time"),
                "outcome": market.get("outcome"),
                "outcome_prices": market.get("outcomePrices"),
                "volume": market.get("volume"),
                "liquidity": market.get("liquidity"),
            })

        offset += limit
        if len(data) < limit:
            break

    df = pd.DataFrame(markets)
    if not df.empty:
        output = DATA_RAW / "polymarket_mlb.csv"
        df.to_csv(output, index=False)
        logger.info(f"Saved {len(df)} Polymarket MLB markets to {output}")

    return df


if __name__ == "__main__":
    fetch_mlb_markets()
