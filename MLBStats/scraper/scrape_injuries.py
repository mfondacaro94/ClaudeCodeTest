"""Scrape MLB injury reports and IL (Injured List) transactions.

Sources:
- baseball-reference.com transaction logs per team
- Roster status pages for current IL placements

Output: data/raw/injuries.csv
"""

import sys
import re
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, RateLimitedSession, DATA_RAW, save_progress, load_progress
from utils.data_cleaning import normalize_team_name

logger = get_logger("scrape_injuries")

BASE_URL = "https://www.baseball-reference.com"
START_YEAR = 2015
END_YEAR = 2025

TEAMS = [
    "ARI", "ATL", "BAL", "BOS", "CHC", "CHW", "CIN", "CLE", "COL", "DET",
    "HOU", "KCR", "LAA", "LAD", "MIA", "MIL", "MIN", "NYM", "NYY", "OAK",
    "PHI", "PIT", "SDP", "SEA", "SFG", "STL", "TBR", "TEX", "TOR", "WSN",
]


def scrape_team_transactions(session: RateLimitedSession, team: str, year: int) -> list[dict]:
    """Scrape IL placements and activations from team transaction pages."""
    url = f"{BASE_URL}/teams/{team}/{year}-transactions.shtml"

    try:
        resp = session.get(url)
    except Exception as e:
        logger.warning(f"Failed to fetch transactions {team} {year}: {e}")
        return []

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(resp.text, "lxml")

    transactions = []
    # Transaction entries are typically in a list or table
    content = soup.find("div", id="content")
    if not content:
        return []

    for p in content.find_all(["p", "li"]):
        text = p.get_text(strip=True)
        if not text:
            continue

        # Look for IL-related transactions
        is_il = any(kw in text.lower() for kw in [
            "injured list", "disabled list", "placed on",
            "activated from", "transferred to", "60-day il",
            "10-day il", "15-day il", "7-day il",
        ])

        is_trade = any(kw in text.lower() for kw in [
            "traded", "acquired", "in exchange for",
            "cash considerations", "player to be named",
        ])

        is_roster = any(kw in text.lower() for kw in [
            "called up", "optioned", "designated for assignment",
            "outrighted", "recalled", "selected the contract",
            "claimed off waivers", "released",
        ])

        if not (is_il or is_trade or is_roster):
            continue

        # Try to extract date from the text or parent
        date_str = ""
        date_match = re.search(r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{4}", text)
        if date_match:
            date_str = date_match.group(0)

        # Extract player names (often linked)
        player_links = p.find_all("a")
        player_names = []
        player_urls = []
        for link in player_links:
            href = link.get("href", "")
            if "/players/" in href:
                player_names.append(link.get_text(strip=True))
                player_urls.append(href)

        tx_type = "unknown"
        if is_il:
            if "placed" in text.lower() or "transferred" in text.lower():
                tx_type = "il_placement"
            elif "activated" in text.lower():
                tx_type = "il_activation"
        elif is_trade:
            tx_type = "trade"
        elif is_roster:
            if "called up" in text.lower() or "recalled" in text.lower() or "selected" in text.lower():
                tx_type = "callup"
            elif "optioned" in text.lower() or "outrighted" in text.lower():
                tx_type = "sent_down"
            elif "designated" in text.lower() or "released" in text.lower():
                tx_type = "dfa_release"
            elif "claimed" in text.lower():
                tx_type = "waiver_claim"

        for pname, purl in zip(player_names, player_urls):
            # Extract player ID from URL
            pid_match = re.search(r"/players/\w/(\w+)\.shtml", purl)
            pid = pid_match.group(1) if pid_match else ""

            transactions.append({
                "date": date_str,
                "year": year,
                "team": normalize_team_name(team),
                "player_name": pname,
                "player_id": pid,
                "player_url": purl,
                "tx_type": tx_type,
                "description": text[:300],
            })

    return transactions


def main():
    session = RateLimitedSession(delay=4.0)
    progress_path = DATA_RAW / "injuries_progress.ndjson"
    progress = load_progress(progress_path)
    scraped = {f"{r['team']}_{r['year']}" for r in progress}

    all_tx = []

    for year in range(START_YEAR, END_YEAR + 1):
        for team in TEAMS:
            key = f"{team}_{year}"
            if key in scraped:
                continue

            txs = scrape_team_transactions(session, team, year)
            all_tx.extend(txs)

            progress.append({"team": team, "year": year, "count": len(txs)})
            if len(progress) % 20 == 0:
                save_progress(progress_path, progress)

        logger.info(f"Completed injury/transaction scraping for {year}")

    save_progress(progress_path, progress)

    # Save
    tx_path = DATA_RAW / "injuries.csv"
    tx_df = pd.DataFrame(all_tx)

    if tx_path.exists():
        existing = pd.read_csv(tx_path)
        tx_df = pd.concat([existing, tx_df], ignore_index=True)
        tx_df = tx_df.drop_duplicates(subset=["date", "team", "player_id", "tx_type"], keep="last")

    tx_df.to_csv(tx_path, index=False)
    logger.info(f"Saved {len(tx_df)} injury/roster transactions to {tx_path}")


if __name__ == "__main__":
    main()
