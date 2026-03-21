"""Parse baseball-reference HTML tables and normalize data."""

import re
import pandas as pd
from bs4 import BeautifulSoup, Comment


def parse_br_table(html: str, table_id: str) -> pd.DataFrame:
    """Extract a table from baseball-reference HTML by its id.

    Baseball-reference wraps many tables in HTML comments for lazy loading.
    This function checks both visible tables and commented-out ones.
    """
    soup = BeautifulSoup(html, "lxml")

    # Try visible table first
    table = soup.find("table", id=table_id)

    # If not found, search inside HTML comments
    if table is None:
        for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
            if f'id="{table_id}"' in comment:
                comment_soup = BeautifulSoup(comment, "lxml")
                table = comment_soup.find("table", id=table_id)
                if table:
                    break

    if table is None:
        return pd.DataFrame()

    rows = []
    headers = []

    # Extract headers from thead
    thead = table.find("thead")
    if thead:
        header_row = thead.find_all("tr")[-1]  # last row (skip over_header)
        for th in header_row.find_all("th"):
            stat = th.get("data-stat", th.get_text(strip=True))
            headers.append(stat)

    # Extract data rows from tbody
    tbody = table.find("tbody")
    if tbody:
        for tr in tbody.find_all("tr"):
            if "thead" in tr.get("class", []) or "spacer" in tr.get("class", []):
                continue
            if tr.get("class") and "partial_table" in tr["class"]:
                continue
            row = {}
            for cell in tr.find_all(["th", "td"]):
                stat = cell.get("data-stat", "")
                # Use csk (sort key) for numeric precision when available
                csk = cell.get("csk")
                if csk and stat not in ("year_id", "year_ID", "team_name_abbr",
                                         "team_ID", "comp_name_abbr", "pos",
                                         "awards", "series_result",
                                         "name_display", "player", "ranker"):
                    row[stat] = csk
                else:
                    # Get text, stripping bold/italic wrappers
                    text = cell.get_text(strip=True)
                    # Also grab any href for player/team links
                    link = cell.find("a")
                    if link and link.get("href"):
                        row[f"{stat}_url"] = link["href"]
                    row[stat] = text
            if row:
                rows.append(row)

    # Extract footer rows from tfoot
    tfoot = table.find("tfoot")
    footer_rows = []
    if tfoot:
        for tr in tfoot.find_all("tr"):
            if "spacer" in tr.get("class", []):
                continue
            row = {}
            for cell in tr.find_all(["th", "td"]):
                stat = cell.get("data-stat", "")
                csk = cell.get("csk")
                if csk:
                    row[stat] = csk
                else:
                    row[stat] = cell.get_text(strip=True)
            if row:
                footer_rows.append(row)

    df = pd.DataFrame(rows)
    return df


def normalize_team_name(name: str) -> str:
    """Normalize team abbreviations to a standard set."""
    aliases = {
        "ANA": "LAA", "CAL": "LAA", "MON": "WSN", "FLA": "MIA",
        "TBD": "TBR", "TB": "TBR", "CWS": "CHW", "CHC": "CHC",
        "KC": "KCR", "SD": "SDP", "SF": "SFG", "LAD": "LAD",
        "WSH": "WSN", "WAS": "WSN", "ATH": "OAK",
    }
    name = name.strip().upper()
    return aliases.get(name, name)


def parse_innings_pitched(ip_str: str) -> float:
    """Convert innings pitched string (e.g., '6.1') to decimal innings."""
    try:
        if pd.isna(ip_str) or ip_str == "":
            return 0.0
        ip_str = str(ip_str).strip()
        if "." in ip_str:
            whole, frac = ip_str.split(".")
            return int(whole) + int(frac) / 3.0
        return float(ip_str)
    except (ValueError, TypeError):
        return 0.0


def safe_float(val, default=None):
    """Safely convert a value to float."""
    try:
        if pd.isna(val) or val == "" or val is None:
            return default
        return float(val)
    except (ValueError, TypeError):
        return default
