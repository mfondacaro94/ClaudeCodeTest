"""MLB stadium coordinates, metadata, and travel distance calculations.

Used for:
- Weather lookups (lat/lon)
- Travel distance features (how far is the away team from home)
- Roof type (open-air vs retractable vs dome affects weather relevance)
"""

import math

# Stadium data: lat, lon, elevation_ft, roof type, time zone offset from ET
# roof: "open" = fully outdoor, "retractable" = can close, "dome" = always closed
STADIUMS = {
    "ARI": {"name": "Chase Field", "lat": 33.4455, "lon": -112.0667, "elev": 1082, "roof": "retractable", "tz_offset": -2, "city": "Phoenix"},
    "ATL": {"name": "Truist Park", "lat": 33.8907, "lon": -84.4677, "elev": 1050, "roof": "open", "tz_offset": 0, "city": "Atlanta"},
    "BAL": {"name": "Camden Yards", "lat": 39.2838, "lon": -76.6216, "elev": 30, "roof": "open", "tz_offset": 0, "city": "Baltimore"},
    "BOS": {"name": "Fenway Park", "lat": 42.3467, "lon": -71.0972, "elev": 20, "roof": "open", "tz_offset": 0, "city": "Boston"},
    "CHC": {"name": "Wrigley Field", "lat": 41.9484, "lon": -87.6553, "elev": 595, "roof": "open", "tz_offset": -1, "city": "Chicago"},
    "CHW": {"name": "Guaranteed Rate Field", "lat": 41.8300, "lon": -87.6339, "elev": 595, "roof": "open", "tz_offset": -1, "city": "Chicago"},
    "CIN": {"name": "Great American Ball Park", "lat": 39.0974, "lon": -84.5065, "elev": 490, "roof": "open", "tz_offset": 0, "city": "Cincinnati"},
    "CLE": {"name": "Progressive Field", "lat": 41.4962, "lon": -81.6852, "elev": 653, "roof": "open", "tz_offset": 0, "city": "Cleveland"},
    "COL": {"name": "Coors Field", "lat": 39.7559, "lon": -104.9942, "elev": 5200, "roof": "open", "tz_offset": -2, "city": "Denver"},
    "DET": {"name": "Comerica Park", "lat": 42.3390, "lon": -83.0485, "elev": 600, "roof": "open", "tz_offset": 0, "city": "Detroit"},
    "HOU": {"name": "Minute Maid Park", "lat": 29.7573, "lon": -95.3555, "elev": 42, "roof": "retractable", "tz_offset": -1, "city": "Houston"},
    "KCR": {"name": "Kauffman Stadium", "lat": 39.0517, "lon": -94.4803, "elev": 820, "roof": "open", "tz_offset": -1, "city": "Kansas City"},
    "LAA": {"name": "Angel Stadium", "lat": 33.8003, "lon": -117.8827, "elev": 160, "roof": "open", "tz_offset": -3, "city": "Anaheim"},
    "LAD": {"name": "Dodger Stadium", "lat": 34.0739, "lon": -118.2400, "elev": 515, "roof": "open", "tz_offset": -3, "city": "Los Angeles"},
    "MIA": {"name": "LoanDepot Park", "lat": 25.7781, "lon": -80.2196, "elev": 7, "roof": "retractable", "tz_offset": 0, "city": "Miami"},
    "MIL": {"name": "American Family Field", "lat": 43.0280, "lon": -87.9712, "elev": 635, "roof": "retractable", "tz_offset": -1, "city": "Milwaukee"},
    "MIN": {"name": "Target Field", "lat": 44.9818, "lon": -93.2776, "elev": 815, "roof": "open", "tz_offset": -1, "city": "Minneapolis"},
    "NYM": {"name": "Citi Field", "lat": 40.7571, "lon": -73.8458, "elev": 15, "roof": "open", "tz_offset": 0, "city": "New York"},
    "NYY": {"name": "Yankee Stadium", "lat": 40.8296, "lon": -73.9262, "elev": 45, "roof": "open", "tz_offset": 0, "city": "New York"},
    "OAK": {"name": "Oakland Coliseum", "lat": 37.7516, "lon": -122.2006, "elev": 5, "roof": "open", "tz_offset": -3, "city": "Oakland"},
    "PHI": {"name": "Citizens Bank Park", "lat": 39.9061, "lon": -75.1665, "elev": 20, "roof": "open", "tz_offset": 0, "city": "Philadelphia"},
    "PIT": {"name": "PNC Park", "lat": 40.4469, "lon": -80.0057, "elev": 730, "roof": "open", "tz_offset": 0, "city": "Pittsburgh"},
    "SDP": {"name": "Petco Park", "lat": 32.7076, "lon": -117.1570, "elev": 17, "roof": "open", "tz_offset": -3, "city": "San Diego"},
    "SEA": {"name": "T-Mobile Park", "lat": 47.5914, "lon": -122.3326, "elev": 10, "roof": "retractable", "tz_offset": -3, "city": "Seattle"},
    "SFG": {"name": "Oracle Park", "lat": 37.7786, "lon": -122.3893, "elev": 5, "roof": "open", "tz_offset": -3, "city": "San Francisco"},
    "STL": {"name": "Busch Stadium", "lat": 38.6226, "lon": -90.1928, "elev": 455, "roof": "open", "tz_offset": -1, "city": "St. Louis"},
    "TBR": {"name": "Tropicana Field", "lat": 27.7682, "lon": -82.6534, "elev": 42, "roof": "dome", "tz_offset": 0, "city": "St. Petersburg"},
    "TEX": {"name": "Globe Life Field", "lat": 32.7473, "lon": -97.0845, "elev": 545, "roof": "retractable", "tz_offset": -1, "city": "Arlington"},
    "TOR": {"name": "Rogers Centre", "lat": 43.6414, "lon": -79.3894, "elev": 266, "roof": "retractable", "tz_offset": 0, "city": "Toronto"},
    "WSN": {"name": "Nationals Park", "lat": 38.8730, "lon": -77.0074, "elev": 25, "roof": "open", "tz_offset": 0, "city": "Washington"},
}


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great-circle distance between two points in miles."""
    R = 3959  # Earth radius in miles

    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return R * c


def travel_distance(away_team: str, home_team: str) -> float:
    """Calculate travel distance in miles between two teams' stadiums.

    Returns 0 if either team is not found.
    """
    away = STADIUMS.get(away_team)
    home = STADIUMS.get(home_team)
    if not away or not home:
        return 0.0
    return round(haversine_miles(away["lat"], away["lon"], home["lat"], home["lon"]), 1)


def timezone_change(away_team: str, home_team: str) -> int:
    """Hours of timezone change for the away team (negative = traveling west)."""
    away = STADIUMS.get(away_team)
    home = STADIUMS.get(home_team)
    if not away or not home:
        return 0
    return home["tz_offset"] - away["tz_offset"]


def is_dome_or_retractable(team: str) -> bool:
    """Check if a team's stadium has a roof (weather less relevant)."""
    stadium = STADIUMS.get(team, {})
    return stadium.get("roof", "open") in ("dome", "retractable")


def elevation_ft(team: str) -> int:
    """Get stadium elevation in feet (Coors Field = 5200 ft, huge impact)."""
    return STADIUMS.get(team, {}).get("elev", 0)
