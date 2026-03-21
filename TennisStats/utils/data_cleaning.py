"""Tennis-specific data cleaning: name normalization, score parsing, geo data."""

import re
import math
import unicodedata


def strip_accents(s: str) -> str:
    """Remove diacritics/accents from a string."""
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def normalize_player_name(name: str) -> str:
    """Normalize player name to 'lastname f' lowercase key for cross-source matching.

    Handles:
    - tennis-data.co.uk format: 'Djokovic N.' -> 'djokovic n'
    - Sackmann format: 'Novak Djokovic' -> 'djokovic n'
    - Accented names: 'Möller K.' -> 'moller k'
    """
    if not name or not isinstance(name, str):
        return ""

    name = strip_accents(name.strip())

    # tennis-data.co.uk format: "LastName F." or "Del Potro J.M."
    if re.search(r"\b[A-Z]\.$", name.strip()):
        # Already in LastName Initial format
        parts = name.replace(".", "").strip().split()
        if len(parts) >= 2:
            initials = parts[-1]  # last part is initial(s)
            lastname = " ".join(parts[:-1])
            return f"{lastname} {initials[0]}".lower()

    # Sackmann format: "Novak Djokovic" (first last)
    parts = name.strip().split()
    if len(parts) >= 2:
        first_initial = parts[0][0]
        lastname = " ".join(parts[1:])
        return f"{lastname} {first_initial}".lower()

    return name.lower()


def parse_score(score_str: str) -> dict:
    """Parse match score string into structured data.

    Examples: '6-4 3-6 7-6(5)', '6-3 6-4', '7-6(4) 3-6 6-3 6-7(2) 6-4'
    Returns dict with n_sets, total_games, tiebreaks, was_retirement, was_walkover.
    """
    result = {
        "n_sets": 0,
        "total_games": 0,
        "tiebreaks": 0,
        "was_retirement": False,
        "was_walkover": False,
    }

    if not score_str or not isinstance(score_str, str):
        return result

    score_str = score_str.strip()

    if "W/O" in score_str.upper() or "WALKOVER" in score_str.upper():
        result["was_walkover"] = True
        return result

    if "RET" in score_str.upper() or "DEF" in score_str.upper():
        result["was_retirement"] = True

    # Extract set scores like "6-4", "7-6(5)"
    sets = re.findall(r"(\d+)-(\d+)(?:\((\d+)\))?", score_str)
    result["n_sets"] = len(sets)

    for w_games, l_games, tb in sets:
        result["total_games"] += int(w_games) + int(l_games)
        if tb:
            result["tiebreaks"] += 1

    return result


def safe_float(val, default=None) -> float:
    """Safely convert a value to float."""
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return default
        return v
    except (ValueError, TypeError):
        return default


# --- Geographic data for distance-to-home feature ---

# Country centroids (lat, lon) keyed by IOC/ISO3 codes
COUNTRY_COORDS = {
    "AFG": (33.9, 67.7), "ALB": (41.2, 20.2), "ALG": (28.0, 2.6),
    "AND": (42.5, 1.5), "ANG": (-12.3, 17.9), "ANT": (17.1, -61.8),
    "ARG": (-34.6, -58.4), "ARM": (40.2, 44.5), "AUS": (-33.9, 151.2),
    "AUT": (48.2, 16.4), "AZE": (40.4, 49.9), "BAH": (25.1, -77.3),
    "BAN": (23.7, 90.4), "BAR": (13.1, -59.6), "BDI": (-3.4, 29.4),
    "BEL": (50.8, 4.4), "BEN": (6.5, 2.6), "BER": (32.3, -64.8),
    "BIH": (43.9, 17.7), "BLR": (53.9, 27.6), "BOL": (-16.5, -68.1),
    "BOT": (-24.7, 25.9), "BRA": (-23.5, -46.6), "BRN": (4.9, 114.9),
    "BUL": (42.7, 23.3), "BUR": (12.4, -1.5), "CAM": (11.6, 104.9),
    "CAN": (45.4, -75.7), "CGO": (-4.3, 15.3), "CHI": (-33.4, -70.7),
    "CHN": (39.9, 116.4), "CIV": (5.3, -4.0), "CMR": (3.9, 11.5),
    "COL": (4.7, -74.1), "CRC": (9.9, -84.1), "CRO": (45.8, 16.0),
    "CUB": (23.1, -82.4), "CYP": (35.2, 33.4), "CZE": (50.1, 14.4),
    "DEN": (55.7, 12.6), "DOM": (18.5, -69.9), "ECU": (-0.2, -78.5),
    "EGY": (30.0, 31.2), "ESA": (13.7, -89.2), "ESP": (40.4, -3.7),
    "EST": (59.4, 24.7), "ETH": (9.0, 38.7), "FIN": (60.2, 24.9),
    "FRA": (48.9, 2.3), "GAB": (0.4, 9.5), "GBR": (51.5, -0.1),
    "GEO": (41.7, 44.8), "GER": (52.5, 13.4), "GHA": (5.6, -0.2),
    "GRE": (37.9, 23.7), "GUA": (14.6, -90.5), "GUM": (13.4, 144.8),
    "HAI": (18.5, -72.3), "HKG": (22.3, 114.2), "HON": (14.1, -87.2),
    "HUN": (47.5, 19.1), "INA": (-6.2, 106.8), "IND": (28.6, 77.2),
    "IRI": (35.7, 51.4), "IRL": (53.3, -6.3), "IRQ": (33.3, 44.4),
    "ISL": (64.1, -21.9), "ISR": (32.1, 34.8), "ISV": (18.3, -64.9),
    "ITA": (41.9, 12.5), "JAM": (18.0, -76.8), "JOR": (31.9, 35.9),
    "JPN": (35.7, 139.7), "KAZ": (51.2, 71.4), "KEN": (-1.3, 36.8),
    "KGZ": (42.9, 74.6), "KOR": (37.6, 127.0), "KSA": (24.7, 46.7),
    "KUW": (29.4, 47.9), "LAT": (56.9, 24.1), "LBA": (32.9, 13.2),
    "LBN": (33.9, 35.5), "LIE": (47.1, 9.5), "LTU": (54.7, 25.3),
    "LUX": (49.6, 6.1), "MAD": (-18.9, 47.5), "MAR": (33.9, -6.8),
    "MAS": (3.1, 101.7), "MDA": (47.0, 28.8), "MEX": (19.4, -99.1),
    "MGL": (47.9, 106.9), "MKD": (42.0, 21.4), "MLI": (12.6, -8.0),
    "MLT": (35.9, 14.5), "MNE": (42.4, 19.3), "MON": (43.7, 7.4),
    "MOZ": (-25.9, 32.6), "MRI": (-20.2, 57.5), "NAM": (-22.6, 17.1),
    "NED": (52.4, 4.9), "NEP": (27.7, 85.3), "NGR": (9.1, 7.5),
    "NIG": (13.5, 2.1), "NOR": (59.9, 10.8), "NZL": (-41.3, 174.8),
    "OMA": (23.6, 58.5), "PAK": (33.7, 73.0), "PAN": (9.0, -79.5),
    "PAR": (-25.3, -57.6), "PER": (-12.0, -77.0), "PHI": (14.6, 121.0),
    "POL": (52.2, 21.0), "POR": (38.7, -9.1), "PRK": (39.0, 125.8),
    "PUR": (18.5, -66.1), "QAT": (25.3, 51.5), "ROU": (44.4, 26.1),
    "RSA": (-33.9, 18.4), "RUS": (55.8, 37.6), "RWA": (-1.9, 30.1),
    "SEN": (14.7, -17.5), "SGP": (1.3, 103.8), "SLE": (8.5, -13.2),
    "SLO": (46.1, 14.5), "SMR": (43.9, 12.4), "SRB": (44.8, 20.5),
    "SRI": (6.9, 79.9), "SUD": (15.6, 32.5), "SUI": (46.9, 7.4),
    "SVK": (48.1, 17.1), "SWE": (59.3, 18.1), "SYR": (33.5, 36.3),
    "THA": (13.8, 100.5), "TJK": (38.6, 68.8), "TKM": (37.9, 58.4),
    "TOG": (6.1, 1.2), "TPE": (25.0, 121.5), "TTO": (10.7, -61.5),
    "TUN": (36.8, 10.2), "TUR": (39.9, 32.9), "UAE": (24.5, 54.7),
    "UGA": (0.3, 32.6), "UKR": (50.5, 30.5), "URU": (-34.9, -56.2),
    "USA": (38.9, -77.0), "UZB": (41.3, 69.3), "VEN": (10.5, -66.9),
    "VIE": (21.0, 105.8), "ZAM": (-15.4, 28.3), "ZIM": (-17.8, 31.1),
}

# Major ATP tournament locations (lat, lon)
TOURNAMENT_COORDS = {
    # Grand Slams
    "Australian Open": (-37.82, 144.98),
    "Roland Garros": (48.85, 2.25),
    "Wimbledon": (51.43, -0.21),
    "US Open": (40.75, -73.85),
    # Masters 1000
    "Indian Wells": (33.72, -116.31),
    "Miami": (25.76, -80.19),
    "Monte Carlo": (43.75, 7.33),
    "Madrid": (40.42, -3.70),
    "Rome": (41.90, 12.50),
    "Canada": (45.50, -73.57),  # Montreal/Toronto alternates
    "Montreal": (45.50, -73.57),
    "Toronto": (43.65, -79.38),
    "Cincinnati": (39.10, -84.51),
    "Shanghai": (31.23, 121.47),
    "Paris": (48.85, 2.35),
    "Paris Masters": (48.85, 2.35),
    # ATP 500
    "Rotterdam": (51.92, 4.48),
    "Rio de Janeiro": (-22.97, -43.17),
    "Dubai": (25.20, 55.27),
    "Acapulco": (16.86, -99.88),
    "Barcelona": (41.39, 2.17),
    "London": (51.50, -0.13),
    "Queen's Club": (51.49, -0.21),
    "Halle": (52.06, 8.36),
    "Hamburg": (53.55, 9.99),
    "Washington": (38.90, -77.04),
    "Beijing": (39.91, 116.39),
    "Tokyo": (35.68, 139.69),
    "Vienna": (48.21, 16.37),
    "Basel": (47.56, 7.59),
    # ATP 250 (common venues)
    "Brisbane": (-27.47, 153.03),
    "Doha": (25.29, 51.53),
    "Adelaide": (-34.93, 138.60),
    "Auckland": (-36.85, 174.76),
    "Pune": (18.52, 73.86),
    "Montpellier": (43.61, 3.88),
    "Dallas": (32.78, -96.80),
    "Marseille": (43.30, 5.37),
    "Santiago": (-33.45, -70.67),
    "Buenos Aires": (-34.60, -58.38),
    "Los Cabos": (22.89, -109.92),
    "Atlanta": (33.75, -84.39),
    "Kitzbuhel": (47.45, 12.39),
    "Umag": (45.43, 13.52),
    "Winston-Salem": (36.10, -80.24),
    "Metz": (49.12, 6.18),
    "Astana": (51.17, 71.43),
    "Sofia": (42.70, 23.32),
    "Antwerp": (51.22, 4.40),
    "Stockholm": (59.33, 18.07),
    "St. Petersburg": (59.93, 30.32),
    "Moscow": (55.76, 37.62),
    "Florence": (43.77, 11.25),
    "Nur-Sultan": (51.17, 71.43),
    "Lyon": (45.76, 4.84),
    "Geneva": (46.20, 6.14),
    "Stuttgart": (48.78, 9.18),
    "Eastbourne": (50.77, 0.29),
    "Mallorca": (39.57, 2.65),
    "Newport": (41.49, -71.31),
    "Bastad": (56.43, 12.85),
    "Gstaad": (46.47, 7.29),
    "Chengdu": (30.57, 104.07),
    "Zhuhai": (22.27, 113.58),
    "San Diego": (32.72, -117.16),
    "Gijon": (43.54, -5.66),
    "Naples": (40.85, 14.27),
    "Tel Aviv": (32.08, 34.78),
    "Seoul": (37.57, 126.98),
    # ATP Finals
    "ATP Finals": (45.47, 9.19),  # Turin
    "Turin": (45.47, 9.19),
}


def match_tournament_coords(tournament_name: str, location: str = "") -> tuple:
    """Find coordinates for a tournament by name or location.

    Returns (lat, lon) or (None, None) if not found.
    """
    if not tournament_name:
        return None, None

    name = tournament_name.strip()

    # Direct match
    if name in TOURNAMENT_COORDS:
        return TOURNAMENT_COORDS[name]

    # Partial match on tournament name
    name_lower = name.lower()
    for key, coords in TOURNAMENT_COORDS.items():
        if key.lower() in name_lower or name_lower in key.lower():
            return coords

    # Try location field
    if location:
        loc_lower = location.strip().lower()
        for key, coords in TOURNAMENT_COORDS.items():
            if key.lower() in loc_lower or loc_lower in key.lower():
                return coords

    return None, None


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute great-circle distance between two points in km."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))


def compute_distance_to_home(country_code: str, tournament_name: str,
                              location: str = "") -> float:
    """Compute distance from player's home country to tournament venue.

    Returns distance in km, or None if either location is unknown.
    """
    if not country_code or country_code not in COUNTRY_COORDS:
        return None

    t_lat, t_lon = match_tournament_coords(tournament_name, location)
    if t_lat is None:
        return None

    h_lat, h_lon = COUNTRY_COORDS[country_code]
    return haversine_km(h_lat, h_lon, t_lat, t_lon)
