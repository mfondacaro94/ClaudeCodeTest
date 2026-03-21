"""Feature engineering: raw data -> master.csv -> ml_ready.csv.

Transforms scraped game results, team stats, pitcher stats, and batter stats
into ML-ready matchup features.

Feature groups:
1. Rolling team performance (10/20/40 game windows)
2. Starting pitcher game-level + season stats
3. Team batting aggregate stats (from roster-level batter data)
4. Travel distance & timezone change for away team
5. Rest days & series position
6. Time of day, day of week, season progression
7. Diff/ratio matchup features across all of the above
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.helpers import get_logger, DATA_RAW, DATA_PROCESSED
from utils.data_cleaning import normalize_team_name, safe_float
from utils.stadium_data import travel_distance, timezone_change

logger = get_logger("feature_engineering")

ROLLING_WINDOWS = [10, 20, 40]


# ── Data Loaders ─────────────────────────────────────────────────────────

def load_games() -> pd.DataFrame:
    path = DATA_RAW / "games.csv"
    if not path.exists():
        logger.error(f"Games file not found: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "home_team", "away_team"])
    df = df.sort_values("date").reset_index(drop=True)

    df["home_runs"] = pd.to_numeric(df["home_runs"], errors="coerce")
    df["away_runs"] = pd.to_numeric(df["away_runs"], errors="coerce")
    df["home_win"] = pd.to_numeric(df["home_win"], errors="coerce")
    df = df.dropna(subset=["home_runs", "away_runs", "home_win"])

    logger.info(f"Loaded {len(df)} games")
    return df


def load_pitchers() -> pd.DataFrame:
    path = DATA_RAW / "pitchers.csv"
    if not path.exists():
        logger.warning("No pitchers.csv — skipping pitcher features")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} pitcher-season rows")
    return df


def load_batters() -> pd.DataFrame:
    path = DATA_RAW / "batters.csv"
    if not path.exists():
        logger.warning("No batters.csv — skipping batter features")
        return pd.DataFrame()
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} batter-season rows")
    return df


def load_weather() -> pd.DataFrame:
    path = DATA_RAW / "weather.csv"
    if not path.exists():
        logger.warning("No weather.csv — skipping weather features")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info(f"Loaded {len(df)} weather records")
    return df


def load_injuries() -> pd.DataFrame:
    path = DATA_RAW / "injuries.csv"
    if not path.exists():
        logger.warning("No injuries.csv — skipping injury features")
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    logger.info(f"Loaded {len(df)} injury/transaction records")
    return df


# ── Feature Builders ─────────────────────────────────────────────────────

def compute_rolling_team_stats(games: pd.DataFrame) -> pd.DataFrame:
    """Rolling window stats per team. Shifted by 1 to prevent leakage."""
    home_logs = games[["date", "home_team", "home_runs", "away_runs", "home_win"]].copy()
    home_logs.columns = ["date", "team", "runs", "runs_allowed", "won"]

    away_logs = games[["date", "away_team", "away_runs", "home_runs", "home_win"]].copy()
    away_logs.columns = ["date", "team", "runs", "runs_allowed", "won"]
    away_logs["won"] = 1 - away_logs["won"]

    team_log = pd.concat([home_logs, away_logs], ignore_index=True)
    team_log = team_log.sort_values(["team", "date"]).reset_index(drop=True)
    team_log["run_diff"] = team_log["runs"] - team_log["runs_allowed"]

    rolling_dfs = []
    for window in ROLLING_WINDOWS:
        grouped = team_log.groupby("team")
        roll = pd.DataFrame({"date": team_log["date"], "team": team_log["team"]})

        for col in ["runs", "runs_allowed", "run_diff", "won"]:
            roll[f"roll{window}_{col}"] = grouped[col].transform(
                lambda x: x.shift(1).rolling(window, min_periods=3).mean()
            )
        rolling_dfs.append(roll)

    result = rolling_dfs[0]
    for rdf in rolling_dfs[1:]:
        result = result.merge(rdf, on=["date", "team"], how="outer")

    # Season-to-date stats
    year_col = team_log["date"].dt.year
    team_log["game_num"] = team_log.groupby(["team", year_col]).cumcount() + 1
    szn_grouped = team_log.groupby(["team", year_col])

    team_log["szn_runs_pg"] = szn_grouped["runs"].transform(lambda x: x.shift(1).expanding().mean())
    team_log["szn_ra_pg"] = szn_grouped["runs_allowed"].transform(lambda x: x.shift(1).expanding().mean())
    team_log["szn_win_pct"] = szn_grouped["won"].transform(lambda x: x.shift(1).expanding().mean())

    result = result.merge(
        team_log[["date", "team", "game_num", "szn_runs_pg", "szn_ra_pg", "szn_win_pct"]],
        on=["date", "team"], how="left"
    )
    result = result.drop_duplicates(subset=["date", "team"], keep="last")
    return result


def build_pitcher_features(pitchers: pd.DataFrame) -> pd.DataFrame:
    """Per-pitcher season-level features."""
    if pitchers.empty:
        return pd.DataFrame()

    cols_map = {
        "player_id": "sp_id", "year_id": "year", "team_name_abbr": "team",
        "p_earned_run_avg": "sp_era", "p_fip": "sp_fip", "p_whip": "sp_whip",
        "p_so_per_nine": "sp_k9", "p_bb_per_nine": "sp_bb9",
        "p_hr_per_nine": "sp_hr9", "p_ip": "sp_ip", "p_war": "sp_war",
        "p_w": "sp_wins", "p_l": "sp_losses",
        "p_strikeouts_per_base_on_balls": "sp_k_bb",
        "p_earned_run_avg_plus": "sp_era_plus", "p_gs": "sp_gs",
    }
    available = [c for c in cols_map if c in pitchers.columns]
    pdf = pitchers[available].rename(columns={c: cols_map[c] for c in available}).copy()
    for col in pdf.columns:
        if col not in ("sp_id", "team", "year"):
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")
    return pdf


def build_sp_features(games: pd.DataFrame, pitcher_feats: pd.DataFrame) -> pd.DataFrame:
    """Match starting pitchers to games and add their stats.

    Uses pitcher_gamelogs.csv (from MLB Stats API) to identify who started
    each game, then merges season-level pitcher stats and game-level context.
    """
    gl_path = DATA_RAW / "pitcher_gamelogs.csv"
    if not gl_path.exists():
        logger.warning("No pitcher_gamelogs.csv — skipping SP features.")
        return games

    gl = pd.read_csv(gl_path)
    gl["date"] = pd.to_datetime(gl["date"], errors="coerce")
    games["date"] = pd.to_datetime(games["date"], errors="coerce")

    # Filter to starters only
    sp = gl[gl["is_start"] == True].copy()

    for prefix, is_home_val in [("home", True), ("away", False)]:
        side = sp[sp["is_home"] == is_home_val].copy()
        # Keep only one SP per game (first listed)
        side = side.drop_duplicates(subset=["date", "home_team", "away_team"], keep="first")

        # ONLY use pre-game info — NOT game-day stats (ip, er, so, bb, hr, pitches)
        # Those are outcomes, not predictors — using them is data leakage!
        rename = {
            "player_id": f"{prefix}_sp_id",
            "throws": f"{prefix}_sp_throws",
            "era_cume": f"{prefix}_sp_era_cume",
        }
        available = {k: v for k, v in rename.items() if k in side.columns}
        side = side[["date", "home_team", "away_team"] + list(available.keys())]
        side = side.rename(columns=available)

        games = games.merge(side, on=["date", "home_team", "away_team"], how="left")

    # Convert SP throws to binary (1 = lefty)
    for prefix in ["home", "away"]:
        col = f"{prefix}_sp_throws"
        if col in games.columns:
            games[f"{prefix}_sp_is_lefty"] = (games[col] == "L").astype(int)
            games = games.drop(columns=[col])

    # Note: season-level pitcher stats (from BR) use different IDs than MLB API game logs.
    # We rely on the game-level stats (IP, ER, SO, BB, pitches, era_cume) from the API
    # which are already merged above and more granular than season averages.

    home_matched = games.get("home_sp_id", pd.Series()).notna().sum()
    away_matched = games.get("away_sp_id", pd.Series()).notna().sum()
    logger.info(f"SP match rate: home={home_matched}/{len(games)} ({home_matched/len(games):.1%}), "
                f"away={away_matched}/{len(games)} ({away_matched/len(games):.1%})")

    return games


def build_bullpen_features(games: pd.DataFrame) -> pd.DataFrame:
    """Compute bullpen workload: total IP pitched by relievers in last 3 days per team."""
    gl_path = DATA_RAW / "pitcher_gamelogs.csv"
    if not gl_path.exists():
        logger.warning("No pitcher_gamelogs.csv — skipping bullpen features.")
        return games

    gl = pd.read_csv(gl_path)
    gl["date"] = pd.to_datetime(gl["date"], errors="coerce")

    # Relievers only
    relievers = gl[gl["is_start"] == False].copy()
    relievers["ip"] = pd.to_numeric(relievers["ip"], errors="coerce").fillna(0)

    # Sum reliever IP per team per date
    daily_bp = relievers.groupby(["date", "team"])["ip"].sum().reset_index()
    daily_bp = daily_bp.rename(columns={"ip": "bp_ip"})

    # Compute rolling 3-day bullpen workload per team
    daily_bp = daily_bp.sort_values(["team", "date"])
    daily_bp["bp_ip_last3"] = daily_bp.groupby("team")["bp_ip"].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).sum()
    )

    # Merge into games
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        bp = daily_bp[["date", "team", "bp_ip_last3"]].copy()
        bp = bp.rename(columns={"team": team_col, "bp_ip_last3": f"{prefix}_bp_ip_last3"})
        games = games.merge(bp, on=["date", team_col], how="left")

    logger.info(f"Bullpen features: {games['home_bp_ip_last3'].notna().sum()}/{len(games)} matched")
    return games


def build_umpire_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add home plate umpire run-scoring tendency."""
    ump_path = DATA_RAW / "umpires.csv"
    if not ump_path.exists():
        logger.warning("No umpires.csv — skipping umpire features.")
        return games

    umps = pd.read_csv(ump_path)
    umps["date"] = pd.to_datetime(umps["date"], errors="coerce")

    # Merge umpire ID into games
    games = games.merge(
        umps[["date", "home_team", "away_team", "hp_umpire_id"]],
        on=["date", "home_team", "away_team"], how="left"
    )

    # Compute umpire's historical avg total runs (shifted to prevent leakage)
    # Use games data which has runs
    if "home_runs" in games.columns and "away_runs" in games.columns:
        games["_total_runs"] = games["home_runs"] + games["away_runs"]
        games = games.sort_values("date")

        ump_avgs = games.groupby("hp_umpire_id")["_total_runs"].transform(
            lambda x: x.shift(1).expanding().mean()
        )
        games["ump_avg_runs"] = ump_avgs
        games = games.drop(columns=["_total_runs"])

    matched = games["hp_umpire_id"].notna().sum()
    logger.info(f"Umpire match rate: {matched}/{len(games)} ({matched/len(games):.1%})")
    return games


def build_park_factor_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add park factors (BPF/PPF) for the home stadium."""
    pf_path = DATA_RAW / "park_factors.csv"
    if not pf_path.exists():
        logger.warning("No park_factors.csv — skipping park factor features.")
        return games

    pf = pd.read_csv(pf_path)
    games["_year"] = games["date"].dt.year

    games = games.merge(
        pf.rename(columns={"team": "home_team", "year": "_year"}),
        on=["home_team", "_year"], how="left"
    )
    games = games.drop(columns=["_year"])

    # Fill missing with neutral (100)
    games["bpf"] = games["bpf"].fillna(100)
    games["ppf"] = games["ppf"].fillna(100)

    matched = (games["bpf"] != 100).sum()
    logger.info(f"Park factors matched: {matched}/{len(games)} ({matched/len(games):.1%})")
    return games


def build_rest_and_series_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add rest days and series position features.

    - Days since last game for each team
    - Series game number (1st, 2nd, 3rd, 4th of series)
    """
    games = games.sort_values("date").reset_index(drop=True)

    # Build team game log to compute rest days
    home_log = games[["date", "home_team"]].rename(columns={"home_team": "team"})
    away_log = games[["date", "away_team"]].rename(columns={"away_team": "team"})
    team_log = pd.concat([home_log, away_log]).sort_values(["team", "date"])
    team_log["prev_game"] = team_log.groupby("team")["date"].shift(1)
    team_log["rest_days"] = (team_log["date"] - team_log["prev_game"]).dt.days

    # Merge rest days back (home team)
    home_rest = team_log.drop_duplicates(subset=["date", "team"], keep="last")
    home_rest = home_rest.rename(columns={"team": "home_team", "rest_days": "home_rest_days"})
    games = games.merge(home_rest[["date", "home_team", "home_rest_days"]],
                        on=["date", "home_team"], how="left")

    # Away team rest
    away_rest = team_log.drop_duplicates(subset=["date", "team"], keep="last")
    away_rest = away_rest.rename(columns={"team": "away_team", "rest_days": "away_rest_days"})
    games = games.merge(away_rest[["date", "away_team", "away_rest_days"]],
                        on=["date", "away_team"], how="left")

    # Series position: consecutive games between same two teams
    games = games.sort_values("date").reset_index(drop=True)
    matchup = games.apply(
        lambda r: tuple(sorted([r["home_team"], r["away_team"]])), axis=1
    )
    games["_matchup"] = matchup
    games["_prev_matchup"] = games["_matchup"].shift(1)
    games["_new_series"] = (games["_matchup"] != games["_prev_matchup"]).astype(int)

    # Compute series game number
    series_num = []
    current = 0
    for is_new in games["_new_series"]:
        if is_new:
            current = 1
        else:
            current += 1
        series_num.append(current)
    games["series_game_num"] = series_num

    games = games.drop(columns=["_matchup", "_prev_matchup", "_new_series"], errors="ignore")

    return games


def build_team_batter_aggregates(batters: pd.DataFrame) -> pd.DataFrame:
    """Aggregate batter stats per team-season (lineup strength proxy)."""
    if batters.empty:
        return pd.DataFrame()

    num_cols = {
        "b_war": "team_bat_war", "b_hr": "team_bat_hr",
        "b_rbi": "team_bat_rbi", "b_sb": "team_bat_sb",
        "b_batting_avg": "team_bat_avg", "b_onbase_perc": "team_bat_obp",
        "b_slugging_perc": "team_bat_slg", "b_onbase_plus_slugging": "team_bat_ops",
        "b_onbase_plus_slugging_plus": "team_bat_ops_plus",
    }

    for col in num_cols:
        if col in batters.columns:
            batters[col] = pd.to_numeric(batters[col], errors="coerce")

    available = [c for c in num_cols if c in batters.columns]
    if not available:
        return pd.DataFrame()

    # Sum counting stats, weighted-average rate stats
    sum_stats = [c for c in available if c in ("b_war", "b_hr", "b_rbi", "b_sb")]
    rate_stats = [c for c in available if c not in sum_stats]

    agg_dict = {}
    for c in sum_stats:
        agg_dict[c] = "sum"
    for c in rate_stats:
        agg_dict[c] = "mean"  # approximate; proper weighting would use PA

    grouped = batters.groupby(["team_name_abbr", "year_id"]).agg(agg_dict).reset_index()
    grouped = grouped.rename(columns={"team_name_abbr": "team", "year_id": "year"})
    grouped = grouped.rename(columns=num_cols)

    for col in grouped.columns:
        if col not in ("team", "year"):
            grouped[col] = pd.to_numeric(grouped[col], errors="coerce")

    return grouped


def build_weather_features(games: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Merge weather data into games and create weather-aware features."""
    if weather.empty:
        return games

    weather_cols = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "wind_speed_10m", "wind_direction_10m", "surface_pressure",
    ]
    merge_cols = ["date", "home_team"]
    avail = [c for c in weather_cols if c in weather.columns]
    weather_sub = weather[merge_cols + avail].copy()

    # Rename for clarity
    rename = {
        "temperature_2m": "wx_temp_f",
        "relative_humidity_2m": "wx_humidity",
        "precipitation": "wx_precip_in",
        "wind_speed_10m": "wx_wind_mph",
        "wind_direction_10m": "wx_wind_dir",
        "surface_pressure": "wx_pressure_hpa",
    }
    weather_sub = weather_sub.rename(columns={k: v for k, v in rename.items() if k in weather_sub.columns})

    games = games.merge(weather_sub, on=merge_cols, how="left")

    # Weather relevance flag: weather matters less in domed stadiums
    games["wx_outdoor"] = games["home_team"].apply(
        lambda t: 0 if is_dome_or_retractable(t) else 1
    )

    # Interaction: outdoor weather effects
    if "wx_temp_f" in games.columns:
        games["wx_temp_outdoor"] = games["wx_temp_f"] * games["wx_outdoor"]
    if "wx_wind_mph" in games.columns:
        games["wx_wind_outdoor"] = games["wx_wind_mph"] * games["wx_outdoor"]

    return games


def build_travel_features(games: pd.DataFrame) -> pd.DataFrame:
    """Add travel distance and timezone change features."""
    games["travel_dist_miles"] = games.apply(
        lambda r: travel_distance(r["away_team"], r["home_team"]), axis=1
    )
    games["tz_change_hours"] = games.apply(
        lambda r: timezone_change(r["away_team"], r["home_team"]), axis=1
    )
    games["tz_change_abs"] = games["tz_change_hours"].abs()

    return games


def build_injury_features(games: pd.DataFrame, injuries: pd.DataFrame,
                          pitchers: pd.DataFrame, batters: pd.DataFrame) -> pd.DataFrame:
    """Compute team-level injury impact: total WAR currently on IL at game date.

    For each game, look at IL placements and activations before the game date
    to determine which players are on IL, then sum their WAR as an injury cost.
    """
    if injuries.empty:
        logger.info("No injury data — skipping injury features")
        games["home_il_war"] = 0.0
        games["away_il_war"] = 0.0
        games["diff_il_war"] = 0.0
        return games

    # Build a WAR lookup: player_id -> season WAR
    war_lookup = {}
    for df, war_col in [(pitchers, "p_war"), (batters, "b_war")]:
        if df.empty or war_col not in df.columns:
            continue
        for _, row in df.iterrows():
            pid = row.get("player_id", "")
            yr = row.get("year_id", "")
            war = safe_float(row.get(war_col), 0)
            if pid and yr:
                war_lookup[(pid, str(yr))] = war_lookup.get((pid, str(yr)), 0) + war

    # Build IL status per team per date
    il_placements = injuries[injuries["tx_type"] == "il_placement"].copy()
    il_activations = injuries[injuries["tx_type"] == "il_activation"].copy()

    def compute_il_war(team: str, game_date, year: int) -> float:
        """Sum WAR of players on IL for a team at a given date."""
        placed = il_placements[
            (il_placements["team"] == team) &
            (il_placements["date"] <= game_date)
        ]
        activated = il_activations[
            (il_activations["team"] == team) &
            (il_activations["date"] <= game_date)
        ]

        # Players currently on IL = placed but not yet activated
        placed_ids = set(placed["player_id"].dropna())
        activated_ids = set(activated["player_id"].dropna())
        on_il = placed_ids - activated_ids

        total_war = sum(war_lookup.get((pid, str(year)), 0) for pid in on_il)
        return total_war

    # This is expensive for large datasets — compute in batches
    home_il = []
    away_il = []
    for _, game in games.iterrows():
        yr = game["date"].year
        home_il.append(compute_il_war(game["home_team"], game["date"], yr))
        away_il.append(compute_il_war(game["away_team"], game["date"], yr))

    games["home_il_war"] = home_il
    games["away_il_war"] = away_il
    games["diff_il_war"] = games["home_il_war"] - games["away_il_war"]

    return games


def build_trade_features(games: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    """Compute recent trade/roster churn features.

    Teams that just made a big trade or call-up may be disrupted or boosted.
    """
    if injuries.empty:
        games["home_recent_trades"] = 0
        games["away_recent_trades"] = 0
        games["home_recent_callups"] = 0
        games["away_recent_callups"] = 0
        return games

    trades = injuries[injuries["tx_type"] == "trade"]
    callups = injuries[injuries["tx_type"] == "callup"]

    def count_recent(tx_df: pd.DataFrame, team: str, date, days: int = 7) -> int:
        cutoff = date - pd.Timedelta(days=days)
        return len(tx_df[(tx_df["team"] == team) & (tx_df["date"] >= cutoff) & (tx_df["date"] <= date)])

    home_trades = []
    away_trades = []
    home_callups = []
    away_callups = []

    for _, game in games.iterrows():
        d = game["date"]
        home_trades.append(count_recent(trades, game["home_team"], d))
        away_trades.append(count_recent(trades, game["away_team"], d))
        home_callups.append(count_recent(callups, game["home_team"], d))
        away_callups.append(count_recent(callups, game["away_team"], d))

    games["home_recent_trades"] = home_trades
    games["away_recent_trades"] = away_trades
    games["home_recent_callups"] = home_callups
    games["away_recent_callups"] = away_callups
    games["diff_recent_trades"] = games["home_recent_trades"] - games["away_recent_trades"]
    games["diff_recent_callups"] = games["home_recent_callups"] - games["away_recent_callups"]

    return games


def build_time_features(games: pd.DataFrame) -> pd.DataFrame:
    """Day-of-week, weekend, and season progression features."""
    games["day_of_week"] = games["date"].dt.dayofweek
    games["month"] = games["date"].dt.month
    games["is_weekend"] = games["day_of_week"].isin([5, 6]).astype(int)

    # Season progression (0 = opening day, 1 = game 162)
    if "home_game_num" in games.columns:
        games["season_pct"] = games["home_game_num"] / 162.0
    else:
        day_of_year = games["date"].dt.dayofyear
        games["season_pct"] = ((day_of_year - 87) / 185.0).clip(0, 1)

    return games


# ── Matchup Feature Construction ─────────────────────────────────────────

def build_matchup_diffs(games: pd.DataFrame) -> pd.DataFrame:
    """Build diff/ratio features between home and away stats."""
    # CRITICAL: exclude outcome columns and identifiers to prevent data leakage
    leaky = {"home_runs", "home_win", "home_team", "away_team",
             "home_sp_id", "away_sp_id"}
    home_cols = [c for c in games.columns
                 if c.startswith("home_") and c not in leaky
                 and not c.endswith(("_team",))]

    for home_col in home_cols:
        suffix = home_col.replace("home_", "")
        away_col = f"away_{suffix}"
        if away_col in games.columns:
            h = pd.to_numeric(games[home_col], errors="coerce")
            a = pd.to_numeric(games[away_col], errors="coerce")
            games[f"diff_{suffix}"] = h - a
            games[f"ratio_{suffix}"] = np.where(
                (a != 0) & a.notna(), h / a, 1.0
            )

    return games


def create_ml_ready(master: pd.DataFrame) -> pd.DataFrame:
    """Select ML features + target + identifiers for training."""
    # Exclude diff/ratio features derived from dead-weight columns
    dead_weight = {"il_war", "recent_trades", "recent_callups"}

    feature_cols = [c for c in master.columns
                    if (c.startswith(("diff_", "ratio_"))
                        and not any(dw in c for dw in dead_weight))
                    or c in (
                        "day_of_week", "month", "is_weekend", "season_pct",
                        "travel_dist_miles", "tz_change_abs", "tz_change_hours",
                        "home_rest_days", "away_rest_days", "series_game_num",
                        "home_sp_is_lefty", "away_sp_is_lefty",
                        "home_bp_ip_last3", "away_bp_ip_last3",
                        "ump_avg_runs", "bpf", "ppf",
                    )]

    target_col = "home_win"
    id_cols = ["date", "home_team", "away_team"]

    available = [c for c in feature_cols if c in master.columns]

    # Keep id columns for backtest merge, but don't train on them
    ml_df = master[available + [target_col] + id_cols].copy()

    # Drop rows with too many NaN features
    thresh = len(available) * 0.4  # need at least 40% of features
    ml_df = ml_df.dropna(thresh=int(thresh) + len(id_cols) + 1)

    logger.info(f"ML-ready dataset: {len(ml_df)} rows, {len(available)} features")
    logger.info(f"Feature groups: {len([c for c in available if c.startswith('diff_')])} diffs, "
                f"{len([c for c in available if c.startswith('ratio_')])} ratios, "
                f"{len([c for c in available if not c.startswith(('diff_', 'ratio_'))])} contextual")
    return ml_df


# ── Main Pipeline ────────────────────────────────────────────────────────

def main():
    # Step 1: Load all raw data
    games = load_games()
    if games.empty:
        logger.error("No games data. Run scraper/scrape_games.py first.")
        return

    pitchers = load_pitchers()
    batters = load_batters()

    # Step 2: Rolling team performance
    logger.info("Computing rolling team stats...")
    rolling_stats = compute_rolling_team_stats(games)

    # Step 3: Merge home/away rolling stats
    logger.info("Merging team rolling stats...")
    for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
        stats = rolling_stats.copy()
        rename = {c: f"{prefix}_{c}" for c in stats.columns if c not in ("date", "team")}
        stats = stats.rename(columns=rename).rename(columns={"team": team_col})
        games = games.merge(stats, on=["date", team_col], how="left")

    # Step 4: Pitcher features (season-level)
    logger.info("Building pitcher features...")
    pitcher_feats = build_pitcher_features(pitchers)

    # Step 5: Batter aggregate features (team-season level)
    logger.info("Building batter aggregate features...")
    batter_aggs = build_team_batter_aggregates(batters)
    if not batter_aggs.empty:
        for prefix, team_col in [("home", "home_team"), ("away", "away_team")]:
            ba = batter_aggs.copy()
            rename = {c: f"{prefix}_{c}" for c in ba.columns if c not in ("team", "year")}
            ba = ba.rename(columns=rename).rename(columns={"team": team_col})
            ba["year"] = pd.to_numeric(ba["year"], errors="coerce")
            games["_year"] = games["date"].dt.year
            games = games.merge(ba, left_on=[team_col, "_year"], right_on=[team_col, "year"],
                                how="left", suffixes=("", f"_{prefix}_bat"))
            games = games.drop(columns=["year", "_year"], errors="ignore")

    # Step 6: Travel / distance features
    logger.info("Building travel features...")
    games = build_travel_features(games)

    # Step 7: Starting pitcher features (game-level)
    logger.info("Building starting pitcher features...")
    games = build_sp_features(games, pitcher_feats)

    # Step 8: Bullpen workload features
    logger.info("Building bullpen features...")
    games = build_bullpen_features(games)

    # Step 9: Umpire features
    logger.info("Building umpire features...")
    games = build_umpire_features(games)

    # Step 10: Park factor features
    logger.info("Building park factor features...")
    games = build_park_factor_features(games)

    # Step 11: Rest days and series position
    logger.info("Building rest/series features...")
    games = build_rest_and_series_features(games)

    # Step 12: Time / schedule features
    logger.info("Building time features...")
    games = build_time_features(games)

    # Step 13: Build diff/ratio matchup features
    logger.info("Building matchup diff/ratio features...")
    games = build_matchup_diffs(games)

    # Step 12: Save master
    master_path = DATA_PROCESSED / "master.csv"
    games.to_csv(master_path, index=False)
    logger.info(f"Saved master.csv: {games.shape}")

    # Step 13: Create ML-ready dataset
    ml_ready = create_ml_ready(games)
    ml_path = DATA_PROCESSED / "ml_ready.csv"
    ml_ready.to_csv(ml_path, index=False)
    logger.info(f"Saved ml_ready.csv: {ml_ready.shape}")


if __name__ == "__main__":
    main()
