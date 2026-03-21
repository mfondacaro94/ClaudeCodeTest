# MLBStats Betting Model — Project Context

## What This Project Is
An MLB game outcome prediction model that predicts home team win probability for every game. Uses a 7-model ensemble (5 CatBoost + 2 XGBoost) trained on 2015-2025 data. The goal is to find profitable betting edges against real Vegas moneylines.

## Current Model Performance (as of 2026-03-21, V3-audited)
- **Test set**: 2,487 games (Sept 2024 - Sept 2025), 1,631 with real odds from 3+ books
- **Model accuracy**: 56.2% | **Vegas accuracy**: 57.9% | **Gap**: 1.7%
- **ROI vs real odds (best available line, capped -450 to +400)**:
  - All value picks: 728 bets, 51.5% win, +$125 (+0.2% ROI) — basically breakeven
  - Edge > 5%: 318 bets, 50.9% win, +$1,638 (+5.2% ROI) — genuine edge
  - Edge > 10%: 105 bets, 55.2% win, +$2,228 (+21.2% ROI) — strong high-conviction picks
- **Baseline**: Always betting Vegas favorites loses -2.2% ROI
- **Features**: 77 (rolling stats, prior-year batter aggregates, SP pre-game ERA, SP handedness, bullpen workload, umpire tendency, park factors, travel, rest days, season progression)
- **Training split**: 80/10/10 chronological (no future leakage)
- **Audited**: Data leakage fixed (batter stats shifted to prior year), odds outliers capped, all verified by independent agents

## Data Pipeline
```
SCRAPING (data/raw/)
├── games.csv              — 24,697 games with WP/LP/SV, attendance, day/night
├── batters.csv            — 11,317 batter-season rows (3,323 players)
├── pitchers.csv           — 8,951 pitcher-season rows (2,591 players)
├── pitcher_gamelogs.csv   — SCRAPING NOW via MLB Stats API (all pitcher appearances per game)
├── injuries.csv           — 18,519 transactions (2015-2025)
├── umpires.csv            — 25,191 games with HP umpire (from Retrosheet, already complete)
├── park_factors.csv       — 330 rows, BPF/PPF per team per year (already complete)
├── teams_batting.csv      — 330 team-season rows
├── teams_pitching.csv     — 330 team-season rows
└── weather.csv            — 4,721 rows (NOT USED - zero feature importance)

ODDS DATA (data/odds/)
├── mlb_odds_dataset.json  — 76MB raw from github.com/ArnavSaraogi/mlb-odds-scraper
└── historical_odds.csv    — 11,657 games with moneyline odds from 6 sportsbooks (2021-2025)

FEATURE ENGINEERING (models/feature_engineering.py)
└── data/processed/ml_ready.csv — 24,697 rows, 62 features (will grow after new features added)

MODELS (models/saved/)
├── model_1-5.cbm          — CatBoost models (seeds 42-46)
├── xgb_model_1-2.json     — XGBoost models (seeds 42-43)
└── model_manifest.json

DASHBOARD (dashboard/)
├── app.py                 — Main Streamlit app (launch: streamlit run dashboard/app.py)
├── backtest.py            — Full backtest with Kelly/flat sizing, P&L curves, real odds
├── predictor.py           — Prediction logic
├── upcoming.py            — Upcoming games
└── visualizations.py      — Plotly charts
```

## Key Design Decisions
- **Data source priority**: Use MLB Stats API (statsapi.mlb.com) over baseball-reference when possible. It's free, has no rate limits, and returns structured JSON. BR should only be used for data not available via the API.
- **NEVER run multiple scrapers simultaneously** against the same site. We got IP-banned from BR by running 3 scrapers at once. Always run sequentially.
- **Progress tracking**: All scrapers use .ndjson progress files for incremental resume. Safe to interrupt.
- **Chunked saves**: Scrapers save to CSV every 25-50 items to prevent data loss on interruption.
- **Team name normalization**: All teams use standard 3-letter abbreviations (OAK not ATH, WSN not WSH, etc.). See `utils/data_cleaning.py:normalize_team_name()`.
- **Removed features**: Weather (all 9 features = zero importance), IL WAR, Coors flag, is_april, is_sept_plus. These added noise, not signal.
- **Top predictors**: Team batting WAR (#1 by far), rolling 40-game run differential, season-to-date runs allowed/game, travel distance.

## Feature Improvement Roadmap (7 items)
Status as of 2026-03-21:

1. **Game-level starting pitcher matchup** — DONE
   - Scraper: `scraper/scrape_mlb_api_pitchers.py` (MLB Stats API)
   - 238,217 pitcher appearances scraped, 93.2% SP match rate
   - Features: `home/away_sp_era_cume`, `home/away_sp_is_lefty` + diffs/ratios
   - IMPORTANT: Only use pre-game stats (era_cume, handedness). Game-day stats (IP, ER, SO) are leakage!

2. **Bullpen usage/availability** — DONE
   - Same pitcher_gamelogs.csv, `build_bullpen_features()` computes rolling 3-day reliever IP
   - Features: `home/away_bp_ip_last3` (95.6% match rate)

3. **Platoon splits (L/R matchups)** — DONE
   - `throws` column from MLB API, `home/away_sp_is_lefty` binary features
   - Feature importance: 0.6 (moderate signal)

4. **Rest days / series position** — DONE
   - `home_rest_days`, `away_rest_days`, `series_game_num`

5. **Umpire tendencies** — DONE
   - `ump_avg_runs`: expanding mean of total runs in umpire's historical games (shifted to prevent leakage)
   - 43.9% match rate (Retrosheet gaps)

6. **Park factors** — DONE
   - `bpf`, `ppf`: batter/pitcher park factors per team per year (88% match rate)

7. **Recent lineup changes** — DEFERRED
   - Daily starting lineups are hard to get historically
   - Will add for live predictions later

## Next Steps
1. Hyperparameter optimization (Optuna) to squeeze more from the 73 features
2. Option B backtest: layer in real odds for full Kelly criterion P&L simulation
3. Live prediction pipeline for upcoming games
4. Consider additional data: pitch-level Statcast data, weather (revisit with park-adjusted approach)

## Important Files
- `models/feature_engineering.py` — Full feature pipeline (raw → master.csv → ml_ready.csv)
- `models/train.py` — 7-model ensemble training with chronological split
- `dashboard/app.py` — Streamlit dashboard (backtest, odds comparison, model performance)
- `dashboard/backtest.py` — Full backtest with Kelly sizing, P&L curves, real Vegas odds
- `scraper/scrape_mlb_api_pitchers.py` — MLB Stats API scraper (replaces BR game log scraper)
- `utils/helpers.py` — RateLimitedSession, progress save/load, paths
- `utils/data_cleaning.py` — parse_br_table(), normalize_team_name(), safe_float()
- `utils/odds_math.py` — American/decimal odds conversion, Kelly criterion, EV calculation
- `utils/stadium_data.py` — Travel distance, timezone, dome/retractable lookups

## Git Workflow
Always commit and push after modifications. User prefers seeing changes reflected on GitHub.

## Tech Stack
- Python 3.9, CatBoost, XGBoost, pandas, Streamlit, Plotly, BeautifulSoup
- Primary data source: MLB Stats API (statsapi.mlb.com) — free, no key, structured JSON
- Secondary data source: baseball-reference.com (HTML scraping, use sparingly to avoid bans)
- Umpire data: Retrosheet (retrosheet.org) — free CSV downloads
- Odds data: github.com/ArnavSaraogi/mlb-odds-scraper (free historical dataset, 2021-2025)
