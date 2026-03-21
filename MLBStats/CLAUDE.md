# MLBStats Betting Model — Project Context

## What This Project Is
An MLB game outcome prediction model that predicts home team win probability for every game. Uses a 7-model ensemble (5 CatBoost + 2 XGBoost) trained on 2015-2025 data. The goal is to find profitable betting edges against real Vegas moneylines.

## Current Model Performance (as of 2026-03-20)
- **Test set**: 2,470 games (Sept 2024 - Sept 2025)
- **Model accuracy**: 56.6% | **Vegas accuracy**: 58.1%
- **ROI vs real odds (best available line)**: +2.4% all picks, +6.2% at 5%+ edge, +10.2% at 10%+ edge
- **Baseline**: Always betting Vegas favorites loses -2.1% ROI (vig)
- **Features**: 62 (rolling team stats, batter aggregates, travel, rest days, season progression)
- **Training split**: 80/10/10 chronological (no future leakage)

## Data Pipeline
```
SCRAPING (data/raw/)
├── games.csv          — 24,697 games with WP/LP/SV, attendance, day/night
├── batters.csv        — 11,317 batter-season rows (3,323 players)
├── pitchers.csv       — 8,951 pitcher-season rows (2,591 players)
├── injuries.csv       — 18,519 transactions (2015-2025)
├── teams_batting.csv  — 330 team-season rows
├── teams_pitching.csv — 330 team-season rows
├── weather.csv        — 4,721 rows (NOT USED - zero feature importance)
└── sp_gamelogs.csv    — TO BE SCRAPED (starting pitcher per-game stats)

ODDS DATA (data/odds/)
├── mlb_odds_dataset.json — 76MB raw from github.com/ArnavSaraogi/mlb-odds-scraper
└── historical_odds.csv   — 11,657 games with moneyline odds from 6 sportsbooks (2021-2025)

FEATURE ENGINEERING (models/feature_engineering.py)
└── data/processed/ml_ready.csv — 24,697 rows, 62 features

MODELS (models/saved/)
├── model_1-5.cbm      — CatBoost models (seeds 42-46)
├── xgb_model_1-2.json — XGBoost models (seeds 42-43)
└── model_manifest.json
```

## Key Design Decisions
- **Rate limiting**: 6-second delay between baseball-reference requests. Only run ONE scraper at a time (multiple scrapers hit 429 rate limits).
- **Progress tracking**: All scrapers use .ndjson progress files for incremental resume. Safe to interrupt.
- **Team name normalization**: All teams use standard 3-letter abbreviations (OAK not ATH, WSN not WSH, etc.). See `utils/data_cleaning.py:normalize_team_name()`.
- **Removed features**: Weather (all 9 features = zero importance), IL WAR, Coors flag, is_april, is_sept_plus. These added noise, not signal.
- **Top predictors**: Team batting WAR (#1 by far), rolling 40-game run differential, season-to-date runs allowed/game, travel distance.

## Feature Improvement Roadmap (7 items)
Status as of 2026-03-20:

1. **Game-level starting pitcher matchup** — IN PROGRESS
   - Scraper built: `scraper/scrape_sp_gamelogs.py`
   - Feature builder built: `build_sp_features()` in feature_engineering.py
   - Needs: Run scraper (3,815 pitcher-years, ~6.4 hrs)
   - Why: SP stats are currently at team-season level, not game level. Who's pitching matters enormously.

2. **Bullpen usage/availability** — PLANNED
   - Modify SP scraper to pull ALL pitchers (not just starters) → reliever game logs
   - Track innings pitched in last 1-3 days per team's bullpen
   - Adds ~3,000 more pitcher-years to scrape

3. **Platoon splits (L/R matchups)** — PLANNED
   - Pitcher handedness: BR marks lefties with * on names (extractable from existing data)
   - Team batting splits vs LHP/RHP would add signal
   - No new scraping needed for basic handedness

4. **Rest days / series position** — DONE
   - `home_rest_days`, `away_rest_days`, `series_game_num` features built
   - Low importance so far (0.1-0.5) but may improve with more granular data

5. **Umpire tendencies** — PLANNED
   - Home plate umpire per game from Retrosheet
   - Some umps have measurably different strike zones → affects run scoring
   - ~330 pages to scrape (~33 min)

6. **Park factors** — PLANNED
   - Run-scoring environment per stadium per year
   - BR has park factors page (11 pages, ~1 min)

7. **Recent lineup changes** — DEFERRED
   - Daily starting lineups are hard to get historically
   - Will add for live predictions later

## Overnight Scrape Plan
Run sequentially to avoid rate limits:
1. Park factors (11 pages, ~1 min)
2. Umpire data from Retrosheet (330 pages, ~33 min)
3. All pitcher game logs — starters AND relievers (6,800 pages, ~11.3 hrs)

Total: ~12 hours. Covers features #1, #2, #3, #5, #6.

## Important Files
- `models/feature_engineering.py` — Full feature pipeline (raw → master.csv → ml_ready.csv)
- `models/train.py` — 7-model ensemble training with chronological split
- `scripts/run_all_scrapers.py` — Sequential scraper runner (avoids rate limits)
- `utils/helpers.py` — RateLimitedSession, progress save/load, paths
- `utils/data_cleaning.py` — parse_br_table(), normalize_team_name(), safe_float()
- `utils/stadium_data.py` — Travel distance, timezone, dome/retractable lookups

## Git Workflow
Always commit and push after modifications. User prefers seeing changes reflected on GitHub.

## Tech Stack
- Python 3.9, CatBoost, XGBoost, pandas, BeautifulSoup, lxml
- Data source: baseball-reference.com (HTML scraping with rate limiting)
- Odds source: github.com/ArnavSaraogi/mlb-odds-scraper (free historical dataset)
