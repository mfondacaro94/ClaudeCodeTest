# TennisStats Betting Model — Project Context

## What This Project Is
A tennis match outcome prediction model that predicts match winner probability. Will use a multi-model ensemble (CatBoost + XGBoost) trained on historical ATP/WTA match data. The goal is to find profitable betting edges against real Vegas moneylines.

## Status: PROJECT SETUP — NOT STARTED
This project was scaffolded on 2026-03-21. No scraping, feature engineering, or modeling has been done yet.

## Architecture (following UFC/MLB proven pattern)
```
TennisStats/
├── CLAUDE.md                  — This file (project context for Claude)
├── setup_guide_reference.md   — UFC setup guide for reference architecture
├── requirements.txt           — Python dependencies (to be created)
│
├── scraper/                   — Data collection
│   ├── (scrapers TBD)         — Player stats, match results, rankings, odds
│
├── utils/                     — Shared utilities
│   ├── helpers.py             — Logging, serialization, rate limiting
│   ├── data_cleaning.py       — Name normalization, stat parsing
│   └── odds_math.py           — American/decimal/probability conversions, Kelly
│
├── models/                    — ML pipeline
│   ├── feature_engineering.py — Raw data → master.csv → ml_ready.csv
│   ├── train.py               — Train ensemble (5 CatBoost + 2 XGBoost)
│   ├── evaluate.py            — Evaluate models, save metrics
│   └── saved/                 — Trained model artifacts
│
├── dashboard/                 — Streamlit web app
│   ├── app.py                 — Main app with sidebar navigation
│   ├── backtest.py            — Kelly criterion backtesting with real odds
│   └── (other pages TBD)
│
├── scripts/                   — Automation
│   └── (pipeline runners TBD)
│
├── data/
│   ├── raw/                   — Scraped data
│   ├── processed/             — Feature-engineered data
│   └── odds/                  — Historical odds data
│
├── tests/                     — Unit tests
└── logs/                      — Pipeline logs
```

## Data Sources to Research
- **Match results**: ATP/WTA official data, tennis-data.co.uk (free CSVs with odds!), Jeff Sackmann's GitHub (comprehensive free tennis data)
- **Player stats**: Serve %, break points, aces, double faults, surface-specific records
- **Rankings**: ATP/WTA rankings history (ELO ratings available from various sources)
- **Odds**: tennis-data.co.uk includes historical closing odds from multiple bookmakers
- **Live odds**: The Odds API (free tier, same as MLB project)

## Key Differences from MLB
- **Individual sport**: Player form, fitness, fatigue matter more than team composition
- **Surface matters enormously**: Clay (Roland Garros), grass (Wimbledon), hard court — players have wildly different win rates by surface
- **Tournament structure**: Knockout format, seedings, draw analysis
- **Fatigue/scheduling**: Players who went 5 sets yesterday are at a disadvantage
- **Head-to-head**: Some players consistently beat others regardless of ranking
- **Ranking momentum**: Players on winning streaks vs declining form

## Potential Feature Groups
1. Player ranking (current + trend)
2. Surface-specific win rate
3. Recent form (last 10/20 matches)
4. Head-to-head record
5. Fatigue (days since last match, sets played recently)
6. Tournament round / seeding
7. Age / experience
8. Serve stats (ace %, double fault %, 1st serve %)
9. Break point conversion / saving
10. Indoor vs outdoor

## Lessons Learned from MLB Project
- **Use official APIs first** (MLB Stats API was way better than scraping BR)
- **NEVER run multiple scrapers simultaneously** against the same site
- **Save progress every 25-50 items** to avoid data loss
- **Check for data leakage** — season-level stats merged into games caused look-ahead bias in MLB. Always use lagged/shifted data.
- **Audit with agents** before trusting backtest results
- **Real odds matter** — flat -110 assumptions are meaningless. Need real historical closing lines.
- **Cap odds outliers** — rogue sportsbook feeds can have values like +60,000
- **Filter to games with 3+ books** for honest backtesting
- **High-conviction picks** (10%+ edge) are where the real money is

## Next Steps (for first chat session)
1. Research and choose data sources (tennis-data.co.uk is likely best starting point — free CSVs with match results AND odds included)
2. Download historical data
3. Explore data structure, clean, normalize
4. Feature engineering
5. Train initial model
6. Backtest against real odds
7. Build dashboard

## Git Workflow
Always commit and push after modifications. User prefers seeing changes reflected on GitHub.

## Tech Stack (same as MLB)
- Python 3.9, CatBoost, XGBoost, pandas, Streamlit, Plotly
- Odds math: Kelly criterion, American/decimal conversion (reuse from MLB)
