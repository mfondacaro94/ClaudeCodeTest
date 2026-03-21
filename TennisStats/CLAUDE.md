# TennisStats Betting Model — Project Context

## What This Project Is
ATP tennis match outcome prediction model using a 7-model ensemble (5 CatBoost + 2 XGBoost) trained on historical ATP match data from 2010-2025. Finds profitable betting edges against real bookmaker odds (Pinnacle, Bet365, etc.) using Kelly criterion bet sizing.

## Status: V1 MODEL COMPLETE
- **Test Accuracy: 73.3%** | ROC-AUC: 0.834 | Brier: 0.164
- Test period: 2024-05-30 to 2025-11-16 (3,683 matches)
- 40,766 total matches, 36,827 ML-ready after feature filtering
- 126 features (116 diff/ratio + 10 context)

## Top Features (by importance)
1. diff_bpFaced (break points faced difference)
2. diff_1stWon (first serve points won)
3. ratio_1stWon
4. diff_2ndWon (second serve points won)
5. ratio_2ndWon
6. ratio_bpFaced
7. diff_bpSaved
8. diff_svpt (service points)
9. ratio_svpt
10. ELO ratings (#13-14)

## Data Sources
- **tennis-data.co.uk**: Primary source — match results + odds from 5+ bookmakers (Pinnacle, Bet365, Ladbrokes, etc.), 2010-2025. 40,766 ATP matches. Odds in **decimal format**.
- **Jeff Sackmann's tennis_atp GitHub**: Secondary — match-level serve/return stats, player bios (DOB, height, hand, country), rankings history. Matched at 79.3% rate.
- **The Odds API**: For live upcoming match odds (not yet integrated).

## Architecture
```
TennisStats/
├── CLAUDE.md                              — This file
├── setup_guide_reference.md               — UFC setup guide reference
├── requirements.txt                       — Python dependencies
│
├── scraper/
│   ├── download_tennis_data_uk.py         — Download odds+results by year (resumable)
│   ├── download_sackmann.py               — Download stats+players+rankings (resumable)
│   └── merge_sources.py                   — Match & merge both sources
│
├── utils/
│   ├── helpers.py                         — Logging, NDJSON progress, RateLimitedSession
│   ├── data_cleaning.py                   — Name normalization, geo coords, score parsing
│   └── odds_math.py                       — Decimal/American conversion, Kelly criterion
│
├── models/
│   ├── feature_engineering.py             — ELO, rolling stats, H2H, fatigue, distance-to-home
│   ├── train.py                           — Train 5 CatBoost + 2 XGBoost ensemble
│   ├── evaluate.py                        — Metrics, feature importance, confusion matrix
│   └── saved/                             — Model artifacts + manifests
│
├── dashboard/
│   ├── app.py                             — Streamlit main app
│   ├── backtest.py                        — Kelly criterion backtest with real odds
│   └── model_performance.py               — Test metrics + feature importance charts
│
├── scripts/
│   └── run_pipeline.py                    — Full pipeline orchestrator
│
├── data/
│   ├── raw/                               — Downloaded CSVs (by year + combined)
│   ├── processed/                         — master.csv (433 cols) + ml_ready.csv (139 cols)
│   └── odds/                              — (reserved for live odds cache)
│
├── tests/                                 — Unit tests (TBD)
└── logs/                                  — Pipeline logs
```

## Feature Groups (all use shift(1) to prevent leakage)
1. **ELO ratings**: Overall + surface-specific, K=32/24/16 by experience
2. **Surface win rate**: Rolling win % on current surface (10/20 match windows)
3. **Recent form**: Win rate last 5/10/20 matches
4. **Head-to-head**: H2H wins overall + on current surface
5. **Fatigue**: Days since last match, sets/matches in last 7/14 days
6. **Tournament context**: Round number, best-of-3 vs 5, tournament level, seeding
7. **Serve stats**: Ace rate, DF rate, 1st serve %, 1st/2nd serve win % (rolling)
8. **Return stats**: Break point conversion %, return points won %
9. **Age/experience**: Age, height, handedness, career match count
10. **Indoor/outdoor**: Court environment indicator
11. **Distance to home**: Haversine km from player country to tournament city

## Key Design Decisions
- **P1/P2 randomly assigned** (50% flip) to balance dataset — P1 win rate ~0.51
- **All stats are diff_ and ratio_ pairs** (P1 minus P2, P1 divided by P2)
- **Decimal odds** from tennis-data.co.uk (not American) — use `decimal_to_implied_prob()`
- **Filter to 3+ bookmakers** for honest backtesting (23,590 matches qualify)
- **Odds capped at 50.0 decimal** to remove outliers
- **Chronological 80/10/10 split**, linear sample weights (0.5 to 1.0)

## Running the Pipeline
```bash
# Full pipeline from scratch
python3 scripts/run_pipeline.py

# Or step by step:
python3 scraper/download_tennis_data_uk.py
python3 scraper/download_sackmann.py
python3 scraper/merge_sources.py
python3 models/feature_engineering.py
python3 models/train.py
python3 models/evaluate.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Next Steps
- Run backtest analysis to measure ROI at various edge thresholds
- Add live odds integration via The Odds API
- Hyperparameter optimization with Optuna
- Add more dashboard pages (matchup predictor, odds comparison)
- Weekly auto-update pipeline

## Git Workflow
Always commit and push after modifications. User prefers seeing changes reflected on GitHub.

## Tech Stack
- Python 3.9, CatBoost, XGBoost, pandas, Streamlit, Plotly
- Odds math: Kelly criterion, decimal/American conversion
