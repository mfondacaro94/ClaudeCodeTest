# UFC Fight Predictor — Complete Setup & Replication Guide

A machine learning system that predicts UFC fight outcomes, compares predictions against Vegas odds and Polymarket, backtests with Kelly criterion bet sizing, and serves everything through an interactive Streamlit dashboard.

**Model stack:** 5x CatBoost + 2x XGBoost ensemble (78.5% test accuracy)
**Method model:** 3x CatBoost MultiClass (Decision 65%, KO/TKO 50%, Submission 13%)
**Data source:** ufcstats.com (fighters + fights), TheOddsAPI (live odds), Kaggle (historical odds), Polymarket (prediction markets)

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Project Structure](#2-project-structure)
3. [Installation](#3-installation)
4. [Step 1: Scrape Fighter Data](#step-1-scrape-fighter-data)
5. [Step 2: Scrape Fight Data](#step-2-scrape-fight-data)
6. [Step 3: Load Historical Odds](#step-3-load-historical-odds)
7. [Step 4: Feature Engineering](#step-4-feature-engineering)
8. [Step 5: Hyperparameter Optimization (Optional)](#step-5-hyperparameter-optimization-optional)
9. [Step 6: Train Win/Loss Model](#step-6-train-winloss-model)
10. [Step 7: Train Method-of-Victory Model](#step-7-train-method-of-victory-model)
11. [Step 8: Evaluate Models](#step-8-evaluate-models)
12. [Step 9: Fetch Polymarket Data (Optional)](#step-9-fetch-polymarket-data-optional)
13. [Step 10: Launch Dashboard](#step-10-launch-dashboard)
14. [Step 11: Set Up Weekly Auto-Updates](#step-11-set-up-weekly-auto-updates)
15. [Dashboard Pages](#dashboard-pages)
16. [API Keys](#api-keys)
17. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

- **Python 3.10+** (tested on 3.13)
- **macOS or Linux** (Windows works but launchd scheduling is macOS-only)
- **~2 GB disk space** for data + models
- **Internet connection** for scraping and live odds

---

## 2. Project Structure

```
ufc_predictor/
├── .env                             # API keys (ODDS_API_KEY)
├── requirements.txt                 # Python dependencies
│
├── scraper/                         # Data collection
│   ├── scrape_fighters.py           # Fighter profiles from ufcstats.com
│   ├── scrape_fights.py             # Fight results + round-by-round stats
│   └── scrape_odds.py               # Live odds from TheOddsAPI
│
├── utils/                           # Shared utilities
│   ├── helpers.py                   # Logging, serialization
│   ├── data_cleaning.py             # Parse heights, weights, stats
│   ├── balancing.py                 # Dataset balancing via fight symmetry
│   ├── odds_math.py                 # American/decimal/probability conversions
│   ├── load_historical_odds.py      # Normalize historical odds from Kaggle
│   └── fetch_polymarket.py          # Fetch resolved Polymarket UFC markets
│
├── models/                          # ML pipeline
│   ├── feature_engineering.py       # Raw data -> master.csv -> ml_ready.csv
│   ├── hyperparameter_search.py     # Optuna HPO for CatBoost + XGBoost
│   ├── train.py                     # Train win/loss ensemble (5 CB + 2 XGB)
│   ├── train_method.py              # Train method-of-victory model (3 CB)
│   ├── evaluate.py                  # Evaluate models, save metrics
│   ├── feature_selection.py         # SHAP feature importance
│   └── saved/                       # Trained model artifacts
│       ├── model_1.cbm ... model_5.cbm
│       ├── xgb_model_1.json, xgb_model_2.json
│       ├── method_model_42.cbm ... method_model_44.cbm
│       ├── model_manifest.json
│       ├── method_manifest.json
│       ├── evaluation_metrics.json
│       └── feature_importance.csv
│
├── dashboard/                       # Streamlit web app
│   ├── app.py                       # Main app (all pages)
│   ├── predictor.py                 # Prediction logic
│   ├── upcoming.py                  # Upcoming fights + live odds
│   ├── backtest.py                  # Kelly criterion backtesting
│   ├── method_props.py              # Method-of-victory props
│   ├── polymarket.py                # Polymarket P&L simulation
│   ├── odds_comparison.py           # Value bet detection
│   └── visualizations.py            # Plotly charts
│
├── scripts/                         # Automation
│   ├── weekly_update.py             # Full pipeline orchestrator
│   └── setup_schedule.sh            # macOS launchd installer
│
├── data/
│   ├── raw/                         # Scraped data
│   │   ├── players.csv              # Fighter profiles
│   │   ├── competitions.csv         # Fight results
│   │   ├── historical_odds.csv      # Matched Vegas odds
│   │   ├── UFC_betting_odds.csv     # Raw Kaggle odds dataset
│   │   ├── polymarket_ufc.csv       # Polymarket markets
│   │   ├── fighters_progress.ndjson # Scraper checkpoint
│   │   └── fights_progress.ndjson   # Scraper checkpoint
│   ├── processed/
│   │   ├── master.csv               # Full feature dataframe
│   │   └── ml_ready.csv             # ML-ready (diff/ratio features + target)
│   └── odds/
│       └── upcoming_cache.json      # Cached live odds
│
├── tests/                           # Unit tests
│   ├── test_feature_engineering.py
│   ├── test_odds_math.py
│   └── test_predictions.py
│
└── logs/
    └── weekly_update.log            # Pipeline logs
```

---

## 3. Installation

```bash
# Clone or copy the project
cd /path/to/your/directory
# (copy the ufc_predictor folder here)

cd ufc_predictor

# Create virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Verify installation:**
```bash
python3 -c "import catboost, xgboost, streamlit, plotly, shap; print('All dependencies OK')"
```

---

## Step 1: Scrape Fighter Data

Scrapes all UFC fighter profiles from ufcstats.com — career stats, physical attributes, win/loss records.

```bash
python scraper/scrape_fighters.py
```

**What it does:**
- Crawls ufcstats.com alphabetical fighter listings (A-Z)
- Extracts: name, height, weight, reach, stance, DOB, wins, losses, draws, striking stats, takedown stats, submission stats
- Saves progress every 50 fighters to `data/raw/fighters_progress.ndjson` (resumable if interrupted)
- Final output: `data/raw/players.csv`

**Runtime:** ~30-45 minutes (4,400+ fighters, 0.3s delay between requests)
**Output:** `data/raw/players.csv` (~680 KB, ~4,400 rows)

> **Resumable:** If interrupted (Ctrl+C), run again — it skips already-scraped fighters.

---

## Step 2: Scrape Fight Data

Scrapes all completed UFC fight results with round-by-round statistics.

```bash
python scraper/scrape_fights.py
```

**What it does:**
- Crawls all completed event pages from ufcstats.com
- For each fight: extracts fighters, results, method, round, time, detailed round-by-round stats (knockdowns, significant strikes by location/position, takedowns, submissions, control time)
- Saves progress every 100 fights to `data/raw/fights_progress.ndjson`
- Final output: `data/raw/competitions.csv`

**Runtime:** ~15-20 minutes (8,500+ fights)
**Output:** `data/raw/competitions.csv` (~5 MB, ~8,500 rows)

> **Resumable:** Skips already-scraped fight URLs on re-run.

---

## Step 3: Load Historical Odds

Download the historical UFC odds dataset from Kaggle, then normalize and match it to your scraped fights.

### 3a. Download the odds dataset

Go to Kaggle and download the **jerzyszocik** UFC odds dataset. Place the CSV at:
```
data/raw/UFC_betting_odds.csv
```

The dataset should have columns: `fighter_1`, `fighter_2`, `odds_1`, `odds_2`, `event_date`

Alternative supported formats:
- **mdabbert** (Kaggle): `R_fighter`, `B_fighter`, `R_odds`, `B_odds`, `date`
- **jansen88** (GitHub): `favourite`, `underdog`, `favourite_odds`, `underdog_odds`

### 3b. Run the odds loader

```bash
python utils/load_historical_odds.py --input data/raw/UFC_betting_odds.csv --format auto
```

**What it does:**
- Auto-detects CSV format
- Normalizes all odds to American format
- Fuzzy-matches fighter names to `competitions.csv` fights using sorted name pairs
- Handles date proximity (3-day window for matching)
- Computes implied probabilities and vigorish

**Output:** `data/raw/historical_odds.csv` (~2 MB)
**Expected match rate:** ~80% (6,845 of 8,572 fights)

---

## Step 4: Feature Engineering

Transforms raw scraped data into ML-ready features.

```bash
python models/feature_engineering.py
```

**What it does:**
1. Loads `players.csv` and `competitions.csv`
2. Merges fighter career stats into fight records (by URL, fallback to name matching)
3. Computes derived features:
   - **Age** from DOB
   - **Days since last fight** per fighter
4. Creates matchup features:
   - **diff_{stat}** = player1_stat - player2_stat
   - **ratio_{stat}** = player1_stat / player2_stat
5. Filters: removes DQ, No Contest, draws; keeps fights from 2014+
6. Creates binary target: `player1_win` (1 = player1 won, 0 = lost)

**Output:**
- `data/processed/master.csv` — Full feature dataframe (~4,600 rows x 396 cols)
- `data/processed/ml_ready.csv` — ML features only (~4,600 rows x 190 cols: 188 features + target + date)

---

## Step 5: Hyperparameter Optimization (Optional)

Uses Optuna to find optimal hyperparameters. Skip this if you want to use defaults.

```bash
python models/hyperparameter_search.py
```

**What it does:**
- Runs 100 Optuna trials for CatBoost and XGBoost
- Uses TimeSeriesSplit cross-validation (chronological, no data leakage)
- Saves best params to `models/saved/best_params_catboost.json` and `best_params_xgboost.json`

**Runtime:** ~30-60 minutes
**Output:** `models/saved/best_params_catboost.json`, `models/saved/best_params_xgboost.json`

> If skipped, `train.py` uses sensible defaults.

---

## Step 6: Train Win/Loss Model

Trains the main ensemble model for predicting fight winners.

```bash
python models/train.py
```

**What it does:**
- Loads `ml_ready.csv`
- 80/10/10 chronological train/validation/test split (no future data leakage)
- Applies linear sample weights (recent fights weighted more)
- Trains **5 CatBoost models** (seeds 42-46) and **2 XGBoost models** (seeds 42-43)
- Ensemble prediction = average of all 7 models' probabilities
- Saves models and manifest

**Runtime:** ~10 seconds
**Output:**
- `models/saved/model_1.cbm` through `model_5.cbm` (~240 KB each)
- `models/saved/xgb_model_1.json`, `xgb_model_2.json` (~1.2 MB each)
- `models/saved/model_manifest.json` (feature list + metrics)

**Expected test accuracy:** ~78.5%

---

## Step 7: Train Method-of-Victory Model

Trains a separate model predicting how a fight ends: Decision, KO/TKO, or Submission.

```bash
python models/train_method.py
```

**What it does:**
- Parses `method` column from `master.csv` into 3 classes: Decision (0), KO/TKO (1), Submission (2)
- Excludes DQ, No Contest, Overturned, Draw fights
- Trains **3 CatBoost MultiClass models** (seeds 42-44)
- Same 188 diff/ratio features as win/loss model

**Runtime:** ~10 seconds
**Output:**
- `models/saved/method_model_42.cbm`, `43.cbm`, `44.cbm` (~2 MB each)
- `models/saved/method_manifest.json`

**Expected accuracy:** 50% overall (vs 33% random baseline)
- Decision: 65.5%
- KO/TKO: 49.7%
- Submission: 13.3%

---

## Step 8: Evaluate Models

Generates evaluation metrics, confusion matrices, and feature importance.

```bash
python models/evaluate.py
```

**Output:**
- `models/saved/evaluation_metrics.json` — accuracy, log_loss, ROC-AUC, Brier score, test period
- `models/saved/feature_importance.csv` — averaged CatBoost feature importances
- `models/saved/confusion_matrix.png` — win/loss confusion matrix
- `models/saved/method_confusion_matrix.png` — method-of-victory confusion matrix
- `models/saved/calibration_curve.png` — probability calibration plot
- `models/saved/roi_curve.png` — ROI curve from backtest

---

## Step 9: Fetch Polymarket Data (Optional)

Fetches resolved UFC prediction market data from Polymarket for P&L comparison.

```bash
python utils/fetch_polymarket.py
```

**What it does:**
- Uses the public Polymarket Gamma + CLOB APIs (no API key required)
- Fetches all resolved UFC fight markets
- Extracts pre-fight closing prices (implied probabilities)

**Runtime:** 5-15 minutes (~500+ resolved markets)
**Output:** `data/raw/polymarket_ufc.csv` (~320 KB)

---

## Step 10: Launch Dashboard

```bash
streamlit run dashboard/app.py
```

Opens at **http://localhost:8501**

> For a specific port: `streamlit run dashboard/app.py --server.port 8502`

---

## Step 11: Set Up Weekly Auto-Updates

The weekly update pipeline automatically scrapes new fights, updates fighter stats, rebuilds features, and retrains models.

### Manual run:

```bash
# Full pipeline (scrape + features + retrain)
python scripts/weekly_update.py

# Data only, skip retraining
python scripts/weekly_update.py --no-retrain

# Preview without executing
python scripts/weekly_update.py --dry-run

# Force re-scrape ALL fighters (not just recently active)
python scripts/weekly_update.py --force-all-fighters
```

### Automated schedule (macOS):

```bash
chmod +x scripts/setup_schedule.sh
./scripts/setup_schedule.sh
```

This installs a **launchd job** that runs every **Monday at 6:00 AM**.

**Manage the schedule:**
```bash
# Check if running
launchctl list | grep ufc

# Run immediately
launchctl start com.ufc-predictor.weekly-update

# View logs
tail -f logs/weekly_update.log

# Uninstall
launchctl unload ~/Library/LaunchAgents/com.ufc-predictor.weekly-update.plist
rm ~/Library/LaunchAgents/com.ufc-predictor.weekly-update.plist
```

### How the smart update works:

1. **Scrape new fights** — incremental, only fetches fights with new URLs
2. **Identify active fighters** — extracts fighter URLs from newly-scraped fights
3. **Re-scrape active fighters only** — removes them from progress so their updated career stats get refreshed (doesn't re-scrape all 4,400+ fighters)
4. **Rebuild features** — regenerates `master.csv` and `ml_ready.csv`
5. **Update historical odds** — re-maps odds to any new fights
6. **Retrain models** — win/loss ensemble + method-of-victory
7. **Evaluate** — saves updated metrics and confusion matrices

---

## Dashboard Pages

### 1. Upcoming Fights (default landing page)
- Fetches live odds from TheOddsAPI for the next 30 days
- Groups fights by event card
- Shows: Vegas odds, model win probability, edge, bet recommendation
- Green-bordered cards = recommended bets (confidence > threshold + positive Kelly)
- Charts: EV per bet, edge vs odds scatter
- Bet table with Kelly and flat staking amounts

### 2. Fighter Matchup Predictor
- Select any two fighters from dropdown
- Shows win probability (pie/gauge), radar chart, tale of tape
- Model implied odds vs hypothetical lines

### 3. Vegas Odds Comparison
- Input real American odds for a matchup
- Calculates: edge, EV, Kelly optimal bet size
- Value bet verdict (positive EV, negative EV, marginal)

### 4. Fighter Comparison
- Side-by-side stat tables for any two fighters
- Radar charts and recent form

### 5. Backtest
- Simulates the model betting on the last N historical fights
- Kelly criterion compounding + flat staking dual track
- Uses real historical Vegas lines (80% coverage, fallback to -110)
- Metrics: Sharpe, Sortino, CAGR, Win %, Max Win/Loss
- Balance curve, P&L bars, edge scatter, confidence histogram

### 6. Method Props
- Predicts finish method: Decision, KO/TKO, Submission
- Donut chart, bar chart, finish probability gauge
- Model implied odds for prop bets

### 7. Polymarket P&L
- Simulates betting the model's picks on Polymarket vs Vegas
- Apples-to-apples comparison (same matched fights, fresh bankroll)
- Balance curves, edge comparison, market depth

### 8. Model Performance
- Test-set metrics: accuracy, log loss, ROC-AUC, Brier score
- Feature importance (top 30 features)
- Test period and sample size

---

## API Keys

### TheOddsAPI (required for live odds)
1. Sign up at https://the-odds-api.com (free tier: 500 requests/month)
2. Set your key:
   ```bash
   # Option A: environment variable
   export ODDS_API_KEY='your_key_here'

   # Option B: .env file in project root
   echo "ODDS_API_KEY=your_key_here" > .env
   ```

> The dashboard works without a key (except the Upcoming Fights page) — it uses cached data or simulated -110 odds.

---

## Troubleshooting

### "No evaluation metrics found"
```bash
python models/evaluate.py
```
This generates `models/saved/evaluation_metrics.json` required by the Model Performance page.

### "Fighter not found in master data"
The fighter hasn't been scraped yet. Re-run:
```bash
python scraper/scrape_fighters.py
python scraper/scrape_fights.py
python models/feature_engineering.py
```

### Scraper gets stuck or times out
Both scrapers are resumable. Just run them again — they skip already-scraped URLs and pick up where they left off.

### Backtest metrics look wrong (Sharpe > 5, CAGR > 1000%)
These were fixed. Sharpe is annualized using `sqrt(bets_per_year)` derived from actual date span. CAGR uses `years_elapsed = n_bets / bets_per_year`. If you see old values, restart the Streamlit app (it caches aggressively).

### Dashboard shows old data after update
Kill all Streamlit processes and restart on a new port:
```bash
pkill -9 -f streamlit
streamlit run dashboard/app.py --server.port 8502
```

### Weekly update takes too long
The fight scraper scans all ~700 event pages even when there's nothing new (~15 min). This is by design to catch any newly-posted events. Use `--no-retrain` to skip model training if you just want fresh data.

---

## Quick Start (TL;DR)

```bash
# 1. Install
pip install -r requirements.txt

# 2. Scrape everything (~45 min total)
python scraper/scrape_fighters.py
python scraper/scrape_fights.py

# 3. Build features
python models/feature_engineering.py

# 4. Train models (~20 sec)
python models/train.py
python models/train_method.py
python models/evaluate.py

# 5. Set API key for live odds
echo "ODDS_API_KEY=your_key" > .env

# 6. Launch
streamlit run dashboard/app.py

# 7. (Optional) Schedule weekly updates
./scripts/setup_schedule.sh
```

---

## Full Pipeline in One Command

If you already have the raw data and just want to rebuild everything:

```bash
python scripts/weekly_update.py
```

This runs all steps: scrape -> features -> odds -> train -> evaluate.
