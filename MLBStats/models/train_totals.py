"""Train over/under totals model (analogous to UFC method-of-victory).

Predicts whether a game's combined score goes over or under the posted total.
Uses 3 CatBoost models with different seeds.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss

from utils.helpers import get_logger, DATA_PROCESSED, DATA_RAW, MODELS_SAVED, save_json

logger = get_logger("train_totals")

SEEDS = [42, 43, 44]


def load_data():
    """Load master data with historical odds totals for over/under target."""
    master_path = DATA_PROCESSED / "master.csv"
    odds_path = DATA_RAW / "historical_odds.csv"

    master = pd.read_csv(master_path)
    master["date"] = pd.to_datetime(master["date"])

    # We need the posted total line to create the target
    # If historical odds include totals, merge them
    if odds_path.exists():
        odds = pd.read_csv(odds_path)
        if "total_line" in odds.columns:
            odds["date"] = pd.to_datetime(odds["date"])
            master = master.merge(
                odds[["date", "home_team", "away_team", "total_line"]],
                on=["date", "home_team", "away_team"],
                how="left"
            )

    # If no total line available, use league average (~9.0 runs) as proxy
    if "total_line" not in master.columns:
        master["total_line"] = 9.0
        logger.warning("No total lines found. Using 9.0 as default.")

    # Create target: did the game go over?
    master["total_runs"] = master["home_runs"] + master["away_runs"]
    master["went_over"] = (master["total_runs"] > master["total_line"]).astype(int)

    # Feature columns (same as win/loss model + total_line)
    feature_cols = [c for c in master.columns
                    if c.startswith(("diff_", "ratio_"))
                    or c in ("day_of_week", "month", "is_weekend", "total_line")]

    available = [c for c in feature_cols if c in master.columns]

    # Drop NaN-heavy rows
    subset = master[available + ["went_over", "date"]].dropna(
        thresh=int(len(available) * 0.5) + 2
    )

    X = subset[available].fillna(0)
    y = subset["went_over"].astype(int)
    dates = subset["date"]

    # Chronological split
    n = len(X)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    return (
        X.iloc[:train_end], y.iloc[:train_end],
        X.iloc[train_end:val_end], y.iloc[train_end:val_end],
        X.iloc[val_end:], y.iloc[val_end:],
        available
    )


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols = load_data()

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    weights = np.linspace(0.5, 1.0, len(X_train))
    models = []

    for i, seed in enumerate(SEEDS):
        logger.info(f"Training totals model {i+1} (seed={seed})")
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3,
            subsample=0.8,
            random_seed=seed,
            eval_metric="Logloss",
            verbose=100,
            early_stopping_rounds=80,
        )

        train_pool = Pool(X_train, y_train, weight=weights)
        val_pool = Pool(X_val, y_val)
        model.fit(train_pool, eval_set=val_pool, use_best_model=True)

        save_path = MODELS_SAVED / f"totals_model_{seed}.cbm"
        model.save_model(str(save_path))
        models.append(model)

    # Evaluate
    probs = np.mean([m.predict(X_test, prediction_type="Probability")[:, 1] for m in models], axis=0)
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, preds), 4),
        "log_loss": round(log_loss(y_test, probs), 4),
        "over_rate_actual": round(y_test.mean(), 4),
        "over_rate_predicted": round(probs.mean(), 4),
        "n_test": len(y_test),
    }

    logger.info(f"Totals Test Accuracy: {metrics['accuracy']:.4f}")

    manifest = {
        "features": feature_cols,
        "seeds": SEEDS,
        "metrics": metrics,
    }
    save_json(MODELS_SAVED / "totals_manifest.json", manifest)
    logger.info("Totals model training complete.")


if __name__ == "__main__":
    main()
