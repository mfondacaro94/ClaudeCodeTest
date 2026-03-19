"""Train win/loss ensemble model: 5 CatBoost + 2 XGBoost.

Same architecture as the UFC predictor, adapted for MLB game outcomes.
Uses chronological train/validation/test split (no future data leakage).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score, brier_score_loss

from utils.helpers import get_logger, DATA_PROCESSED, MODELS_SAVED, save_json

logger = get_logger("train")

CB_SEEDS = [42, 43, 44, 45, 46]
XGB_SEEDS = [42, 43]

CB_DEFAULT_PARAMS = {
    "iterations": 1500,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3,
    "subsample": 0.8,
    "colsample_bylevel": 0.8,
    "eval_metric": "Logloss",
    "verbose": 100,
    "early_stopping_rounds": 100,
}

XGB_DEFAULT_PARAMS = {
    "n_estimators": 1500,
    "learning_rate": 0.05,
    "max_depth": 6,
    "reg_alpha": 1,
    "reg_lambda": 3,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "early_stopping_rounds": 100,
    "verbosity": 1,
}


def load_data():
    """Load ML-ready data and split chronologically."""
    path = DATA_PROCESSED / "ml_ready.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("home_win", "date")]
    X = df[feature_cols]
    y = df["home_win"].astype(int)
    dates = df["date"]

    # 80/10/10 chronological split
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    logger.info(f"Train period: {dates.iloc[0].date()} - {dates.iloc[train_end-1].date()}")
    logger.info(f"Val period:   {dates.iloc[train_end].date()} - {dates.iloc[val_end-1].date()}")
    logger.info(f"Test period:  {dates.iloc[val_end].date()} - {dates.iloc[-1].date()}")

    # Linear sample weights (recent games weighted more)
    weights = np.linspace(0.5, 1.0, len(X_train))

    return X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, weights, dates


def load_hpo_params(model_type: str) -> dict:
    """Load hyperparameters from Optuna search if available."""
    path = MODELS_SAVED / f"best_params_{model_type}.json"
    if path.exists():
        with open(path) as f:
            params = json.load(f)
        logger.info(f"Loaded HPO params for {model_type}")
        return params
    return {}


def train_catboost(X_train, y_train, X_val, y_val, weights, seed: int, idx: int):
    """Train a single CatBoost model."""
    params = {**CB_DEFAULT_PARAMS}
    hpo = load_hpo_params("catboost")
    params.update(hpo)
    params["random_seed"] = seed

    train_pool = Pool(X_train, y_train, weight=weights)
    val_pool = Pool(X_val, y_val)

    model = CatBoostClassifier(**params)
    model.fit(train_pool, eval_set=val_pool, use_best_model=True)

    save_path = MODELS_SAVED / f"model_{idx}.cbm"
    model.save_model(str(save_path))
    logger.info(f"Saved CatBoost model {idx} (seed={seed})")
    return model


def train_xgboost(X_train, y_train, X_val, y_val, weights, seed: int, idx: int):
    """Train a single XGBoost model."""
    params = {**XGB_DEFAULT_PARAMS}
    hpo = load_hpo_params("xgboost")
    params.update(hpo)
    params["random_state"] = seed

    model = XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight=weights,
        eval_set=[(X_val, y_val)],
        verbose=100,
    )

    save_path = MODELS_SAVED / f"xgb_model_{idx}.json"
    model.save_model(str(save_path))
    logger.info(f"Saved XGBoost model {idx} (seed={seed})")
    return model


def evaluate_ensemble(models: list, X_test, y_test) -> dict:
    """Evaluate the ensemble on the test set."""
    probs = []
    for model in models:
        if hasattr(model, "predict_proba"):
            p = model.predict_proba(X_test)[:, 1]
        else:
            p = model.predict(X_test, prediction_type="Probability")[:, 1]
        probs.append(p)

    ensemble_prob = np.mean(probs, axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    metrics = {
        "accuracy": round(accuracy_score(y_test, ensemble_pred), 4),
        "log_loss": round(log_loss(y_test, ensemble_prob), 4),
        "roc_auc": round(roc_auc_score(y_test, ensemble_prob), 4),
        "brier_score": round(brier_score_loss(y_test, ensemble_prob), 4),
        "n_test": len(y_test),
        "n_models": len(models),
        "home_win_rate_actual": round(y_test.mean(), 4),
        "home_win_rate_predicted": round(ensemble_prob.mean(), 4),
    }

    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Test Log Loss: {metrics['log_loss']:.4f}")
    logger.info(f"Test ROC-AUC:  {metrics['roc_auc']:.4f}")
    logger.info(f"Test Brier:    {metrics['brier_score']:.4f}")

    return metrics


def main():
    X_train, y_train, X_val, y_val, X_test, y_test, feature_cols, weights, dates = load_data()

    # Handle NaN (fill with 0 for tree models)
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)
    X_test = X_test.fillna(0)

    models = []

    # Train CatBoost models
    for i, seed in enumerate(CB_SEEDS, 1):
        model = train_catboost(X_train, y_train, X_val, y_val, weights, seed, i)
        models.append(model)

    # Train XGBoost models
    for i, seed in enumerate(XGB_SEEDS, 1):
        model = train_xgboost(X_train, y_train, X_val, y_val, weights, seed, i)
        models.append(model)

    # Evaluate ensemble
    metrics = evaluate_ensemble(models, X_test, y_test)

    # Save manifest
    manifest = {
        "features": feature_cols,
        "n_catboost": len(CB_SEEDS),
        "n_xgboost": len(XGB_SEEDS),
        "cb_seeds": CB_SEEDS,
        "xgb_seeds": XGB_SEEDS,
        "metrics": metrics,
        "train_size": len(X_train),
        "val_size": len(X_val),
        "test_size": len(X_test),
    }
    save_json(MODELS_SAVED / "model_manifest.json", manifest)
    logger.info("Training complete. Manifest saved.")


if __name__ == "__main__":
    main()
