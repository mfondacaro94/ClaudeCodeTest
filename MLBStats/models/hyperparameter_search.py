"""Optuna hyperparameter optimization for CatBoost and XGBoost."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import optuna
from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import log_loss

from utils.helpers import get_logger, DATA_PROCESSED, MODELS_SAVED, save_json

logger = get_logger("hpo")
N_TRIALS = 80
N_SPLITS = 5


def load_data():
    path = DATA_PROCESSED / "ml_ready.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("home_win", "date")]
    X = df[feature_cols].fillna(0)
    y = df["home_win"].astype(int)
    return X, y


def objective_catboost(trial, X, y):
    params = {
        "iterations": trial.suggest_int("iterations", 500, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.5, 1.0),
        "random_seed": 42,
        "eval_metric": "Logloss",
        "verbose": 0,
        "early_stopping_rounds": 80,
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(Pool(X_train, y_train), eval_set=Pool(X_val, y_val), use_best_model=True)
        prob = model.predict(X_val, prediction_type="Probability")[:, 1]
        scores.append(log_loss(y_val, prob))

    return np.mean(scores)


def objective_xgboost(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 500, 2500),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "random_state": 42,
        "eval_metric": "logloss",
        "early_stopping_rounds": 80,
        "verbosity": 0,
    }

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    scores = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        prob = model.predict_proba(X_val)[:, 1]
        scores.append(log_loss(y_val, prob))

    return np.mean(scores)


def main():
    X, y = load_data()
    logger.info(f"Data: {X.shape}")

    # CatBoost HPO
    logger.info("Starting CatBoost HPO...")
    study_cb = optuna.create_study(direction="minimize")
    study_cb.optimize(lambda trial: objective_catboost(trial, X, y), n_trials=N_TRIALS)
    save_json(MODELS_SAVED / "best_params_catboost.json", study_cb.best_params)
    logger.info(f"Best CatBoost params: {study_cb.best_params}")

    # XGBoost HPO
    logger.info("Starting XGBoost HPO...")
    study_xgb = optuna.create_study(direction="minimize")
    study_xgb.optimize(lambda trial: objective_xgboost(trial, X, y), n_trials=N_TRIALS)
    save_json(MODELS_SAVED / "best_params_xgboost.json", study_xgb.best_params)
    logger.info(f"Best XGBoost params: {study_xgb.best_params}")


if __name__ == "__main__":
    main()
