"""Prediction logic: load models and compute game win probabilities."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catboost import CatBoostClassifier
from xgboost import XGBClassifier

from utils.helpers import MODELS_SAVED, load_json, get_logger

logger = get_logger("predictor")

_models_cache = None
_manifest_cache = None


def load_models():
    """Load all ensemble models (cached)."""
    global _models_cache, _manifest_cache

    if _models_cache is not None:
        return _models_cache, _manifest_cache

    manifest = load_json(MODELS_SAVED / "model_manifest.json")
    models = []

    for i in range(1, manifest["n_catboost"] + 1):
        path = MODELS_SAVED / f"model_{i}.cbm"
        if path.exists():
            m = CatBoostClassifier()
            m.load_model(str(path))
            models.append(("catboost", m))

    for i in range(1, manifest["n_xgboost"] + 1):
        path = MODELS_SAVED / f"xgb_model_{i}.json"
        if path.exists():
            m = XGBClassifier()
            m.load_model(str(path))
            models.append(("xgboost", m))

    logger.info(f"Loaded {len(models)} models")
    _models_cache = models
    _manifest_cache = manifest
    return models, manifest


def predict_game(features: pd.DataFrame) -> dict:
    """Predict win probability for a single game.

    Args:
        features: DataFrame with one row containing the matchup features.

    Returns:
        Dict with home_win_prob, away_win_prob.
    """
    models, manifest = load_models()
    feature_cols = manifest["features"]

    # Ensure all expected features exist
    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0

    X = features[feature_cols].fillna(0)

    probs = []
    for model_type, model in models:
        if model_type == "xgboost":
            p = model.predict_proba(X)[:, 1]
        else:
            p = model.predict(X, prediction_type="Probability")[:, 1]
        probs.append(p)

    home_win_prob = float(np.mean(probs))
    return {
        "home_win_prob": round(home_win_prob, 4),
        "away_win_prob": round(1 - home_win_prob, 4),
    }


def predict_batch(features: pd.DataFrame) -> pd.DataFrame:
    """Predict win probabilities for multiple games."""
    models, manifest = load_models()
    feature_cols = manifest["features"]

    for col in feature_cols:
        if col not in features.columns:
            features[col] = 0

    X = features[feature_cols].fillna(0)

    probs = []
    for model_type, model in models:
        if model_type == "xgboost":
            p = model.predict_proba(X)[:, 1]
        else:
            p = model.predict(X, prediction_type="Probability")[:, 1]
        probs.append(p)

    features = features.copy()
    features["home_win_prob"] = np.mean(probs, axis=0)
    features["away_win_prob"] = 1 - features["home_win_prob"]
    return features
