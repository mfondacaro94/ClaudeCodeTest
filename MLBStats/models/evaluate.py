"""Evaluate models and generate metrics, confusion matrices, and plots."""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, brier_score_loss,
    confusion_matrix, classification_report
)

from utils.helpers import get_logger, DATA_PROCESSED, MODELS_SAVED, save_json, load_json

logger = get_logger("evaluate")


def load_models():
    """Load all trained ensemble models."""
    models = []

    # CatBoost
    for i in range(1, 6):
        path = MODELS_SAVED / f"model_{i}.cbm"
        if path.exists():
            m = CatBoostClassifier()
            m.load_model(str(path))
            models.append(("catboost", m))

    # XGBoost
    for i in range(1, 3):
        path = MODELS_SAVED / f"xgb_model_{i}.json"
        if path.exists():
            m = XGBClassifier()
            m.load_model(str(path))
            models.append(("xgboost", m))

    logger.info(f"Loaded {len(models)} models")
    return models


def get_test_data():
    """Load ML-ready data and extract the test set."""
    path = DATA_PROCESSED / "ml_ready.csv"
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    feature_cols = [c for c in df.columns if c not in ("home_win", "date")]
    n = len(df)
    val_end = int(n * 0.9)

    X_test = df[feature_cols].iloc[val_end:].fillna(0)
    y_test = df["home_win"].iloc[val_end:].astype(int)
    dates_test = df["date"].iloc[val_end:]

    return X_test, y_test, dates_test, feature_cols


def main():
    models = load_models()
    if not models:
        logger.error("No models found. Run models/train.py first.")
        return

    X_test, y_test, dates_test, feature_cols = get_test_data()

    # Ensemble predictions
    probs = []
    for model_type, model in models:
        if model_type == "xgboost":
            p = model.predict_proba(X_test)[:, 1]
        else:
            p = model.predict(X_test, prediction_type="Probability")[:, 1]
        probs.append(p)

    ensemble_prob = np.mean(probs, axis=0)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)

    # Metrics
    metrics = {
        "accuracy": round(accuracy_score(y_test, ensemble_pred), 4),
        "log_loss": round(log_loss(y_test, ensemble_prob), 4),
        "roc_auc": round(roc_auc_score(y_test, ensemble_prob), 4),
        "brier_score": round(brier_score_loss(y_test, ensemble_prob), 4),
        "n_test": int(len(y_test)),
        "test_period_start": str(dates_test.iloc[0].date()),
        "test_period_end": str(dates_test.iloc[-1].date()),
        "home_win_rate": round(float(y_test.mean()), 4),
        "n_models": len(models),
    }

    save_json(MODELS_SAVED / "evaluation_metrics.json", metrics)
    logger.info(f"Accuracy: {metrics['accuracy']}")
    logger.info(f"ROC-AUC:  {metrics['roc_auc']}")
    logger.info(f"Brier:    {metrics['brier_score']}")

    # Feature importance (from first CatBoost model)
    for model_type, model in models:
        if model_type == "catboost":
            importances = model.get_feature_importance()
            fi_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": importances,
            }).sort_values("importance", ascending=False)
            fi_df.to_csv(MODELS_SAVED / "feature_importance.csv", index=False)
            logger.info(f"Top features:\n{fi_df.head(15).to_string()}")
            break

    # Confusion matrix
    cm = confusion_matrix(y_test, ensemble_pred)
    logger.info(f"Confusion Matrix:\n{cm}")

    # Classification report
    report = classification_report(y_test, ensemble_pred, target_names=["Away Win", "Home Win"])
    logger.info(f"\n{report}")

    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()
