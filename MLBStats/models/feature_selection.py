"""SHAP-based feature importance analysis."""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shap
from catboost import CatBoostClassifier

from utils.helpers import get_logger, DATA_PROCESSED, MODELS_SAVED

logger = get_logger("feature_selection")


def main():
    # Load first CatBoost model
    model_path = MODELS_SAVED / "model_1.cbm"
    if not model_path.exists():
        logger.error("No model found. Run train.py first.")
        return

    model = CatBoostClassifier()
    model.load_model(str(model_path))

    # Load test data
    df = pd.read_csv(DATA_PROCESSED / "ml_ready.csv")
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    feature_cols = [c for c in df.columns if c not in ("home_win", "date")]
    n = len(df)
    X_test = df[feature_cols].iloc[int(n * 0.9):].fillna(0)

    # SHAP
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Mean absolute SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "mean_abs_shap": mean_shap,
    }).sort_values("mean_abs_shap", ascending=False)

    importance_df.to_csv(MODELS_SAVED / "feature_importance.csv", index=False)
    logger.info(f"Top 20 features by SHAP:\n{importance_df.head(20).to_string()}")


if __name__ == "__main__":
    main()
