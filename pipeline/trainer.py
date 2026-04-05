"""
pipeline/trainer.py
-------------------
Trains a Gradient Boosting model with time-based train/test split.
Also saves the trained model to disk.
"""

import os
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from pipeline.feature_eng import FEATURE_COLS, TARGET_COL
from utils.logger import get_logger

log = get_logger("trainer")

MODEL_PATH = "models/gbr_model.pkl"

# ── Hyperparameters ───────────────────────────────────────────────────────────
GBR_PARAMS = dict(
    n_estimators      = 400,
    learning_rate     = 0.05,
    max_depth         = 5,
    min_samples_split = 10,
    subsample         = 0.8,
    random_state      = 42,
    loss              = "huber",          # robust to outliers
)


def train_model(df: pd.DataFrame):
    """
    Time-based split: last 6 months → test, rest → train.
    Returns (fitted_model, X_test, y_test).
    """
    df = df.copy()

    # ── Time-based split ──────────────────────────────────────────────────────
    cutoff  = df["date"].max() - pd.DateOffset(months=6)
    train   = df[df["date"] <= cutoff]
    test    = df[df["date"]  > cutoff]

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]
    X_test  = test[FEATURE_COLS]
    y_test  = test[TARGET_COL]

    log.info(f"  Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")

    # ── Cross-validation (on training set only) ───────────────────────────────
    model_cv = GradientBoostingRegressor(**GBR_PARAMS)
    cv_scores = cross_val_score(model_cv, X_train, y_train, cv=5, scoring="r2")
    log.info(f"  CV R² scores: {[f'{s:.3f}' for s in cv_scores]}  mean={cv_scores.mean():.3f}")

    # ── Final fit ─────────────────────────────────────────────────────────────
    model = GradientBoostingRegressor(**GBR_PARAMS)
    model.fit(X_train, y_train)

    # ── Persist model ─────────────────────────────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    log.info(f"  Model saved → {MODEL_PATH}")

    return model, X_test, y_test


def load_model():
    """Load a previously saved model from disk."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"No saved model at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)
