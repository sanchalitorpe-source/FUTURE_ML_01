"""
config.py
---------
Central configuration — paths, hyperparameters, constants.
Override via environment variables for production.
"""

import os

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH    = os.getenv("DATA_PATH",  "data/sales_data.csv")
MODEL_PATH   = os.getenv("MODEL_PATH", "models/gbr_model.pkl")
OUTPUT_DIR   = os.getenv("OUTPUT_DIR", "outputs")

# ── Model ─────────────────────────────────────────────────────────────────────
GBR_PARAMS = {
    "n_estimators":      400,
    "learning_rate":     0.05,
    "max_depth":         5,
    "min_samples_split": 10,
    "subsample":         0.8,
    "random_state":      42,
    "loss":              "huber",
}

# ── Forecasting ───────────────────────────────────────────────────────────────
FORECAST_MONTHS = 6
TEST_MONTHS     = 6          # tail months reserved for test split

# ── Data ─────────────────────────────────────────────────────────────────────
CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Food & Beverage"]
REGIONS    = ["North", "South", "East", "West", "Central"]
