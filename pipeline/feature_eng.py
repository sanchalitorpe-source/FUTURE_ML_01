"""
pipeline/feature_eng.py
-----------------------
Creates temporal, lag, rolling, and interaction features.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("feature_eng")


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ── 1. Temporal features ──────────────────────────────────────────────────
    df["year"]        = df["date"].dt.year
    df["month"]       = df["date"].dt.month
    df["quarter"]     = df["date"].dt.quarter
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)   # cyclical encoding
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    df["is_q4"]       = (df["quarter"] == 4).astype(int)
    df["trend"]       = (df["year"] - df["year"].min()) * 12 + df["month"]

    # ── 2. Lag features (per category-region group) ───────────────────────────
    grp = ["category", "region"]
    df = df.sort_values(["category", "region", "date"])

    for lag in [1, 2, 3, 6, 12]:
        df[f"sales_lag_{lag}"] = df.groupby(grp)["sales"].shift(lag)

    # ── 3. Rolling statistics ─────────────────────────────────────────────────
    for window in [3, 6]:
        rolled = df.groupby(grp)["sales"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"sales_roll_mean_{window}"] = rolled

        rolled_std = df.groupby(grp)["sales"].transform(
            lambda s: s.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"sales_roll_std_{window}"] = rolled_std

    # ── 4. Price × units interaction ──────────────────────────────────────────
    df["revenue_proxy"] = df["avg_price"] * df["units_sold"]

    # ── 5. Promotion interaction ──────────────────────────────────────────────
    df["promo_price_interact"] = df["promotion"] * df["avg_price"]

    # ── 6. Drop rows with NaN lags (first 12 rows per group) ─────────────────
    before = len(df)
    df.dropna(subset=[c for c in df.columns if "lag" in c or "roll" in c], inplace=True)
    log.info(f"  Rows after lag/roll drop: {len(df)} (removed {before - len(df)})")

    return df.reset_index(drop=True)


FEATURE_COLS = [
    "category_enc", "region_enc",
    "year", "month", "quarter",
    "month_sin", "month_cos", "is_q4", "trend",
    "avg_price", "units_sold", "promotion",
    "competitor_idx", "revenue_proxy", "promo_price_interact",
    "sales_lag_1", "sales_lag_2", "sales_lag_3", "sales_lag_6", "sales_lag_12",
    "sales_roll_mean_3", "sales_roll_mean_6",
    "sales_roll_std_3",  "sales_roll_std_6",
]
TARGET_COL = "sales"
