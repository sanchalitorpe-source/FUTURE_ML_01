"""
pipeline/preprocessor.py
------------------------
Cleans raw data: types, missing values, outliers, encoding.
"""

import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("preprocessor")


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    log.info(f"  Input shape: {df.shape}")
    df = df.copy()

    # ── 1. Parse dates ────────────────────────────────────────────────────────
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"])

    # ── 2. Drop full duplicates ───────────────────────────────────────────────
    before = len(df)
    df.drop_duplicates(inplace=True)
    log.info(f"  Duplicates removed: {before - len(df)}")

    # ── 3. Handle missing values ──────────────────────────────────────────────
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    cat_cols = df.select_dtypes(include="object").columns
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    log.info(f"  Nulls remaining: {df.isnull().sum().sum()}")

    # ── 4. Clip outliers (IQR × 3) ───────────────────────────────────────────
    for col in ["sales", "units_sold", "avg_price"]:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr    = q3 - q1
        df[col] = df[col].clip(q1 - 3 * iqr, q3 + 3 * iqr)

    # ── 5. Encode categoricals ────────────────────────────────────────────────
    df["category_enc"] = pd.Categorical(df["category"]).codes
    df["region_enc"]   = pd.Categorical(df["region"]).codes

    log.info(f"  Output shape: {df.shape}")
    return df.sort_values("date").reset_index(drop=True)
