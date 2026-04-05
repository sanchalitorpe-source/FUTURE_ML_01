"""
pipeline/data_loader.py
-----------------------
Loads sales data from CSV (or generates synthetic data if file absent).
"""

import os
import numpy as np
import pandas as pd
from utils.logger import get_logger

log = get_logger("data_loader")

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Food & Beverage"]
REGIONS     = ["North", "South", "East", "West", "Central"]

# ── Synthetic data generator ──────────────────────────────────────────────────

def _generate_synthetic(path: str) -> pd.DataFrame:
    """Creates 3 years of realistic monthly sales data and saves to CSV."""
    log.info("  No CSV found — generating synthetic dataset ...")

    rng  = np.random.default_rng(42)
    rows = []

    for year in range(2022, 2025):
        for month in range(1, 13):
            for cat in CATEGORIES:
                for region in REGIONS:
                    base = {
                        "Electronics":      8_000,
                        "Clothing":         5_000,
                        "Home & Garden":    4_500,
                        "Sports":           3_500,
                        "Food & Beverage":  6_000,
                    }[cat]

                    region_mult = {
                        "North": 1.0, "South": 0.9,
                        "East":  1.1, "West":  1.25, "Central": 0.85,
                    }[region]

                    # Seasonal pattern
                    seasonal = 1 + 0.25 * np.sin((month - 3) * np.pi / 6)

                    # Yearly growth
                    growth = 1 + 0.08 * (year - 2022)

                    noise = rng.normal(1.0, 0.07)

                    sales = base * region_mult * seasonal * growth * noise
                    units = int(sales / rng.uniform(15, 45))
                    price = sales / max(units, 1)

                    rows.append({
                        "date":           f"{year}-{month:02d}-01",
                        "category":       cat,
                        "region":         region,
                        "sales":          round(sales, 2),
                        "units_sold":     units,
                        "avg_price":      round(price, 2),
                        "promotion":      int(rng.random() < 0.20),
                        "competitor_idx": round(rng.uniform(0.8, 1.2), 3),
                    })

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(path), exist_ok=True) if "/" in path else None
    df.to_csv(path, index=False)
    log.info(f"  Synthetic data saved → {path}  ({len(df):,} rows)")
    return df


# ── Public API ────────────────────────────────────────────────────────────────

def load_data(path: str = "data/sales_data.csv") -> pd.DataFrame:
    """Load CSV from *path*; generate synthetic data if not found."""
    if os.path.exists(path):
        df = pd.read_csv(path, parse_dates=["date"])
        log.info(f"  Loaded {len(df):,} rows from {path}")
    else:
        df = _generate_synthetic(path)
        df["date"] = pd.to_datetime(df["date"])

    return df
