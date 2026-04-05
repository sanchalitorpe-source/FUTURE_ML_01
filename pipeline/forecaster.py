"""
pipeline/forecaster.py
-----------------------
Generates future month forecasts by iteratively building features
from the most recent known data + previously predicted values.
"""

import numpy as np
import pandas as pd
from pipeline.feature_eng import FEATURE_COLS, engineer_features
from utils.logger import get_logger

log = get_logger("forecaster")

CATEGORIES = ["Electronics", "Clothing", "Home & Garden", "Sports", "Food & Beverage"]
REGIONS     = ["North", "South", "East", "West", "Central"]


def forecast_future(model, hist_df: pd.DataFrame, months: int = 6) -> pd.DataFrame:
    """
    Iteratively forecast *months* months ahead for every (category, region) pair.

    Strategy:
      - Build a future row stub for each combo.
      - Re-run engineer_features on (history + future stub).
      - Predict → append prediction as the new 'actual' for the next iteration.
    """
    running_df = hist_df.copy()
    last_date  = running_df["date"].max()

    all_preds = []

    for step in range(1, months + 1):
        target_date = last_date + pd.DateOffset(months=step)
        log.info(f"  Forecasting {target_date.strftime('%Y-%m')} ...")

        stubs = []
        for cat in CATEGORIES:
            for reg in REGIONS:
                # Use last known avg_price / competitor_idx for that group
                grp_hist = running_df[
                    (running_df["category"] == cat) &
                    (running_df["region"]   == reg)
                ].sort_values("date")

                last_price  = grp_hist["avg_price"].iloc[-1]  if len(grp_hist) else 25.0
                last_comp   = grp_hist["competitor_idx"].iloc[-1] if len(grp_hist) else 1.0
                last_units  = grp_hist["units_sold"].iloc[-1] if len(grp_hist) else 100

                stub = {
                    "date":           target_date,
                    "category":       cat,
                    "region":         reg,
                    "sales":          np.nan,          # unknown — will be predicted
                    "units_sold":     int(last_units * np.random.uniform(0.95, 1.05)),
                    "avg_price":      round(last_price * np.random.uniform(0.98, 1.02), 2),
                    "promotion":      0,
                    "competitor_idx": round(last_comp  * np.random.uniform(0.99, 1.01), 3),
                    "category_enc":   pd.Categorical(
                                          [cat], categories=running_df["category"].unique()
                                      ).codes[0],
                    "region_enc":     pd.Categorical(
                                          [reg], categories=running_df["region"].unique()
                                      ).codes[0],
                }
                stubs.append(stub)

        stub_df = pd.DataFrame(stubs)

        # Combine history + stubs so engineer_features can compute lags/rolls
        combined = pd.concat([running_df, stub_df], ignore_index=True)
        combined = engineer_features(combined)

        # Extract only the stub rows
        future_feats = combined[combined["date"] == target_date]

        X_fut = future_feats[FEATURE_COLS]
        preds = model.predict(X_fut)

        # Fill predicted sales back into stubs
        for i, idx in enumerate(future_feats.index):
            stub_df.loc[stub_df["date"] == target_date, "sales"] = preds

        # Record predictions
        for i, (cat, reg) in enumerate([(c, r) for c in CATEGORIES for r in REGIONS]):
            all_preds.append({
                "date":     target_date,
                "category": cat,
                "region":   reg,
                "forecast": round(preds[i], 2),
            })

        # Append filled stubs to running history for next iteration
        stub_df_filled = stub_df.copy()
        stub_df_filled["sales"] = preds
        running_df = pd.concat([running_df, stub_df_filled], ignore_index=True)

    forecast_df = pd.DataFrame(all_preds)
    log.info(f"  Total forecast rows: {len(forecast_df)}")
    return forecast_df
