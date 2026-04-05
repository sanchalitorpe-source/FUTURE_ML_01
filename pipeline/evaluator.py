"""
pipeline/evaluator.py
---------------------
Computes regression metrics and saves a residual plot.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from utils.logger import get_logger

log = get_logger("evaluator")


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error (ignores zero-true rows)."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def evaluate(model, X_test, y_test) -> dict:
    """Evaluate model and return metrics dict."""
    y_pred = model.predict(X_test)

    metrics = {
        "r2":   round(r2_score(y_test, y_pred), 4),
        "rmse": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "mae":  round(mean_absolute_error(y_test, y_pred), 2),
        "mape": round(mape(y_test.values, y_pred), 2),
    }

    log.info(f"  Metrics → {metrics}")

    # ── Residual plot ─────────────────────────────────────────────────────────
    residuals = y_test.values - y_pred
    os.makedirs("outputs", exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Model Evaluation", fontsize=14, fontweight="bold")

    # Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.4, color="#2563EB", edgecolors="none")
    lim = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
    axes[0].plot(lim, lim, "r--", linewidth=1.5, label="Perfect fit")
    axes[0].set_xlabel("Actual Sales ($)")
    axes[0].set_ylabel("Predicted Sales ($)")
    axes[0].set_title(f"Actual vs Predicted  (R²={metrics['r2']})")
    axes[0].legend()

    # Residuals distribution
    axes[1].hist(residuals, bins=40, color="#10B981", edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="red", linestyle="--")
    axes[1].set_xlabel("Residual ($)")
    axes[1].set_ylabel("Count")
    axes[1].set_title(f"Residuals  (MAPE={metrics['mape']}%)")

    plt.tight_layout()
    plt.savefig("outputs/evaluation.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  Chart saved → outputs/evaluation.png")

    return metrics
