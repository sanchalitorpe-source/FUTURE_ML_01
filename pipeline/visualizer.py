"""
pipeline/visualizer.py
-----------------------
Generates all charts and saves them to outputs/.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec

os.makedirs("outputs", exist_ok=True)

PALETTE  = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]
CATEGORY_COLORS = {
    "Electronics":      "#2563EB",
    "Clothing":         "#10B981",
    "Home & Garden":    "#F59E0B",
    "Sports":           "#EF4444",
    "Food & Beverage":  "#8B5CF6",
}


def _fmt_dollar(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"${x/1_000:.0f}K")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


# ── Individual charts ─────────────────────────────────────────────────────────

def plot_sales_trend(df: pd.DataFrame, forecast_df: pd.DataFrame | None, ax):
    """Monthly total sales + optional forecast ribbon."""
    monthly = df.groupby("date")["sales"].sum().reset_index()

    ax.plot(monthly["date"], monthly["sales"], color="#2563EB", linewidth=2.5, label="Actual")
    ax.fill_between(monthly["date"], monthly["sales"], alpha=0.08, color="#2563EB")

    if forecast_df is not None:
        fcast = forecast_df.groupby("date")["forecast"].sum().reset_index()
        # Bridge: last actual → first forecast
        bridge = pd.DataFrame({
            "date":     [monthly["date"].iloc[-1], fcast["date"].iloc[0]],
            "forecast": [monthly["sales"].iloc[-1], fcast["forecast"].iloc[0]],
        })
        combined = pd.concat([bridge, fcast])
        ax.plot(combined["date"], combined["forecast"], color="#EF4444",
                linewidth=2, linestyle="--", label="Forecast")
        ax.fill_between(combined["date"],
                        combined["forecast"] * 0.90,
                        combined["forecast"] * 1.10,
                        alpha=0.12, color="#EF4444", label="±10% CI")

    ax.set_title("Monthly Sales Trend & Forecast", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Sales")
    _fmt_dollar(ax)
    ax.legend(framealpha=0.8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def plot_region_bar(df: pd.DataFrame, ax):
    """2024 sales by region — horizontal bar."""
    region_sales = (
        df[df["date"].dt.year == df["date"].dt.year.max()]
        .groupby("region")["sales"]
        .sum()
        .sort_values()
    )
    bars = ax.barh(region_sales.index, region_sales.values,
                   color=PALETTE[:len(region_sales)], edgecolor="white")
    for bar, val in zip(bars, region_sales.values):
        ax.text(val + 2_000, bar.get_y() + bar.get_height() / 2,
                f"${val/1e6:.1f}M", va="center", fontsize=9)
    ax.set_title("Sales by Region (Latest Year)", fontweight="bold")
    ax.set_xlabel("Total Sales")
    _fmt_dollar(ax, axis="x")
    ax.grid(axis="x", alpha=0.3)


def plot_category_pie(df: pd.DataFrame, ax):
    """Category share donut chart."""
    cat_sales = df.groupby("category")["sales"].sum()
    colors = [CATEGORY_COLORS[c] for c in cat_sales.index]
    wedges, texts, autotexts = ax.pie(
        cat_sales.values,
        labels=cat_sales.index,
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        wedgeprops=dict(width=0.55, edgecolor="white", linewidth=1.5),
        pctdistance=0.78,
    )
    for t in autotexts:
        t.set_fontsize(8)
    ax.set_title("Sales by Category", fontweight="bold")


def plot_feature_importance(model, feature_names: list, ax):
    """Top-15 feature importances."""
    imp    = model.feature_importances_
    pairs  = sorted(zip(feature_names, imp), key=lambda x: x[1])[-15:]
    names  = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    ax.barh(names, values, color=colors, edgecolor="white")
    ax.set_title("Feature Importance (Top 15)", fontweight="bold")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)


def plot_forecast_stacked(forecast_df: pd.DataFrame, ax):
    """Stacked bar: monthly forecast by category."""
    pivot = forecast_df.groupby(["date", "category"])["forecast"].sum().unstack(fill_value=0)
    pivot.plot.bar(stacked=True, ax=ax,
                   color=[CATEGORY_COLORS[c] for c in pivot.columns],
                   edgecolor="white", linewidth=0.4)
    ax.set_title("Forecast by Category (Next 6 Months)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("Forecast Sales")
    ax.set_xticklabels([d.strftime("%b %Y") for d in pivot.index], rotation=30, ha="right")
    _fmt_dollar(ax)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
    ax.grid(axis="y", alpha=0.3)


# ── Master layout ─────────────────────────────────────────────────────────────

def plot_all(df: pd.DataFrame, forecast_df: pd.DataFrame | None, metrics: dict):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#F8FAFC")
    gs  = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.35)

    # Row 0
    ax0 = fig.add_subplot(gs[0, :])    # full width — trend
    ax1 = fig.add_subplot(gs[1, 0])    # region bar
    ax2 = fig.add_subplot(gs[1, 1])    # category pie
    ax3 = fig.add_subplot(gs[1, 2])    # feature importance OR stacked forecast

    for ax in [ax0, ax1, ax3]:
        ax.set_facecolor("#FFFFFF")

    plot_sales_trend(df, forecast_df, ax0)
    plot_region_bar(df, ax1)
    plot_category_pie(df, ax2)

    if forecast_df is not None:
        plot_forecast_stacked(forecast_df, ax3)

    # Metrics annotation
    m_text = (f"R²={metrics['r2']}   RMSE=${metrics['rmse']:,.0f}"
              f"   MAE=${metrics['mae']:,.0f}   MAPE={metrics['mape']}%")
    fig.text(0.5, 0.97, m_text, ha="center", fontsize=11,
             color="#374151", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", fc="#E0F2FE", ec="#BAE6FD"))

    fig.suptitle("Sales & Demand Forecasting Dashboard", fontsize=15,
                 fontweight="bold", y=1.01, color="#111827")

    out = "outputs/dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Dashboard saved → {out}")
