"""
Sales & Demand Forecasting Pipeline
====================================
Entry point — runs the full pipeline end-to-end.

Usage:
    python main.py
    python main.py --mode train
    python main.py --mode forecast --months 6
"""

import argparse
from pipeline.data_loader   import load_data
from pipeline.preprocessor  import preprocess
from pipeline.feature_eng   import engineer_features
from pipeline.trainer        import train_model
from pipeline.evaluator      import evaluate
from pipeline.forecaster     import forecast_future
from pipeline.visualizer     import plot_all
from utils.logger            import get_logger

log = get_logger("main")


def run(mode: str = "full", months: int = 6):
    log.info("=== Sales Forecasting Pipeline START ===")

    # 1. Load
    log.info("[1/7] Loading data ...")
    raw_df = load_data("data/sales_data.csv")

    # 2. Preprocess
    log.info("[2/7] Preprocessing ...")
    clean_df = preprocess(raw_df)

    # 3. Feature engineering
    log.info("[3/7] Engineering features ...")
    feat_df = engineer_features(clean_df)

    # 4. Train
    log.info("[4/7] Training model ...")
    model, X_test, y_test = train_model(feat_df)

    # 5. Evaluate
    log.info("[5/7] Evaluating ...")
    metrics = evaluate(model, X_test, y_test)
    log.info(f"  R²={metrics['r2']:.4f}  MAPE={metrics['mape']:.2f}%  RMSE={metrics['rmse']:.2f}")

    if mode in ("full", "forecast"):
        # 6. Forecast
        log.info(f"[6/7] Forecasting next {months} months ...")
        forecast_df = forecast_future(model, feat_df, months=months)
        forecast_df.to_csv("outputs/forecast.csv", index=False)
        log.info("  Saved → outputs/forecast.csv")

    # 7. Visualize
    log.info("[7/7] Generating charts ...")
    plot_all(clean_df, forecast_df if mode != "train" else None, metrics)
    log.info("  Saved → outputs/")

    log.info("=== Pipeline COMPLETE ===")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sales Forecasting Pipeline")
    parser.add_argument("--mode",   default="full",  choices=["full", "train", "forecast"])
    parser.add_argument("--months", default=6,       type=int)
    args = parser.parse_args()
    run(mode=args.mode, months=args.months)
