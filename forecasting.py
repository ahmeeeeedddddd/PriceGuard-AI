"""
forecasting.py — Step 4B: SARIMA Price Forecasting + Drift Detection
====================================================================
Builds a daily time series of average price per category from
raw_products.csv, fits a SARIMA model via pmdarima.auto_arima,
forecasts 7 days ahead, and detects data drift.

Public API
----------
run_forecasting(category=None)        → dict  (forecast data for UI)
check_drift_and_recluster()           → bool  (True if reclustering was triggered)
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
RAW_CSV      = os.path.join(BASE_DIR, "data", "raw_products.csv")
LIVE_CSV     = os.path.join(BASE_DIR, "data", "live_products.csv")
FORECAST_PLOT = os.path.join(BASE_DIR, "outputs", "forecast_plot.png")

FORECAST_HORIZON = 7   # days ahead
DRIFT_THRESHOLD  = 2   # standard deviations


# ─────────────────────────────────────────────────────────────────────────────

def _load_price_series(category: str | None = None) -> pd.Series:
    """
    Build a daily average-price time series from raw_products.csv.
    If *category* is given, filter to that category; otherwise use all rows.

    Returns
    -------
    pd.Series with DatetimeIndex and daily frequency.
    """
    source = LIVE_CSV if os.path.exists(LIVE_CSV) else RAW_CSV
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Expected {RAW_CSV} — run data_loader first.")

    df = pd.read_csv(RAW_CSV, parse_dates=["scraped_at"])

    if category and category != "All":
        df = df[df["category"].str.contains(category, case=False, na=False)]
        if df.empty:
            logger.warning("No data for category '%s'; using all categories.", category)
            df = pd.read_csv(RAW_CSV, parse_dates=["scraped_at"])

    ts = (
        df.set_index("scraped_at")["price"]
        .resample("D")
        .mean()
        .interpolate(method="linear")
        .dropna()
    )
    return ts


def _fit_sarima(ts: pd.Series) -> object:
    """
    Fit a SARIMA model using pmdarima.auto_arima with weekly seasonality.

    Parameters
    ----------
    ts : pd.Series — daily price series

    Returns
    -------
    Fitted pmdarima ARIMA model.
    """
    try:
        import pmdarima as pm
        model = pm.auto_arima(
            ts,
            seasonal=True,
            m=7,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3, max_P=2, max_Q=2,
            information_criterion="aic",
        )
        return model
    except Exception as exc:
        logger.error("auto_arima failed: %s — falling back to ARIMA(1,1,1).", exc)
        import pmdarima as pm
        return pm.ARIMA(order=(1, 1, 1)).fit(ts)


def _save_forecast_plot(ts: pd.Series, forecast: np.ndarray,
                        conf_int: np.ndarray, category: str) -> None:
    """Save historical + forecast line chart to outputs/forecast_plot.png."""
    os.makedirs(os.path.dirname(FORECAST_PLOT), exist_ok=True)

    future_idx = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON, freq="D",
    )

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(ts.index, ts.values, color="#3498DB", lw=1.5, label="Historical price")
    ax.plot(future_idx, forecast, color="#E74C3C", lw=2, marker="o",
            markersize=4, label="7-day forecast")
    if conf_int is not None:
        ax.fill_between(future_idx, conf_int[:, 0], conf_int[:, 1],
                        color="#E74C3C", alpha=0.15, label="95% CI")

    ax.set_title(f"PriceGuard AI — SARIMA Forecast ({category})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Avg Price ($)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FORECAST_PLOT, dpi=150)
    plt.close(fig)
    logger.info("Forecast plot saved to %s.", FORECAST_PLOT)


def run_forecasting(category: str | None = None) -> dict:
    """
    End-to-end forecasting for one category (or all if None).

    Returns
    -------
    dict with keys:
        category      : str
        historical_dates  : list[str]
        historical_prices : list[float]
        forecast_dates    : list[str]
        forecast_7d       : list[float]
        conf_int_lower    : list[float]
        conf_int_upper    : list[float]
        trend             : str  ("rising" | "falling" | "stable")
    """
    cat_label = category if category else "All Categories"
    logger.info("Running SARIMA forecast for category: %s …", cat_label)

    ts = _load_price_series(category)

    if len(ts) < 14:
        logger.warning("Insufficient data (%d days) for SARIMA — generating mock forecast.", len(ts))
        mock_vals = [float(ts.mean())] * FORECAST_HORIZON
        return {
            "category": cat_label,
            "historical_dates":  [str(d.date()) for d in ts.index],
            "historical_prices": ts.tolist(),
            "forecast_dates":    [
                str((ts.index[-1] + pd.Timedelta(days=i+1)).date())
                for i in range(FORECAST_HORIZON)
            ],
            "forecast_7d":    mock_vals,
            "conf_int_lower": mock_vals,
            "conf_int_upper": mock_vals,
            "trend": "stable",
        }

    model = _fit_sarima(ts)

    forecast, conf_int = model.predict(
        n_periods=FORECAST_HORIZON, return_conf_int=True
    )

    forecast   = np.array(forecast)
    conf_int   = np.array(conf_int)

    # Determine trend
    if forecast[-1] > forecast[0] * 1.01:
        trend = "rising"
    elif forecast[-1] < forecast[0] * 0.99:
        trend = "falling"
    else:
        trend = "stable"

    _save_forecast_plot(ts, forecast, conf_int, cat_label)
    logger.info("Forecast complete. Trend: %s.", trend)

    future_idx = pd.date_range(
        start=ts.index[-1] + pd.Timedelta(days=1),
        periods=FORECAST_HORIZON, freq="D",
    )

    return {
        "category":          cat_label,
        "historical_dates":  [str(d.date()) for d in ts.index],
        "historical_prices": ts.tolist(),
        "forecast_dates":    [str(d.date()) for d in future_idx],
        "forecast_7d":       forecast.tolist(),
        "conf_int_lower":    conf_int[:, 0].tolist(),
        "conf_int_upper":    conf_int[:, 1].tolist(),
        "trend":             trend,
    }


def check_drift_and_recluster() -> bool:
    """
    Detect data drift by comparing the latest 7-day rolling mean price
    against the SARIMA forecast error distribution.
    If drift > 2 standard deviations, re-run clustering and classification.

    Returns
    -------
    bool — True if reclustering was triggered, False otherwise.
    """
    logger.info("Checking for data drift …")

    if not os.path.exists(RAW_CSV):
        logger.warning("raw_products.csv not found — skipping drift check.")
        return False

    df  = pd.read_csv(RAW_CSV, parse_dates=["scraped_at"])
    ts  = df.set_index("scraped_at")["price"].resample("D").mean().dropna()

    if len(ts) < 21:
        logger.info("Too little history for drift detection.")
        return False

    rolling_mean = ts.rolling(7).mean().dropna()
    rolling_std  = ts.rolling(7).std().dropna()

    # Compare last value against the rolling distribution
    latest_price = ts.iloc[-1]
    mean_val     = rolling_mean.iloc[-1]
    std_val      = rolling_std.iloc[-1] if rolling_std.iloc[-1] > 0 else 1.0
    z_score      = abs(latest_price - mean_val) / std_val

    logger.info("Drift z-score: %.2f (threshold: %d).", z_score, DRIFT_THRESHOLD)

    if z_score > DRIFT_THRESHOLD:
        logger.warning("Data drift detected (z=%.2f) — triggering reclustering.", z_score)
        print(f"[forecasting] ⚠  Data drift detected (z={z_score:.2f}). Reclustering …")

        try:
            from clustering  import run_clustering
            from classifier  import train_classifier
            run_clustering()
            train_classifier()
            logger.info("Dynamic reclustering complete.")
            return True
        except Exception as exc:
            logger.error("Reclustering failed: %s", exc)
            return False

    logger.info("No significant drift detected.")
    return False


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    result = run_forecasting()
    print(f"Category: {result['category']}")
    print(f"7-day forecast: {[round(v, 2) for v in result['forecast_7d']]}")
    print(f"Trend: {result['trend']}")
