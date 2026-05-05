"""
data_loader.py — Step 1A: Kaggle Dataset Ingestion & Cleaning
=============================================================
Loads the pre-downloaded amazon_products.csv, maps it to the
PriceGuard schema, synthesises timestamps for SARIMA, cleans
outliers, and saves data/raw_products.csv.

Public API
----------
run_data_loader() → pd.DataFrame
"""

import os
import logging
import numpy as np
import pandas as pd

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_CSV   = os.path.join(BASE_DIR, "data", "amazon_products.csv")
OUT_CSV   = os.path.join(BASE_DIR, "data", "raw_products.csv")

# ── Column mapping from Kaggle schema → PriceGuard schema ────────────────────
KAGGLE_COL_MAP = {
    "Product Name": "product_name",
    "Selling Price": "price",
    "Category": "category",
}

NUMERIC_FEATURES = ["price", "rating", "review_count", "discount_percentage"]


# ─────────────────────────────────────────────────────────────────────────────

def _parse_price(val) -> float:
    """
    Convert Kaggle price strings like '$237.68' or '$74.99 - $249.99'
    to a float. Returns NaN for unparseable values.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    # Take first price in a range
    s = s.split("-")[0].strip()
    s = s.replace("$", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _extract_category(raw_cat: str) -> str:
    """
    Kaggle category strings look like:
        'Toys & Games | Building Toys | Building Sets'
    We keep only the top-level category for clustering / SARIMA.
    """
    if pd.isna(raw_cat):
        return "Unknown"
    return str(raw_cat).split("|")[0].strip()


def _synthesise_timestamps(n: int) -> pd.DatetimeIndex:
    """
    Spread *n* rows evenly across the past 90 days so SARIMA has a
    daily time series to train on.
    """
    end   = pd.Timestamp.today().normalize()
    start = end - pd.Timedelta(days=89)
    dates = pd.date_range(start=start, end=end, periods=n)
    return dates


def load_and_clean() -> pd.DataFrame:
    """
    Load amazon_products.csv, apply PriceGuard schema, clean, add
    synthetic timestamps, and return the cleaned DataFrame.

    Returns
    -------
    pd.DataFrame
        Columns: product_name, price, rating, review_count,
                 discount_percentage, category, stock_status, scraped_at
    """
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Kaggle dataset not found at {RAW_CSV}. "
            "Please place amazon_products.csv in the data/ folder."
        )

    logger.info("Loading Kaggle dataset from %s …", RAW_CSV)
    df = pd.read_csv(RAW_CSV, encoding="utf-8", on_bad_lines="skip", low_memory=False)
    logger.info("Loaded %d rows, %d columns.", len(df), len(df.columns))

    # ── Rename core columns ───────────────────────────────────────────────────
    df = df.rename(columns=KAGGLE_COL_MAP)

    # ── Parse price ───────────────────────────────────────────────────────────
    df["price"] = df["price"].apply(_parse_price)

    # ── Rating (some rows have text — coerce to numeric) ─────────────────────
    if "rating" not in df.columns:
        df["rating"] = np.nan
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    # ── review_count ─────────────────────────────────────────────────────────
    if "review_count" not in df.columns:
        # Kaggle Amazon dataset often lacks review counts; generate synthetic
        rng = np.random.default_rng(42)
        df["review_count"] = rng.integers(0, 5000, size=len(df)).astype(int)
    else:
        df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce").fillna(0).astype(int)

    # ── discount_percentage ───────────────────────────────────────────────────
    if "discount_percentage" not in df.columns:
        # Kaggle dataset has List Price vs Selling Price — compute discount
        if "List Price" in df.columns:
            list_price = df["List Price"].apply(_parse_price)
            df["discount_percentage"] = (
                ((list_price - df["price"]) / list_price.replace(0, np.nan)) * 100
            ).clip(lower=0)
        else:
            rng = np.random.default_rng(43)
            df["discount_percentage"] = rng.uniform(0, 40, size=len(df))
    else:
        df["discount_percentage"] = pd.to_numeric(df["discount_percentage"], errors="coerce").fillna(0)

    # ── category ─────────────────────────────────────────────────────────────
    if "category" not in df.columns:
        df["category"] = "Unknown"
    else:
        df["category"] = df["category"].apply(_extract_category)

    # ── stock_status (1 = in stock, not in Kaggle dataset) ───────────────────
    df["stock_status"] = 1

    # ── Drop rows with missing price or rating ────────────────────────────────
    before = len(df)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    # Fill missing ratings with median
    df["rating"] = df["rating"].fillna(df["rating"].median())
    df["rating"] = df["rating"].clip(lower=0, upper=5)
    logger.info("Dropped %d rows with null/invalid price.", before - len(df))

    # ── Cap review_count at 99th percentile ───────────────────────────────────
    cap = df["review_count"].quantile(0.99)
    df["review_count"] = df["review_count"].clip(upper=cap).astype(int)

    # ── Synthesise scraped_at timestamps (90-day spread) ─────────────────────
    df = df.reset_index(drop=True)
    df["scraped_at"] = _synthesise_timestamps(len(df))

    # ── Keep only PriceGuard columns ─────────────────────────────────────────
    keep = ["product_name", "price", "rating", "review_count",
            "discount_percentage", "category", "stock_status", "scraped_at"]
    df = df[keep].reset_index(drop=True)

    logger.info("Cleaned dataset: %d rows ready for clustering.", len(df))
    return df


def run_data_loader() -> pd.DataFrame:
    """
    Entry-point called by pipeline.py.

    Loads, cleans, and saves data/raw_products.csv.

    Returns
    -------
    pd.DataFrame
    """
    df = load_and_clean()
    os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    logger.info("Saved cleaned dataset to %s.", OUT_CSV)
    print(f"[data_loader] ✓  {len(df)} products saved to {OUT_CSV}")
    return df


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_data_loader()
    print(df.head())
    print(df.dtypes)
