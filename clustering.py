"""
clustering.py — Step 2: Unsupervised Clustering (DBSCAN + Fuzzy C-Means)
=========================================================================
Pipeline:
  1. Load data/raw_products.csv
  2. Normalise numeric features with StandardScaler
  3. Run DBSCAN to flag / remove outliers (label == -1)
  4. Run Fuzzy C-Means (skfuzzy) on clean data, n_clusters=3
  5. Sort clusters by mean price → 0=budget, 1=mid-range, 2=premium
  6. Save data/labeled_products.csv
  7. Save PCA scatter plot to outputs/cluster_plot.png

Public API
----------
run_clustering() → pd.DataFrame
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import joblib
import skfuzzy as fuzz

warnings.simplefilter("ignore", FutureWarning)

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RAW_CSV     = os.path.join(BASE_DIR, "data", "raw_products.csv")
LABELED_CSV   = os.path.join(BASE_DIR, "data", "labeled_products.csv")
PLOT_PATH     = os.path.join(BASE_DIR, "outputs", "cluster_plot.png")
CENTROIDS_PATH = os.path.join(BASE_DIR, "models", "fcm_centroids.pkl")

FEATURES    = ["price", "rating", "review_count", "discount_percentage", "stock_status"]
N_CLUSTERS  = 3
RANDOM_SEED = 42


# ─────────────────────────────────────────────────────────────────────────────

def _load_data() -> pd.DataFrame:
    """Load raw_products.csv and ensure required columns exist."""
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(f"Expected {RAW_CSV} — run data_loader first.")
    df = pd.read_csv(RAW_CSV, parse_dates=["scraped_at"])
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in raw_products.csv: {missing}")
    return df


def _remove_dbscan_outliers(X_scaled: np.ndarray,
                             df: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Run DBSCAN (eps=0.5, min_samples=5) and drop outlier rows (label == -1).

    Returns
    -------
    X_clean : np.ndarray  — scaled features, outliers removed
    df_clean : pd.DataFrame — matching rows
    """
    db = DBSCAN(eps=0.5, min_samples=5)
    db_labels = db.fit_predict(X_scaled)
    mask = db_labels != -1
    n_outliers = (~mask).sum()
    logger.info("DBSCAN removed %d outliers (%.1f%%).",
                n_outliers, 100 * n_outliers / len(mask))
    return X_scaled[mask], df[mask].reset_index(drop=True)


def _fuzzy_cmeans(X_scaled: np.ndarray) -> np.ndarray:
    """
    Run Fuzzy C-Means via skfuzzy and return hard cluster assignments
    (argmax of membership matrix).

    skfuzzy expects features as columns — shape (n_features, n_samples).
    """
    data = X_scaled.T  # skfuzzy convention: (features, samples)
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data,
        c=N_CLUSTERS,
        m=2,           # fuzziness coefficient
        error=0.005,
        maxiter=1000,
        init=None,
        seed=RANDOM_SEED,
    )
    hard_labels = np.argmax(u, axis=0)  # shape (n_samples,)
    return hard_labels, cntr


def _remap_by_price(labels: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """
    Sort raw FCM cluster IDs by their mean price so that:
        0 = budget  (lowest mean price)
        1 = mid-range
        2 = premium (highest mean price)
    """
    mean_prices = {}
    for c in range(N_CLUSTERS):
        mask = labels == c
        mean_prices[c] = df.loc[mask, "price"].mean() if mask.any() else 0.0

    sorted_clusters = sorted(mean_prices, key=mean_prices.get)  # ascending price
    remap = {old: new for new, old in enumerate(sorted_clusters)}
    return np.array([remap[l] for l in labels])


def _save_cluster_plot(df_labeled: pd.DataFrame, X_scaled: np.ndarray) -> None:
    """
    Create a 2-D PCA scatter plot coloured by cluster label and save to
    outputs/cluster_plot.png.
    """
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)

    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(X_scaled)

    label_names = {0: "Budget", 1: "Mid-Range", 2: "Premium"}
    colours     = {0: "#E74C3C", 1: "#F39C12", 2: "#2ECC71"}

    fig, ax = plt.subplots(figsize=(9, 6))
    for c in range(N_CLUSTERS):
        mask = df_labeled["cluster_label"] == c
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=12, alpha=0.6,
            color=colours[c],
            label=f"{label_names[c]} (n={mask.sum()})",
        )

    ax.set_title("PriceGuard AI — Product Pricing Clusters (PCA)")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=150)
    plt.close(fig)
    logger.info("Cluster plot saved to %s.", PLOT_PATH)


def run_clustering() -> pd.DataFrame:
    """
    Full clustering pipeline. Loads raw data, removes outliers, fits FCM,
    relabels clusters by price, persists labeled_products.csv and cluster_plot.png.

    Returns
    -------
    pd.DataFrame with new column 'cluster_label' (int: 0, 1, 2).
    """
    df = _load_data()
    logger.info("Clustering %d products …", len(df))

    # ── Normalise ─────────────────────────────────────────────────────────────
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURES].fillna(0))

    # ── DBSCAN outlier removal ─────────────────────────────────────────────────
    X_clean, df_clean = _remove_dbscan_outliers(X_scaled, df)

    # ── Fuzzy C-Means ─────────────────────────────────────────────────────────
    raw_labels, cntr = _fuzzy_cmeans(X_clean)
    final_labels   = _remap_by_price(raw_labels, df_clean)
    
    # ── Save Centroids for Scoring (μ-signal) ──────────────────────────────
    # Map raw centroids to the same 0,1,2 sort order as labels
    mean_prices = {}
    for c in range(N_CLUSTERS):
        mask = raw_labels == c
        mean_prices[c] = df_clean.loc[mask, "price"].mean() if mask.any() else 0.0
    sorted_idx = sorted(mean_prices, key=mean_prices.get)
    ordered_centroids = cntr[sorted_idx]
    joblib.dump(ordered_centroids, CENTROIDS_PATH)
    logger.info("FCM Centroids saved to %s.", CENTROIDS_PATH)

    df_clean       = df_clean.copy()
    df_clean["cluster_label"] = final_labels

    logger.info(
        "Cluster distribution — budget: %d | mid-range: %d | premium: %d",
        (final_labels == 0).sum(),
        (final_labels == 1).sum(),
        (final_labels == 2).sum(),
    )

    # ── Persist ───────────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(LABELED_CSV), exist_ok=True)
    df_clean.to_csv(LABELED_CSV, index=False)
    logger.info("Labeled dataset saved to %s.", LABELED_CSV)

    _save_cluster_plot(df_clean, X_clean)

    print(
        f"[clustering] ✓  {len(df_clean)} products labeled "
        f"(budget={( final_labels==0).sum()}, "
        f"mid={( final_labels==1).sum()}, "
        f"premium={(final_labels==2).sum()})"
    )
    return df_clean


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = run_clustering()
    print(df["cluster_label"].value_counts())
