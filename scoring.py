"""
scoring.py — 5-Signal Intelligent Score
========================================
Computes a composite Intelligent Score from five independent signals:

  S       — class severity   (how alarming is the cluster label?)
  Gap     — SVM confidence   (how certain is the classifier?)
  μ       — FCM proximity    (how close is the point to its centroid?)
  ARIMA   — trend multiplier (what direction are prices trending?)
  SHAP    — cosine similarity (does this explanation match typical cluster?)

Formula
-------
  score = 0.25·S  +  0.20·Gap  +  0.20·μ  +  0.20·ARIMA  +  0.15·SHAP_cos

Weights sum to 1.00.  Score lives in [0, 1] where higher = more alarming.

Public API
----------
compute_score(product_dict, label, confidence, shap_vals, forecast_trend) → dict
"""

import os
import math
import logging
import warnings
import numpy as np
import joblib

warnings.simplefilter("ignore", FutureWarning)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR           = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH        = os.path.join(BASE_DIR, "models", "scaler.pkl")
CENTROIDS_PATH     = os.path.join(BASE_DIR, "models", "fcm_centroids.pkl")
SHAP_MEANS_PATH    = os.path.join(BASE_DIR, "models", "shap_cluster_means.pkl")

FEATURES           = ["price", "rating", "review_count", "discount_percentage", "stock_status"]

# ── Signal weights ─────────────────────────────────────────────────────────────
W_S      = 0.25
W_GAP    = 0.20
W_MU     = 0.20
W_ARIMA  = 0.20
W_SHAP   = 0.15

# ── Class severity mapping (0=budget is most alarming) ────────────────────────
SEVERITY = {0: 1.0, 1: 0.6, 2: 0.2, "budget": 1.0, "mid-range": 0.6, "premium": 0.2}

# ── ARIMA trend multiplier ─────────────────────────────────────────────────────
ARIMA_WEIGHT = {"rising": 1.0, "stable": 0.8, "falling": 0.5}


# ─────────────────────────────────────────────────────────────────────────────

def _load_artifact(path):
    """Load a joblib artifact, return None if missing."""
    if os.path.exists(path):
        return joblib.load(path)
    return None


def _signal_s(label) -> float:
    """S — class severity signal [0, 1]. Higher = more alarming."""
    key = label if isinstance(label, int) else str(label).lower()
    return float(SEVERITY.get(key, 0.5))


def _signal_gap(confidence: float) -> float:
    """Gap(SVM) — raw SVM max-class probability [0, 1]."""
    return float(np.clip(confidence, 0.0, 1.0))


def _signal_mu(product_dict: dict, label) -> float:
    """
    μ(predicted) — normalised geometric proximity to the predicted FCM centroid.
    Returns 1.0 (closest) → 0.0 (furthest) in scaled feature space.
    """
    centroids = _load_artifact(CENTROIDS_PATH)
    scaler    = _load_artifact(SCALER_PATH)
    if centroids is None or scaler is None:
        return 0.5   # neutral fallback

    x_raw = np.array([[
        float(product_dict.get("price",               0)),
        float(product_dict.get("rating",              3)),
        float(product_dict.get("review_count",        0)),
        float(product_dict.get("discount_percentage", 0)),
        float(product_dict.get("stock_status",        1)),
    ]])
    x_scaled = scaler.transform(x_raw)[0]

    # Map label to centroid index (centroids are ordered by mean price asc)
    cluster_id = label if isinstance(label, int) else {"budget": 0, "mid-range": 1, "premium": 2}.get(str(label).lower(), 0)
    cluster_id = int(np.clip(cluster_id, 0, len(centroids) - 1))

    centroid    = centroids[cluster_id]
    distance    = float(np.linalg.norm(x_scaled - centroid))
    # Normalise: max possible distance in 5-D unit-sphere space ≈ sqrt(5)·3 ≈ 6.7
    normalised  = 1.0 - float(np.clip(distance / 6.7, 0.0, 1.0))
    return normalised


def _signal_arima(forecast_trend: str) -> float:
    """ARIMA cluster trend multiplier. Rising = full trust, falling = dampened."""
    trend = str(forecast_trend).lower().strip()
    return ARIMA_WEIGHT.get(trend, 0.8)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0 on zero-norm."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _signal_shap(shap_vals: dict, label) -> float:
    """
    SHAP cosine similarity — compares live SHAP vector to the stored
    cluster-mean SHAP vector (the implicit RAG document).
    Returns similarity in [0, 1] (clamped from [-1, 1]).
    """
    shap_means = _load_artifact(SHAP_MEANS_PATH)
    if shap_means is None or not shap_vals:
        return 0.5   # neutral

    cluster_id = label if isinstance(label, int) else {"budget": 0, "mid-range": 1, "premium": 2}.get(str(label).lower(), 0)

    if cluster_id not in shap_means:
        return 0.5

    live_vec  = np.array([shap_vals.get(f, 0.0) for f in FEATURES])
    mean_vec  = np.array(shap_means[cluster_id])
    raw_cos   = _cosine_similarity(live_vec, mean_vec)
    # Rescale from [-1,1] → [0,1]
    return float((raw_cos + 1.0) / 2.0)


def compute_score(
    product_dict: dict,
    label,
    confidence: float,
    shap_vals: dict,
    forecast_trend: str = "stable",
) -> dict:
    """
    Compute the 5-signal Intelligent Score.

    Parameters
    ----------
    product_dict   : dict of product features
    label          : predicted cluster label (int or str)
    confidence     : SVM max-class probability
    shap_vals      : {feature: shap_value} for the live sample
    forecast_trend : "rising" | "stable" | "falling"

    Returns
    -------
    dict with keys:
        score          — float [0, 1]
        s              — severity signal
        gap            — SVM confidence signal
        mu             — centroid proximity signal
        arima          — trend multiplier signal
        shap_cosine    — SHAP cosine similarity signal
        breakdown      — human-readable explanation
    """
    s          = _signal_s(label)
    gap        = _signal_gap(confidence)
    mu         = _signal_mu(product_dict, label)
    arima      = _signal_arima(forecast_trend)
    shap_cos   = _signal_shap(shap_vals, label)

    score = (W_S * s) + (W_GAP * gap) + (W_MU * mu) + (W_ARIMA * arima) + (W_SHAP * shap_cos)
    score = float(np.clip(score, 0.0, 1.0))

    logger.info(
        "IntelligentScore=%.4f [S=%.3f Gap=%.3f μ=%.3f ARIMA=%.3f SHAP_cos=%.3f]",
        score, s, gap, mu, arima, shap_cos,
    )

    breakdown = (
        f"S={s:.3f}(×{W_S}) + Gap={gap:.3f}(×{W_GAP}) + "
        f"μ={mu:.3f}(×{W_MU}) + ARIMA={arima:.3f}(×{W_ARIMA}) + "
        f"SHAP_cos={shap_cos:.3f}(×{W_SHAP}) = {score:.4f}"
    )

    return {
        "score":       score,
        "s":           s,
        "gap":         gap,
        "mu":          mu,
        "arima":       arima,
        "shap_cosine": shap_cos,
        "breakdown":   breakdown,
    }
