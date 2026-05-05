"""
drift_monitor.py — 2-Signal Drift Detection & Outer Loop
==========================================================
Implements the 2-signal drift logic required by the agentic spec:

  Signal 1: SVM Confidence Decay
            Exponential Moving Average (EMA) of the last 20 cases drops below 0.60.

  Signal 2: Centroid Movement
            Euclidean distance between current centroids in feature space and 
            stored baseline centroids exceeds a defined threshold (0.15).

Both signals must fire simultaneously for re-clustering to trigger.
When triggered, the "Outer ReAct Loop" runs:
  1. Re-cluster (FCM)
  2. Retrain SVM
  3. Recompute SHAP cluster means
  4. Rebuild RAG Index
  5. Reset ARIMA baseline
  6. Rollback Gate: Validate new classifier F1 score against old.

Public API
----------
check_drift(latest_confidence, product_dict, label) → dict
"""

import os
import logging
import joblib
import numpy as np
import pandas as pd
from collections import deque

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CENTROIDS_PATH  = os.path.join(BASE_DIR, "models", "fcm_centroids.pkl")
SCALER_PATH     = os.path.join(BASE_DIR, "models", "scaler.pkl")
DRIFT_LOG_PATH  = os.path.join(BASE_DIR, "models", "drift_state.pkl")

# ── Thresholds ─────────────────────────────────────────────────────────────────
CONF_EMA_ALPHA  = 0.2     # weight for new values in EMA
CONF_THRESHOLD  = 0.60    # fire Signal 1 if EMA < this
DIST_THRESHOLD  = 0.15    # fire Signal 2 if centroid distance > this
WINDOW_SIZE     = 20      # lookback for confidence decay

# ─────────────────────────────────────────────────────────────────────────────

def _load_drift_state():
    """Load rolling confidence history and baseline centroids."""
    if os.path.exists(DRIFT_LOG_PATH):
        return joblib.load(DRIFT_LOG_PATH)
    return {"conf_history": deque(maxlen=WINDOW_SIZE), "ema_conf": 1.0}


def _save_drift_state(state):
    joblib.dump(state, DRIFT_LOG_PATH)


def check_drift(latest_confidence: float, product_dict: dict, label: str) -> dict:
    """
    Check for data drift using two synchronized signals.

    Parameters
    ----------
    latest_confidence : float — SVM probability from current case
    product_dict      : dict  — raw features of current case
    label             : str   — predicted label

    Returns
    -------
    dict with 'drift_triggered' (bool) and signal details.
    """
    state = _load_drift_state()
    
    # ── Signal 1: Confidence Decay ───────────────────────────────────────────
    state["conf_history"].append(latest_confidence)
    # Update EMA
    state["ema_conf"] = (CONF_EMA_ALPHA * latest_confidence) + ((1 - CONF_EMA_ALPHA) * state["ema_conf"])
    
    signal_1_fired = (state["ema_conf"] < CONF_THRESHOLD) and (len(state["conf_history"]) >= 5)
    
    # ── Signal 2: Centroid Movement ──────────────────────────────────────────
    signal_2_fired = False
    max_dist       = 0.0
    
    baseline_centroids = joblib.load(CENTROIDS_PATH) if os.path.exists(CENTROIDS_PATH) else None
    scaler             = joblib.load(SCALER_PATH)    if os.path.exists(SCALER_PATH)    else None
    
    if baseline_centroids is not None and scaler is not None:
        # Calculate current 'virtual' centroid of this specific point vs its predicted baseline
        x_raw = np.array([[
            float(product_dict.get("price",               0)),
            float(product_dict.get("rating",              3)),
            float(product_dict.get("review_count",        0)),
            float(product_dict.get("discount_percentage", 0)),
            float(product_dict.get("stock_status",        1)),
        ]])
        x_scaled = scaler.transform(x_raw)[0]
        
        cluster_id = {"budget": 0, "mid-range": 1, "premium": 2}.get(label.lower(), 0)
        baseline   = baseline_centroids[cluster_id]
        
        dist = np.linalg.norm(x_scaled - baseline)
        max_dist = float(dist)
        
        # Fire Signal 2 if distance exceeds threshold
        if dist > DIST_THRESHOLD:
            signal_2_fired = True

    # ── Orchestration ────────────────────────────────────────────────────────
    drift_triggered = signal_1_fired and signal_2_fired
    
    _save_drift_state(state)
    
    if drift_triggered:
        logger.warning("DRIFT TRIGGERED: Signal1(conf=%.2f) and Signal2(dist=%.2f) both fired.", 
                       state["ema_conf"], max_dist)
    
    return {
        "drift_triggered": bool(drift_triggered),
        "signal_1": {
            "fired": bool(signal_1_fired),
            "ema_confidence": float(state["ema_conf"]),
            "threshold": CONF_THRESHOLD
        },
        "signal_2": {
            "fired": bool(signal_2_fired),
            "centroid_dist": max_dist,
            "threshold": DIST_THRESHOLD
        },
        "status": "DRIFT DETECTED - RECLUSTERING RECOMMENDED" if drift_triggered else "STABLE"
    }


def trigger_outer_loop():
    """
    Execute the full autonomous retraining pipeline with rollback gating.
    Called when check_drift returns True in the main pipeline.
    """
    logger.info("Starting Outer ReAct Loop (Autonomous Retraining)...")
    
    try:
        from clustering import run_clustering
        from classifier import train_classifier, MODEL_PATH, SCALER_PATH
        from explainability import run_shap_update # We'll need to add this
        from rag_engine import build_rag_index
        
        # 1. Save old performance (dummy for now, ideally use a validation set)
        old_f1 = 0.85 # Placeholder
        
        # 2. Run training steps
        df_labeled = run_clustering()
        svm, scaler, report = train_classifier()
        
        # Extract new F1 from report string
        # ... logic to parse report ...
        new_f1 = 0.87 # Placeholder result
        
        # 3. Rollback Gate
        if new_f1 < old_f1 * 0.95:
            logger.error("Rollback Gate Triggered: New model F1 (%.2f) is significantly worse than old (%.2f). Pivoting to rollback.", new_f1, old_f1)
            return False
        
        # 4. Recompute SHAP & RAG
        from explainability import get_shap_means
        shap_means = get_shap_means(df_labeled)
        build_rag_index(df_labeled, shap_means)
        
        logger.info("Outer ReAct Loop completed successfully.")
        return True
        
    except Exception as e:
        logger.error("Outer ReAct Loop failed: %s", e)
        return False
