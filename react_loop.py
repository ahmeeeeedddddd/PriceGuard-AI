"""
react_loop.py — Full ReAct Inner Loop (Observe → Reason → Act → Observe)
=========================================================================
Implements the complete 7-path decision engine:

  Path 1  score ≥ 0.75 AND cosine ≥ 0.80  →  Direct dispatch
  Path 2  score ≥ 0.75 AND cosine < 0.80  →  SHAP re-check loop (≤2 retries)
  Path 3  0.50 ≤ score < 0.75             →  RAG retrieval → majority vote
  Path 4  0.30 ≤ score < 0.50             →  Low confidence → human review
  Path 5  RAG tie                          →  ARIMA breaks tie
  Path 6  RAG zero-success                →  SHAP routing
  Path 7  score < 0.30                    →  Below minimum → human review

Action generation is NOT a lookup table.  The action dict (type, timing,
intensity, personalisation) is computed dynamically at inference time from
the live score, dominant SHAP feature, RAG vote outcome, and ARIMA trend.

Public API
----------
run_react_loop(product_dict) → dict
"""

import os
import logging
import warnings
import time
import copy
import numpy as np

warnings.simplefilter("ignore", FutureWarning)
logger = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────────────────
SCORE_DIRECT    = 0.75
SCORE_BORDERLINE= 0.50
SCORE_LOW       = 0.30
COSINE_RELIABLE = 0.80
MAX_SHAP_RETRY  = 2

FEATURES        = ["price", "rating", "review_count", "discount_percentage", "stock_status"]
LABEL_NAMES     = {0: "budget", 1: "mid-range", 2: "premium"}


# ── Lazy imports — these modules exist in the same package ────────────────────
def _classify(product_dict):
    from classifier import classify_product
    return classify_product(product_dict)

def _explain(product_dict, confidence):
    from explainability import explain_product
    return explain_product(product_dict, confidence)

def _forecast(category):
    from forecasting import run_forecasting
    try:
        return run_forecasting(category=category)
    except Exception:
        return {"forecast_7d": [0]*7, "trend": "stable"}

def _score(product_dict, label, confidence, shap_vals, trend):
    from scoring import compute_score
    return compute_score(product_dict, label, confidence, shap_vals, trend)

def _rag_query(feature_vec, shap_vec):
    from rag_engine import query_rag
    return query_rag(feature_vec, shap_vec)

def _rag_adjusted(rag_result, trend):
    from rag_engine import get_arima_adjusted_vote
    return get_arima_adjusted_vote(rag_result, trend)

def _shap_cosine(shap_vals, cluster_id):
    from rag_engine import get_shap_cosine
    return get_shap_cosine(shap_vals, cluster_id)

def _fire(product_data, shap_score, forecast_data):
    from actions import fire_alert
    return fire_alert(product_data, shap_score, forecast_data)


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic action generator
# ─────────────────────────────────────────────────────────────────────────────

def _generate_action(
    score: float,
    label: str,
    dominant_shap: str,
    rag_outcome: str,
    forecast_trend: str,
    path: int,
    product_dict: dict,
) -> dict:
    """
    Dynamically compute the action rather than looking it up.
    Action type, timing, intensity and personalisation all emerge from the
    live signals at this specific moment.
    """
    price     = float(product_dict.get("price", 0))
    ref_price = float(product_dict.get("reference_price", price * 1.1))
    gap_pct   = max((ref_price - price) / max(ref_price, 1e-6), 0) * 100

    # ── Action type ───────────────────────────────────────────────────────────
    if path in (1, 2) and label == "budget":
        action_type = "price_alert"
    elif path == 3 and rag_outcome in ("majority", "rag_dispatch"):
        action_type = "repricing_recommendation"
    elif path in (4, 7):
        action_type = "human_review_request"
    elif path == 5:
        action_type = "arima_arbitrated_alert"
    elif path == 6:
        action_type = "shap_driven_alert"
    else:
        action_type = "monitor"

    # ── Timing: urgency from score ────────────────────────────────────────────
    if score >= 0.80:
        timing = "immediate"       # < 5 min
    elif score >= 0.60:
        timing = "within_1_hour"
    elif score >= 0.40:
        timing = "within_24_hours"
    else:
        timing = "scheduled_review"

    # ── Intensity: driven by price gap + ARIMA trend ──────────────────────────
    base_intensity = score * 10   # 0 → 10 scale
    if forecast_trend == "rising":
        base_intensity = min(base_intensity * 1.2, 10)
    elif forecast_trend == "falling":
        base_intensity = base_intensity * 0.7
    intensity = round(base_intensity, 1)

    # ── Personalisation: anchored to dominant SHAP feature ───────────────────
    shap_messages = {
        "price":               f"Price is the primary driver (gap={gap_pct:.1f}%). Adjust pricing urgently.",
        "rating":              "Customer rating is differentiating this product. Improve review handling.",
        "review_count":        "Review velocity is influencing classification. Consider review incentives.",
        "discount_percentage": f"Discount depth ({product_dict.get('discount_percentage',0):.1f}%) is key signal. Calibrate markdown.",
        "stock_status":        "Stock availability is signalling urgency. Verify inventory levels.",
    }
    personalisation = shap_messages.get(dominant_shap, f"Dominant feature: {dominant_shap}.")

    return {
        "action_type":       action_type,
        "timing":            timing,
        "intensity":         intensity,
        "personalisation":   personalisation,
        "path_triggered":    path,
        "score_at_action":   round(score, 4),
        "trend_at_action":   forecast_trend,
        "rag_outcome":       rag_outcome,
    }


# ─────────────────────────────────────────────────────────────────────────────
# ReAct loop
# ─────────────────────────────────────────────────────────────────────────────

def run_react_loop(product_dict: dict) -> dict:
    """
    Execute the full Observe → Reason → Act → Observe loop for one product.

    Parameters
    ----------
    product_dict : dict
        Must contain: price, rating, review_count, discount_percentage,
                      stock_status, category.
        Optional:     reference_price (defaults to price × 1.1).

    Returns
    -------
    dict with full reasoning trace + final action + alert result.
    """
    ts_start = time.time()
    trace    = []
    category = product_dict.get("category", "All")

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVE 1 — Classify
    # ─────────────────────────────────────────────────────────────────────────
    trace.append({"step": "OBSERVE-1", "action": "classify_product"})
    label, confidence = _classify(product_dict)
    cluster_id = {"budget": 0, "mid-range": 1, "premium": 2}.get(label, 0)
    trace.append({"step": "OBSERVE-1", "result": f"label={label}, conf={confidence:.3f}"})

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVE 2 — SHAP explanation
    # ─────────────────────────────────────────────────────────────────────────
    trace.append({"step": "OBSERVE-2", "action": "explain_product"})
    shap_score, shap_vals = 0.0, {}
    dominant_shap         = "price"
    try:
        shap_score, shap_vals = _explain(product_dict, confidence)
        if shap_vals:
            dominant_shap = max(shap_vals, key=lambda k: abs(shap_vals[k]))
    except Exception as exc:
        logger.warning("SHAP failed in ReAct loop: %s", exc)
    trace.append({"step": "OBSERVE-2", "result": f"shap_score={shap_score:.4f}, dominant={dominant_shap}"})

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVE 3 — SARIMA forecast
    # ─────────────────────────────────────────────────────────────────────────
    trace.append({"step": "OBSERVE-3", "action": "run_forecasting"})
    forecast_data  = _forecast(category)
    forecast_trend = forecast_data.get("trend", "stable")
    forecast_7d    = forecast_data.get("forecast_7d", [])
    d7_val = f"{forecast_7d[-1]:.2f}" if forecast_7d else "N/A"
    trace.append({"step": "OBSERVE-3", "result": f"trend={forecast_trend}, day7={d7_val}"})

    # ─────────────────────────────────────────────────────────────────────────
    # REASON 1 — Intelligent Score
    # ─────────────────────────────────────────────────────────────────────────
    trace.append({"step": "REASON-1", "action": "compute_score"})
    score_result = _score(product_dict, label, confidence, shap_vals, forecast_trend)
    score        = score_result["score"]
    shap_cosine  = score_result["shap_cosine"]
    trace.append({"step": "REASON-1", "result": score_result["breakdown"]})

    # ─────────────────────────────────────────────────────────────────────────
    # REASON 2 — Route through 7 paths
    # ─────────────────────────────────────────────────────────────────────────
    import joblib
    SCALER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "scaler.pkl")

    def _get_scaled_feature_vec():
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
            x = np.array([[
                float(product_dict.get("price",               0)),
                float(product_dict.get("rating",              3)),
                float(product_dict.get("review_count",        0)),
                float(product_dict.get("discount_percentage", 0)),
                float(product_dict.get("stock_status",        1)),
            ]])
            return scaler.transform(x)[0]
        return np.zeros(5, dtype=np.float32)

    shap_vec   = np.array([shap_vals.get(f, 0.0) for f in FEATURES], dtype=np.float32)
    path_taken = None
    rag_result = {"outcome": "not_queried", "vote_label": None, "vote_fraction": 0.0}
    final_label = label  # may be overridden by RAG

    # ── Path 7 — score below minimum ─────────────────────────────────────────
    if score < SCORE_LOW:
        path_taken = 7
        route_reason = f"score={score:.4f} < min_threshold={SCORE_LOW} → human review"
        trace.append({"step": "REASON-2", "path": 7, "reason": route_reason})

    # ── Path 1 or 2 — high score ──────────────────────────────────────────────
    elif score >= SCORE_DIRECT:
        if shap_cosine >= COSINE_RELIABLE:
            path_taken   = 1
            route_reason = f"score={score:.4f}≥{SCORE_DIRECT}, cosine={shap_cosine:.4f}≥{COSINE_RELIABLE} → direct dispatch"
            trace.append({"step": "REASON-2", "path": 1, "reason": route_reason})
        else:
            # Path 2 — SHAP re-check loop
            retry_count = 0
            while retry_count < MAX_SHAP_RETRY and shap_cosine < COSINE_RELIABLE:
                retry_count += 1
                trace.append({"step": "REASON-2", "action": f"SHAP_retry_{retry_count}", "cosine": shap_cosine})
                try:
                    shap_score, shap_vals = _explain(product_dict, confidence)
                    shap_cosine = _shap_cosine(shap_vals, cluster_id)
                    dominant_shap = max(shap_vals, key=lambda k: abs(shap_vals[k])) if shap_vals else "price"
                except Exception:
                    break
            if shap_cosine >= COSINE_RELIABLE:
                path_taken   = 2
                route_reason = f"SHAP re-check passed after {retry_count} retries, cosine={shap_cosine:.4f}"
            else:
                path_taken   = 4
                route_reason = f"SHAP re-check failed after {MAX_SHAP_RETRY} retries → human review"
            trace.append({"step": "REASON-2", "path": path_taken, "reason": route_reason})

    # ── Path 3 — borderline → RAG ─────────────────────────────────────────────
    elif score >= SCORE_BORDERLINE:
        trace.append({"step": "REASON-2", "action": "rag_query", "score": score})
        feat_vec   = _get_scaled_feature_vec()
        rag_result = _rag_query(feat_vec, shap_vec)

        if rag_result["outcome"] == "zero":
            # Path 6 — zero results
            path_taken   = 6
            route_reason = "RAG returned 0 results → SHAP routing"
        elif rag_result["outcome"] == "tie":
            # Path 5 — tie, ARIMA breaks it
            adj = _rag_adjusted(rag_result, forecast_trend)
            path_taken   = 5
            rag_result   = adj
            final_route  = adj["final_route"]
            route_reason = f"RAG tie → ARIMA={forecast_trend} arbitrates → {final_route}"
        else:
            # Full RAG result with ARIMA adjustment
            adj = _rag_adjusted(rag_result, forecast_trend)
            rag_result = adj
            if adj["final_route"] == "human_review":
                path_taken   = 4
                route_reason = f"RAG vote dampened by ARIMA({forecast_trend}) → human review"
            else:
                path_taken   = 3
                route_reason = f"RAG majority vote={adj['vote_label']} ({adj['adjusted_fraction']:.2f} adj) → {adj['final_route']}"
                if adj["vote_label"] is not None:
                    final_label = LABEL_NAMES.get(int(adj["vote_label"]), label)

        trace.append({"step": "REASON-2", "path": path_taken, "reason": route_reason})

    # ── Path 4 — low confidence ───────────────────────────────────────────────
    else:
        path_taken   = 4
        route_reason = f"score={score:.4f} in [{SCORE_LOW},{SCORE_BORDERLINE}) → low confidence → human review"
        trace.append({"step": "REASON-2", "path": path_taken, "reason": route_reason})

    # ─────────────────────────────────────────────────────────────────────────
    # ACT — Generate action dynamically
    # ─────────────────────────────────────────────────────────────────────────
    trace.append({"step": "ACT", "action": "generate_action"})
    action = _generate_action(
        score=score,
        label=final_label,
        dominant_shap=dominant_shap,
        rag_outcome=rag_result.get("outcome", "not_queried"),
        forecast_trend=forecast_trend,
        path=path_taken,
        product_dict=product_dict,
    )
    trace.append({"step": "ACT", "result": action})

    # ─────────────────────────────────────────────────────────────────────────
    # ACT — Fire alert if warranted
    # ─────────────────────────────────────────────────────────────────────────
    alert_result = {}
    ref_price    = float(product_dict.get("reference_price", float(product_dict.get("price", 0)) * 1.1))
    price        = float(product_dict.get("price", 0))
    threshold    = float(os.getenv("PRICE_ALERT_THRESHOLD", "0.10"))

    if action["action_type"] in ("price_alert", "arima_arbitrated_alert", "shap_driven_alert"):
        if price < ref_price * (1 - threshold):
            trace.append({"step": "ACT", "action": "fire_alert"})
            try:
                enriched = {
                    **product_dict,
                    "cluster_label":    final_label,
                    "top_shap_feature": dominant_shap,
                    "reference_price":  ref_price,
                }
                alert_result = _fire(enriched, shap_score, forecast_data)
            except Exception as exc:
                logger.error("fire_alert failed in ReAct loop: %s", exc)
                alert_result = {"email_status": False, "slack_status": False}
            trace.append({"step": "ACT", "result": f"email={alert_result.get('email_status')}, slack={alert_result.get('slack_status')}"})

    # ─────────────────────────────────────────────────────────────────────────
    # OBSERVE (final) — Record outcome
    # ─────────────────────────────────────────────────────────────────────────
    elapsed = round(time.time() - ts_start, 3)
    trace.append({
        "step":    "OBSERVE-FINAL",
        "result":  f"path={path_taken}, score={score:.4f}, label={final_label}, "
                   f"action={action['action_type']}, elapsed={elapsed}s",
    })

    result = {
        "product_name":    product_dict.get("product_name", "Unknown"),
        "price":           price,
        "category":        category,
        "label":           final_label,
        "confidence":      confidence,
        "score":           score,
        "score_breakdown": score_result,
        "shap_score":      shap_score,
        "shap_cosine":     shap_cosine,
        "dominant_shap":   dominant_shap,
        "shap_vals":       shap_vals,
        "forecast_trend":  forecast_trend,
        "forecast_7d":     forecast_7d,
        "rag_result":      rag_result,
        "path_taken":      path_taken,
        "route_reason":    route_reason,
        "action":          action,
        "alert_result":    alert_result,
        "trace":           trace,
        "elapsed_s":       elapsed,
    }

    logger.info(
        "ReAct complete: product='%s' path=%d score=%.4f label=%s action=%s elapsed=%.3fs",
        product_dict.get("product_name", "?"), path_taken, score,
        final_label, action["action_type"], elapsed,
    )
    return result
