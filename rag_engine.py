"""
rag_engine.py — Dual RAG (Explicit FAISS + Implicit Cosine)
============================================================
Implements both RAG forms required by the agentic spec:

Implicit RAG
  Cosine similarity of the live SHAP vector against the stored
  per-cluster mean SHAP vectors (the cluster-mean IS the document store).

Explicit RAG
  FAISS flat-L2 index on concatenated [scaled_features | shap_values] (10-D).
  Built once at training time, queried only for borderline scores.
  Returns majority vote on retrieved cluster labels + vote fraction.

ARIMA–RAG Interaction
  When ARIMA trend is "rising" → RAG vote is fully trusted.
  When ARIMA trend is "falling" → confidence is dampened (×0.5 modifier),
  overriding the RAG vote toward human review if dampened conf < threshold.

Public API
----------
build_rag_index(df_labeled, shap_cluster_means)
query_rag(feature_vec, shap_vec, k=5) → dict
get_shap_cosine(live_shap_vals, cluster_id) → float
get_arima_adjusted_vote(rag_result, forecast_trend) → dict
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.simplefilter("ignore", FutureWarning)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH     = os.path.join(BASE_DIR, "models", "scaler.pkl")
SHAP_MEANS_PATH = os.path.join(BASE_DIR, "models", "shap_cluster_means.pkl")
RAG_INDEX_PATH  = os.path.join(BASE_DIR, "models", "rag_index.faiss")
RAG_LABELS_PATH = os.path.join(BASE_DIR, "models", "rag_labels.pkl")
RAG_VECS_PATH   = os.path.join(BASE_DIR, "models", "rag_vectors.pkl")

FEATURES        = ["price", "rating", "review_count", "discount_percentage", "stock_status"]
K_RETRIEVE      = 5      # neighbours to fetch
MAJORITY_THRESH = 0.60   # fraction needed to trust RAG vote

# ── ARIMA dampening ────────────────────────────────────────────────────────────
ARIMA_TRUST = {"rising": 1.0, "stable": 0.85, "falling": 0.50}


# ─────────────────────────────────────────────────────────────────────────────
# Build (training-time)
# ─────────────────────────────────────────────────────────────────────────────

def build_rag_index(df_labeled: pd.DataFrame, shap_cluster_means: dict) -> None:
    """
    Build the Explicit FAISS flat-L2 index from all labeled samples.
    Each vector = [scaled_features(5) | cluster_mean_shap(5)] → 10-D.

    Also saves shap_cluster_means to disk for implicit RAG.

    Parameters
    ----------
    df_labeled : pd.DataFrame — must have FEATURES columns + 'cluster_label'
    shap_cluster_means : dict — {cluster_id: np.ndarray of shape (5,)}
    """
    try:
        import faiss
    except ImportError:
        logger.warning("faiss-cpu not installed — RAG index will use KNN fallback.")
        _build_knn_index(df_labeled, shap_cluster_means)
        return

    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)

    # Save SHAP cluster means (implicit RAG store)
    joblib.dump(shap_cluster_means, SHAP_MEANS_PATH)
    logger.info("Saved SHAP cluster means to %s.", SHAP_MEANS_PATH)

    scaler = joblib.load(SCALER_PATH)
    df     = df_labeled.dropna(subset=FEATURES + ["cluster_label"]).copy()
    X_feat = scaler.transform(df[FEATURES].values.astype(np.float32))
    labels = df["cluster_label"].astype(int).values

    # Build SHAP vectors per row using stored cluster means
    X_shap = np.array([
        shap_cluster_means.get(int(lbl), np.zeros(5, dtype=np.float32))
        for lbl in labels
    ], dtype=np.float32)

    # Concatenate → 10-D index vectors
    X_full = np.hstack([X_feat, X_shap]).astype(np.float32)

    dim   = X_full.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(X_full)

    faiss.write_index(index, RAG_INDEX_PATH)
    joblib.dump(labels,  RAG_LABELS_PATH)
    joblib.dump(X_full,  RAG_VECS_PATH)

    logger.info("FAISS RAG index built: %d vectors, dim=%d → %s", len(X_full), dim, RAG_INDEX_PATH)
    print(f"[rag_engine] ✓  FAISS index built ({len(X_full)} vectors, dim={dim})")


def _build_knn_index(df_labeled: pd.DataFrame, shap_cluster_means: dict) -> None:
    """KNN fallback when faiss-cpu is unavailable. Saves raw vectors for brute-force search."""
    os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)
    joblib.dump(shap_cluster_means, SHAP_MEANS_PATH)

    scaler = joblib.load(SCALER_PATH)
    df     = df_labeled.dropna(subset=FEATURES + ["cluster_label"]).copy()
    X_feat = scaler.transform(df[FEATURES].values.astype(np.float32))
    labels = df["cluster_label"].astype(int).values

    X_shap = np.array([
        shap_cluster_means.get(int(l), np.zeros(5, dtype=np.float32)) for l in labels
    ], dtype=np.float32)

    X_full = np.hstack([X_feat, X_shap]).astype(np.float32)
    joblib.dump(labels, RAG_LABELS_PATH)
    joblib.dump(X_full, RAG_VECS_PATH)
    logger.info("KNN fallback index saved (%d vectors).", len(X_full))
    print(f"[rag_engine] ✓  KNN fallback index built ({len(X_full)} vectors)")


# ─────────────────────────────────────────────────────────────────────────────
# Query (inference-time)
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def get_shap_cosine(live_shap_vals: dict, cluster_id: int) -> float:
    """
    Implicit RAG — cosine similarity of the live SHAP vector against the
    stored cluster-mean SHAP vector for `cluster_id`.

    Returns float in [0, 1] (rescaled from [-1, 1]).
    """
    shap_means = joblib.load(SHAP_MEANS_PATH) if os.path.exists(SHAP_MEANS_PATH) else None
    if shap_means is None:
        return 0.5

    live_vec = np.array([live_shap_vals.get(f, 0.0) for f in FEATURES], dtype=np.float32)
    mean_vec = np.array(shap_means.get(cluster_id, np.zeros(5)), dtype=np.float32)
    raw      = _cosine_similarity(live_vec, mean_vec)
    return float((raw + 1.0) / 2.0)


def query_rag(
    feature_vec: np.ndarray,
    shap_vec: np.ndarray,
    k: int = K_RETRIEVE,
) -> dict:
    """
    Explicit RAG — query the FAISS (or KNN) index for the k nearest neighbours
    and return a majority vote result.

    Parameters
    ----------
    feature_vec : np.ndarray shape (5,) — SCALED features
    shap_vec    : np.ndarray shape (5,) — live SHAP values
    k           : number of neighbours

    Returns
    -------
    dict:
        vote_label     : int | None — winning cluster or None if tie/empty
        vote_fraction  : float      — fraction of neighbours in winning class
        n_retrieved    : int
        outcome        : "majority" | "tie" | "zero"
        neighbour_labels : list[int]
    """
    query = np.hstack([feature_vec, shap_vec]).astype(np.float32).reshape(1, -1)

    # ── Try FAISS ─────────────────────────────────────────────────────────────
    if os.path.exists(RAG_INDEX_PATH) and os.path.exists(RAG_LABELS_PATH):
        try:
            import faiss
            index  = faiss.read_index(RAG_INDEX_PATH)
            labels = joblib.load(RAG_LABELS_PATH)
            k_     = min(k, index.ntotal)
            _, idx = index.search(query, k_)
            retrieved = labels[idx[0]].tolist()
            return _vote(retrieved)
        except Exception as exc:
            logger.warning("FAISS query failed (%s) — falling back to KNN.", exc)

    # ── KNN brute-force fallback ───────────────────────────────────────────────
    if os.path.exists(RAG_VECS_PATH) and os.path.exists(RAG_LABELS_PATH):
        X_all  = joblib.load(RAG_VECS_PATH)
        labels = joblib.load(RAG_LABELS_PATH)
        dists  = np.linalg.norm(X_all - query, axis=1)
        idx    = np.argsort(dists)[:k]
        retrieved = labels[idx].tolist()
        return _vote(retrieved)

    # ── No index available ────────────────────────────────────────────────────
    logger.warning("RAG index not available — returning zero-success.")
    return {"vote_label": None, "vote_fraction": 0.0,
            "n_retrieved": 0, "outcome": "zero", "neighbour_labels": []}


def _vote(labels: list) -> dict:
    """Compute majority vote over retrieved labels."""
    if not labels:
        return {"vote_label": None, "vote_fraction": 0.0,
                "n_retrieved": 0, "outcome": "zero", "neighbour_labels": []}

    from collections import Counter
    counts = Counter(labels)
    top1, top1_count = counts.most_common(1)[0]
    fraction = top1_count / len(labels)

    if len(counts) >= 2:
        top2_count = counts.most_common(2)[1][1]
        if top2_count == top1_count:
            return {"vote_label": None, "vote_fraction": fraction,
                    "n_retrieved": len(labels), "outcome": "tie",
                    "neighbour_labels": labels}

    outcome = "majority" if fraction >= MAJORITY_THRESH else "tie"
    return {
        "vote_label":       int(top1),
        "vote_fraction":    float(fraction),
        "n_retrieved":      len(labels),
        "outcome":          outcome,
        "neighbour_labels": labels,
    }


def get_arima_adjusted_vote(rag_result: dict, forecast_trend: str) -> dict:
    """
    Combine RAG majority vote with ARIMA trend signal.

    - Rising trend  → full trust in RAG vote (×1.0)
    - Stable trend  → moderate trust (×0.85)
    - Falling trend → dampened confidence (×0.50) — if fraction drops below
                      MAJORITY_THRESH, route toward human review instead.

    Returns
    -------
    dict with original rag_result + 'adjusted_fraction', 'arima_vote', 'final_route'
    """
    trust  = ARIMA_TRUST.get(str(forecast_trend).lower(), 0.85)
    adj_frac = rag_result.get("vote_fraction", 0.0) * trust

    if rag_result.get("outcome") == "zero":
        final_route = "shap_routing"
    elif adj_frac < MAJORITY_THRESH:
        final_route = "human_review"
    else:
        final_route = "rag_dispatch"

    arima_votes_with  = forecast_trend == "rising"
    arima_votes_against = forecast_trend == "falling"

    return {
        **rag_result,
        "adjusted_fraction":  adj_frac,
        "arima_vote":         "with" if arima_votes_with else ("against" if arima_votes_against else "neutral"),
        "trust_factor":       trust,
        "final_route":        final_route,
    }
