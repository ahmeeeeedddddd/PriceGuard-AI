"""
explainability.py — Step 4A: SHAP Explainable AI
=================================================
After SVM classification, uses shap.KernelExplainer to compute SHAP
values, generates a bar chart of feature importances, and exposes
explain_product() as the public inference function.

Public API
----------
explain_product(product_dict, confidence_score) → (shap_score, shap_values_dict)
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

warnings.simplefilter("ignore", FutureWarning)
warnings.simplefilter("ignore", UserWarning)

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "models", "svm_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
LABELED_CSV     = os.path.join(BASE_DIR, "data", "labeled_products.csv")
SHAP_PLOT       = os.path.join(BASE_DIR, "outputs", "shap_plot.png")
SHAP_MEANS_PATH = os.path.join(BASE_DIR, "models", "shap_cluster_means.pkl")

FEATURES    = ["price", "rating", "review_count", "discount_percentage", "stock_status"]
RANDOM_SEED = 42

# Number of background samples for KernelExplainer (trade-off speed vs accuracy)
N_BACKGROUND = 50


# ─────────────────────────────────────────────────────────────────────────────

def _load_models():
    """Load SVM model and scaler from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise RuntimeError("Models not found. Run pipeline.py to train first.")
    return joblib.load(MODEL_PATH), joblib.load(SCALER_PATH)


def _get_background_data(scaler) -> np.ndarray:
    """
    Sample background data from labeled_products.csv for KernelExplainer.
    Falls back to a zero matrix if the CSV is not found.
    """
    if os.path.exists(LABELED_CSV):
        df = pd.read_csv(LABELED_CSV)
        X  = df[FEATURES].fillna(0).values
        rng = np.random.default_rng(RANDOM_SEED)
        idx = rng.choice(len(X), size=min(N_BACKGROUND, len(X)), replace=False)
        return scaler.transform(X[idx])
    else:
        logger.warning("labeled_products.csv not found — using zero background.")
        return np.zeros((N_BACKGROUND, len(FEATURES)))


def _save_shap_bar_chart(shap_vals: np.ndarray,
                         feature_names: list[str]) -> None:
    """
    Save a horizontal bar chart of mean absolute SHAP values.
    """
    os.makedirs(os.path.dirname(SHAP_PLOT), exist_ok=True)
    means  = np.abs(shap_vals).mean(axis=0) if shap_vals.ndim == 2 else np.abs(shap_vals)
    sorted_idx = np.argsort(means)

    fig, ax = plt.subplots(figsize=(8, 4))
    colours = ["#2ECC71" if m > means.mean() else "#E74C3C" for m in means[sorted_idx]]
    ax.barh([feature_names[i] for i in sorted_idx], means[sorted_idx], color=colours)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("PriceGuard AI — Feature Importance (SHAP)")
    fig.tight_layout()
    fig.savefig(SHAP_PLOT, dpi=150)
    plt.close(fig)
    logger.info("SHAP bar chart saved to %s.", SHAP_PLOT)


def explain_product(product_dict: dict, confidence_score: float) -> tuple[float, dict]:
    """
    Compute SHAP values for a single product and save the bar-chart.

    Parameters
    ----------
    product_dict : dict
        Keys: price, rating, review_count, discount_percentage, stock_status.
    confidence_score : float
        SVM max-class probability (from classify_product).

    Returns
    -------
    shap_score : float
        confidence_score × max(|shap_values|)  — composite importance score.
    shap_values_dict : dict
        {feature_name: shap_value} for the individual sample.
    """
    svm, scaler = _load_models()

    x_raw = np.array([[
        float(product_dict.get("price",               0)),
        float(product_dict.get("rating",              3)),
        float(product_dict.get("review_count",        0)),
        float(product_dict.get("discount_percentage", 0)),
        float(product_dict.get("stock_status",        1)),
    ]])
    x_scaled = scaler.transform(x_raw)

    background = _get_background_data(scaler)

    # Wrapper: KernelExplainer needs a function that returns probabilities
    def predict_proba(X):
        return svm.predict_proba(X)

    explainer = shap.KernelExplainer(predict_proba, background)
    # shap_values is typically a list of arrays (one per class) for multi-class.
    # We want features for the predicted class.
    shap_results = explainer.shap_values(x_scaled, nsamples=100)
    
    if isinstance(shap_results, list):
        shap_values = shap_results
    else:
        # If it's a single array (N, F, C) or (N, F), wrap it or handle it
        if shap_results.ndim == 3:
            # (N, F, C) -> list of C arrays of (N, F)
            shap_values = [shap_results[:, :, i] for i in range(shap_results.shape[2])]
        else:
            # (N, F) -> wrap in list
            shap_values = [shap_results]

    # Pick the class with highest probability
    pred_class  = int(svm.predict(x_scaled)[0])
    sv_for_pred = shap_values[pred_class][0]   # shape (n_features,)

    # Save bar chart using all-class mean for a global view
    all_sv = np.array([sv[0] for sv in shap_values])   # (n_classes, n_features)
    _save_shap_bar_chart(all_sv, FEATURES)

    # Composite score
    max_abs_shap = float(np.max(np.abs(sv_for_pred)))
    shap_score   = confidence_score * max_abs_shap

    shap_values_dict = {f: float(v) for f, v in zip(FEATURES, sv_for_pred)}
    top_feature = max(shap_values_dict, key=lambda k: abs(shap_values_dict[k]))

    logger.info(
        "SHAP explained — top feature: %s (%.4f), shap_score=%.4f",
        top_feature, shap_values_dict[top_feature], shap_score,
    )
    return shap_score, shap_values_dict


def get_shap_means(df_labeled: pd.DataFrame) -> dict:
    """
    Offline cycle: After training, compute the mean SHAP vector for each cluster.
    This serves as the 'Implicit RAG' document store.
    """
    logger.info("Computing SHAP cluster means for %d samples...", len(df_labeled))
    svm, scaler = _load_models()
    
    # 1. Sample up to 100 per cluster to avoid huge computation
    sampled_df = df_labeled.sample(frac=1, random_state=RANDOM_SEED).groupby("cluster_label").head(100).reset_index(drop=True)
    
    X_raw = sampled_df[FEATURES].fillna(0).values
    X_scaled = scaler.transform(X_raw)
    
    background = _get_background_data(scaler)
    explainer = shap.KernelExplainer(svm.predict_proba, background)
    
    # Wait for computation
    shap_results = explainer.shap_values(X_scaled, nsamples=100)
    
    # Standardise to list of arrays (one per class)
    if isinstance(shap_results, list):
        shap_values = shap_results
    elif shap_results.ndim == 3:
        # (N, F, C) -> switch to (C, N, F) list 
        shap_values = [shap_results[:, :, i] for i in range(shap_results.shape[2])]
    else:
        # Single class case
        shap_values = [shap_results]
    
    cluster_means = {}
    for cluster_id in range(3):
        mask = (sampled_df["cluster_label"] == cluster_id).values
        if mask.any() and cluster_id < len(shap_values):
            # relevant_sv has shape (n_in_cluster, n_features)
            relevant_sv = shap_values[cluster_id][mask]
            cluster_means[cluster_id] = relevant_sv.mean(axis=0).tolist()
        else:
            cluster_means[cluster_id] = [0.0] * 5
            
    return cluster_means


def run_shap_update(df_labeled: pd.DataFrame):
    """Entry point for the training cycle to build SHAP means."""
    means = get_shap_means(df_labeled)
    import joblib
    joblib.dump(means, SHAP_MEANS_PATH)
    logger.info("SHAP cluster means saved to %s", SHAP_MEANS_PATH)
    return means


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "price": 29.99, "rating": 4.2, "review_count": 150,
        "discount_percentage": 10.0, "stock_status": 1,
    }
    score, vals = explain_product(sample, confidence_score=0.87)
    print(f"SHAP score: {score:.4f}")
    print(vals)
