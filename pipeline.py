"""
pipeline.py — PriceGuard AI Orchestrator
=========================================
The single entry-point that implements the 5-step Sense-Reason-Act pipeline:

  Run 1 (no model files):
    Step 1A → data_loader   → clean Kaggle CSV  → raw_products.csv
    Step 2  → clustering    → DBSCAN + FCM       → labeled_products.csv
    Step 3  → classifier    → SVM + GridSearch   → svm_model.pkl / scaler.pkl

  Run 2+ (models exist):
    Step 1B → scraper       → live Noon.com data → live_products.csv
    Step 3  → classify_product() per product
    Step 4A → explain_product() per product
    Step 4B → run_forecasting() per category
    Step 5  → fire_alert() if underpriced

Usage
-----
    python pipeline.py
"""

import os
import sys
import logging
from dotenv import load_dotenv

load_dotenv()

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


# ─────────────────────────────────────────────────────────────────────────────

def models_exist() -> bool:
    """Return True if the trained model files are already on disk."""
    return os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)


# ── TRAINING BRANCH (first run) ───────────────────────────────────────────────

def run_training_pipeline() -> None:
    """
    Step 1A → Step 2 → Step 3 (train branch).
    Called only when svm_model.pkl does not yet exist.
    """
    print("\n" + "="*60)
    print("  PriceGuard AI — TRAINING MODE (first run)")
    print("="*60)

    # Step 1A – data ingestion
    print("\n[Step 1A] Loading & cleaning Kaggle dataset …")
    from data_loader import run_data_loader
    run_data_loader()

    # Step 2 – clustering
    print("\n[Step 2] Clustering products (DBSCAN + Fuzzy C-Means) …")
    from clustering import run_clustering
    run_clustering()

    # Step 3 – supervised classification
    print("\n[Step 3] Training SVM classifier (GridSearchCV) …")
    from classifier import train_classifier
    svm, scaler, report = train_classifier()

    # Step 3.5 – Offline Cycle (SHAP Means & RAG Index)
    print("\n[Step 3.5] Building RAG Index & SHAP Baseline (Offline Cycle) …")
    from explainability import run_shap_update
    from rag_engine     import build_rag_index
    
    # 1. Load labeled data
    import pandas as pd
    labeled_csv = os.path.join(BASE_DIR, "data", "labeled_products.csv")
    df_labeled  = pd.read_csv(labeled_csv)
    
    # 2. Compute SHAP cluster means (implicit RAG)
    shap_means = run_shap_update(df_labeled)
    
    # 3. Build FAISS index (explicit RAG)
    build_rag_index(df_labeled, shap_means)

    print("\n" + "="*60)
    print("  Training complete!  Re-run pipeline.py for inference.")
    print("="*60)


# ── INFERENCE BRANCH (subsequent runs) ────────────────────────────────────────

def run_inference_pipeline() -> list[dict]:
    """
    Step 1B → Step 3 → Step 4A → Step 4B → Step 5 (inference branch).
    Returns a list of result dicts (one per live product).
    """
    print("\n" + "="*60)
    print("  PriceGuard AI — INFERENCE MODE")
    print("="*60)

    # Step 1B – live scraping
    print("\n[Step 1B] Scraping live products …")
    from scraper import run_scraper
    live_products = run_scraper()

    if not live_products:
        print("  No live products found. Exiting inference.")
        return []

    # Reference price: use mean price from training data as baseline
    import pandas as pd
    import numpy as np
    raw_csv = os.path.join(BASE_DIR, "data", "raw_products.csv")
    if os.path.exists(raw_csv):
        df_train = pd.read_csv(raw_csv)
        reference_price = float(df_train["price"].median())
    else:
        reference_price = 50.0

    from react_loop    import run_react_loop
    from drift_monitor import check_drift, trigger_outer_loop

    print(f"\n[ReAct Loop] Processing {len(live_products)} live products …\n")

    results = []
    for product in live_products:
        # Run the full 5-step / 7-path ReAct Inner Loop
        try:
            res = run_react_loop(product)
            results.append(res)
            
            # Print brief summary
            print(f"  ✓  {product.get('product_name','?')[:40]:<40} | "
                  f"Score={res['score']:.3f} | Path={res['path_taken']} | "
                  f"Action={res['action']['action_type']}")
        except Exception as exc:
            logger.error("ReAct loop failed for '%s': %s", product.get('product_name','?'), exc)

    # ── Drift Detection (End of Run — Outer Loop Check) ──────────────────────
    if results:
        print("\n[Drift Check] Monitoring system integrity …")
        last_res = results[-1]
        drift_report = check_drift(last_res["confidence"], live_products[-1], last_res["label"])
        
        if drift_report["drift_triggered"]:
            print("  🚨 DATA DRIFT DETECTED! Triggering Outer ReAct Loop (Autonomous Retraining)...")
            trigger_outer_loop()
        else:
            print(f"  ✓  System Stable (EMA Conf: {drift_report['signal_1']['ema_confidence']:.2f})")

    print(f"\n[pipeline] ✓  Inference complete — {len(results)} products processed.")
    return results


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """Route to training or inference based on model file existence."""
    if not models_exist():
        run_training_pipeline()
    else:
        run_inference_pipeline()


if __name__ == "__main__":
    main()
