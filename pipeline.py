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
import json
import logging
from datetime import datetime
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows
if sys.platform == "win32":
    import sys
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

load_dotenv()

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
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
    print("\n[Step 3] Training SVM classifier (GridSearchCV) ...")
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
    import time
    
    interval = int(os.getenv("SCRAPE_INTERVAL_SECONDS", "300")) # 5 mins default
    default_query    = os.getenv("DEFAULT_QUERY", "laptop")
    default_category = os.getenv("DEFAULT_CATEGORY", "Electronics")
    
    while True:
        print("\n" + "="*60)
        print(f"  PriceGuard AI — INFERENCE MODE ({datetime.now().strftime('%H:%M:%S')})")
        print("="*60)

        # ── Step 1B: Live scraping ──
        # Write initial status
        status_file = os.path.join(DATA_DIR, "scraper_status.json")
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"status": "scraping", "message": "Fetching Jumia Egypt (Multi-category)...", "timestamp": datetime.now().isoformat()}, f)

        from scraper import run_scraper
        live_products = run_scraper()

        if not live_products:
            print("  No live products found. Sleeping...")
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({"status": "idle", "message": "No products found. Sleeping.", "timestamp": datetime.now().isoformat()}, f)
            time.sleep(60)
            continue

        import pandas as pd
        raw_csv = os.path.join(BASE_DIR, "data", "raw_products.csv")
        reference_price = float(pd.read_csv(raw_csv)["price"].median()) if os.path.exists(raw_csv) else 50.0

        from react_loop    import run_react_loop
        from drift_monitor import check_drift, trigger_outer_loop

        print(f"\n[ReAct Loop] Processing {len(live_products)} live products …\n")

        results = []
        for idx, product in enumerate(live_products):
            # Update scraper status for UI
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "processing",
                    "current": product.get("product_name", "?"),
                    "index": idx + 1,
                    "total": len(live_products),
                    "timestamp": datetime.now().isoformat()
                }, f)

            try:
                res = run_react_loop(product)
                results.append(res)
                
                # Persist latest reasoning for dashboard polling
                state_file = os.path.join(DATA_DIR, "last_inference.json")
                with open(state_file, "w", encoding="utf-8") as f:
                    json.dump({"latest": res, "timestamp": datetime.now().isoformat()}, f)

                print(f"  OK {product.get('product_name','?')[:40]:<40} | "
                      f"Score={res['score']:.3f} | Path={res['path_taken']} | "
                      f"Action={res['action']['action_type']}")
            except Exception as exc:
                logger.error("ReAct loop failed: %s", exc)

        # ── Drift Check ──
        if results:
            print("\n[Drift Check] Monitoring system integrity …")
            drift_report = check_drift(results[-1]["confidence"], live_products[-1], results[-1]["label"])
            if drift_report["drift_triggered"]:
                print("  [ALERT] DRIFT DETECTED! Retraining...")
                trigger_outer_loop()
            else:
                print(f"  OK System Stable (EMA Conf: {drift_report['signal_1']['ema_confidence']:.2f})")

        print(f"\n[pipeline] Cycle complete. Waiting {interval}s...")
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"status": "idle", "message": f"Cycle complete. Waiting {interval}s.", "timestamp": datetime.now().isoformat()}, f)
        
        time.sleep(interval)

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
