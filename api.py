"""
api.py — PriceGuard AI Backend
===============================
FastAPI server to power the custom HTML/JS dashboard.
Serves static files and provides real-time data endpoints.
"""

import os
import json
import logging
from typing import Dict
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# Import core agentic modules
from react_loop import run_react_loop
from drift_monitor import check_drift
from actions import fire_alert

load_dotenv()

app = FastAPI(title="PriceGuard AI API")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_DIR = os.path.join(BASE_DIR, "data")

if not os.path.exists(STATIC_DIR):
    os.makedirs(STATIC_DIR)

# ── Models ──────────────────────────────────────────────────────────────────
class ProductInput(BaseModel):
    product_name: str
    price: float | None = None
    category: str
    rating: float = 4.5
    review_count: int = 50
    discount_percentage: float = 0.0
    stock_status: int = 1

# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "agent": "PriceGuard AI"}

@app.get("/api/last_inference")
def get_last_inference():
    """Poll the latest reasoning from the autonomous pipeline."""
    state_file = os.path.join(DATA_DIR, "last_inference.json")
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {"latest": None}
    return {"latest": None}

@app.get("/api/scraper_status")
def get_scraper_status():
    """Poll the current scraping progress."""
    status_file = os.path.join(DATA_DIR, "scraper_status.json")
    if os.path.exists(status_file):
        try:
            with open(status_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except: return {"status": "offline"}
    return {"status": "offline"}

@app.post("/api/run_react")
def run_manual_react(product: ProductInput):
    """
    Modular Production-Style Execution Pipeline:
    1. Scraper Layer
    2. Product Matcher Layer
    3. Validation Layer
    4. Reasoning Layer
    """
    try:
        from scraper import run_scraper
        from matcher import match_products, get_market_stats
        from datetime import datetime

        status_file = os.path.join(DATA_DIR, "scraper_status.json")

        # ── 1. Scraper Layer ───────────────────────────────────────────────────
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "scraping", 
                "message": f"SCANNING MARKET: Searching for '{product.product_name}'...", 
                "timestamp": datetime.now().isoformat()
            }, f)

        # Execute targeted scrape
        scraped = run_scraper(query=product.product_name, category=product.category)
        
        # ── 2. Product Matcher Layer ───────────────────────────────────────────
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "matching", 
                "message": f"VALIDATING RESULTS: Filtering {len(scraped)} raw listings...", 
                "timestamp": datetime.now().isoformat()
            }, f)

        matched = match_products(product.product_name, scraped, target_category=product.category)
        
        # ── 3. Validation Layer ───────────────────────────────────────────────
        if not matched:
            error_msg = f"No relevant '{product.product_name}' products found with similarity > 0.75."
            with open(status_file, "w", encoding="utf-8") as f:
                json.dump({
                    "status": "error", 
                    "message": error_msg, 
                    "timestamp": datetime.now().isoformat()
                }, f)
            return {"error": error_msg}

        market_stats = get_market_stats(matched)
        
        # If user left price empty, default to market average
        if product.price is None or product.price <= 0:
            product.price = market_stats.get("avg_price", 0.0)
        
        # Log top match for UI progress
        top_match = matched[0]
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "reasoning", 
                "message": f"REASONING: Comparing against '{top_match['product_name']}' (Conf: {top_match['similarity_score']:.2f})", 
                "timestamp": datetime.now().isoformat()
            }, f)

        # ── 4. Reasoning Layer ───────────────────────────────────────────────
        # Enrich input with real-market context
        product_data = product.dict()
        product_data["reference_price"] = market_stats["avg_price"]
        
        res = run_react_loop(product_data)
        
        # Add market intelligence to the result
        res["market_stats"] = market_stats
        res["matched_count"] = len(matched)
        res["price_diff_pct"] = ((product.price - market_stats["avg_price"]) / market_stats["avg_price"]) * 100 if market_stats["avg_price"] > 0 else 0
        
        # Determine verdict based on price diff
        if res["price_diff_pct"] < -10:
            res["verdict"] = "UNDERPRICED"
        elif res["price_diff_pct"] > 10:
            res["verdict"] = "OVERPRICED"
        else:
            res["verdict"] = "FAIR PRICE"

        # Final Status Update
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "complete", 
                "message": f"Analysis complete. Verdict: {res['verdict']}", 
                "timestamp": datetime.now().isoformat()
            }, f)

        # ── 5. Alerting Layer (Email Notification) ───────────────────────────
        # Always fire an alert for manual simulations to confirm the pipeline works
        try:
            alert_data = {
                **res,
                "top_shap_feature": res.get("dominant_shap", "price"),
                "cluster_label": res.get("label", "mid-range"),
            }
            # Use market_stats avg_price if available, otherwise fallback
            alert_data["reference_price"] = market_stats.get("avg_price", product.price * 1.1)
            
            # fire_alert(product_data, shap_score, forecast_data)
            alert_res = fire_alert(
                product_data=alert_data,
                shap_score=res.get("shap_score", 0.0),
                forecast_data={"forecast_7d": res.get("forecast_7d", []), "trend": res.get("forecast_trend", "stable")}
            )
            res["alert_result"] = alert_res
        except Exception as alert_exc:
            logging.error(f"Failed to send simulation alert: {alert_exc}")

        return {
            "result": res,
            "market_stats": market_stats,
            "matched_products": matched[:10]  # Return more matches for better dashboard visibility
        }

    except Exception as e:
        logging.error(f"Manual ReAct failure: {e}")
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({
                "status": "error", 
                "message": f"System Error: {str(e)}", 
                "timestamp": datetime.now().isoformat()
            }, f)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/labeled_data")
def get_labeled_data():
    """Return clustered data for the UI map."""
    path = os.path.join(DATA_DIR, "labeled_products.csv")
    if os.path.exists(path):
        import pandas as pd
        df = pd.read_csv(path)
        # Sample for frontend performance
        return df.sample(min(len(df), 500)).to_dict("records")
    return []

# ── Static File Serving ─────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"}
    )

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
