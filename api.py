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
    price: float
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
    """Trigger the ReAct reasoning loop for a manual input."""
    try:
        res = run_react_loop(product.dict())
        # Also check drift for this manual case
        drift_report = check_drift(res["confidence"], product.dict(), res["label"])
        return {
            "result": res,
            "drift": drift_report
        }
    except Exception as e:
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
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
