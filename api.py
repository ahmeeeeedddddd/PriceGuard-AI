"""
api.py — FastAPI Intelligent Interface
=======================================
Exposes the PriceGuard AI system via REST endpoints.
Each endpoint corresponds to exactly one loop layer.

Endpoints:
  POST /classify  →  Runs the complete inner ReAct loop
  GET  /forecast  →  Retrieves SARIMA trend for a category
  GET  /drift     →  Reports current drift signal status
  POST /recluster →  Manually triggers the outer ReAct loop
  GET  /health    →  System status & model metadata
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import Dict, Any, Optional

# Load React Loop and other modules
from react_loop import run_react_loop
from forecasting import run_forecasting
from drift_monitor import check_drift, trigger_outer_loop

app = FastAPI(title="PriceGuard AI Intelligent Service")
logger = logging.getLogger("api")

class ProductInput(BaseModel):
    product_name: str
    price: float
    rating: float = 3.0
    review_count: int = 0
    discount_percentage: float = 0.0
    stock_status: int = 1
    category: str = "Uncategorized"
    reference_price: Optional[float] = None

@app.get("/health")
def health():
    from pipeline import models_exist
    return {
        "status": "online",
        "models_loaded": models_exist(),
        "version": "1.1.0-agentic"
    }

@app.post("/classify")
def classify(product: ProductInput):
    """Runs the full inner ReAct loop for a single product."""
    try:
        result = run_react_loop(product.dict())
        return result
    except Exception as e:
        logger.error("API /classify error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/forecast/{category}")
def forecast(category: str):
    """Retrieves 7-day SARIMA forecast."""
    try:
        return run_forecasting(category=category)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/drift")
def drift_status():
    """Returns current status of drift signals."""
    # This usually needs a recent case to evaluate. 
    # Returning metadata about thresholds for now.
    return {
        "signal_1_threshold": 0.60,
        "signal_2_threshold": 0.15,
        "monitoring": "active"
    }

@app.post("/recluster")
def recluster():
    """Manually triggers the outer ReAct loop (autonomous retraining)."""
    success = trigger_outer_loop()
    if success:
        return {"status": "success", "message": "Outer loop completed. Models updated."}
    else:
        raise HTTPException(status_code=500, detail="Reclustering failed or was blocked by rollback gate.")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
