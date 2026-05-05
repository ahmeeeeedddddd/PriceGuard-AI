# PriceGuard AI 🛡️

**E-Commerce Price Intelligence Agentic System**  
*University Final Project — 5-Step Sense-Reason-Act Pipeline*

---

## Overview

PriceGuard AI monitors competitor prices, clusters products into pricing tiers, classifies new listings in real-time using an SVM, explains decisions with SHAP, forecasts future prices with SARIMA, and autonomously fires email + Slack alerts when a competitor undercuts your threshold.

```
Sense (Data) → Reason (Cluster + Classify + Explain + Forecast) → Act (Alert)
```

---

## Project Structure

```
priceguard_ai/
├── app.py              # Streamlit dashboard
├── scraper.py          # Step 1B: live web scraping (Noon.com)
├── data_loader.py      # Step 1A: Kaggle dataset ingestion & cleaning
├── clustering.py       # Step 2: DBSCAN + Fuzzy C-Means
├── classifier.py       # Step 3: SVM training & inference
├── explainability.py   # Step 4A: SHAP KernelExplainer
├── forecasting.py      # Step 4B: SARIMA + drift detection
├── actions.py          # Step 5: Gemini + SendGrid + Slack alerts
├── pipeline.py         # Orchestrator (train OR infer)
├── data/
│   ├── amazon_products.csv   ← Kaggle dataset (pre-downloaded)
│   ├── raw_products.csv      ← output of data_loader.py
│   ├── labeled_products.csv  ← output of clustering.py
│   └── live_products.csv     ← output of scraper.py
├── models/
│   ├── svm_model.pkl
│   └── scaler.pkl
├── outputs/
│   ├── cluster_plot.png
│   ├── shap_plot.png
│   └── forecast_plot.png
├── .env                ← API keys (never commit)
└── requirements.txt
```

---

## Setup

### 1. Install dependencies

```bash
cd priceguard_ai
pip install -r requirements.txt
```

### 2. Configure API keys

Edit `.env` with your real values:

```env
SENDGRID_API_KEY=SG.xxxx
ALERT_FROM_EMAIL=alerts@yourcompany.com
ALERT_TO_EMAIL=manager@yourcompany.com
GEMINI_API_KEY=AIzaSy...
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PRICE_ALERT_THRESHOLD=0.10
```

### 3. Ensure the Kaggle dataset is present

```
data/amazon_products.csv   ← already placed
```

---

## Running the System

### Train (first run — no models exist)
```bash
python pipeline.py
```
This will:
1. Clean the Kaggle CSV → `data/raw_products.csv`
2. Run DBSCAN + Fuzzy C-Means → `data/labeled_products.csv` + `outputs/cluster_plot.png`
3. Train SVM via GridSearchCV → `models/svm_model.pkl` + `models/scaler.pkl`

### Infer (subsequent runs — models exist)
```bash
python pipeline.py
```
This will:
1. Scrape live products from Noon.com → `data/live_products.csv`
2. Classify each product with the SVM
3. Compute SHAP explanations → `outputs/shap_plot.png`
4. Forecast 7-day prices with SARIMA → `outputs/forecast_plot.png`
5. Fire email + Slack alerts for underpriced products

### Launch the dashboard
```bash
streamlit run app.py
```

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENSE (Data)                              │
│  Step 1A: Kaggle CSV (training)  │  Step 1B: Live Scraper       │
└───────────────────┬──────────────┴──────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────┐
│                       REASON (Models)                             │
│  Step 2: DBSCAN + Fuzzy C-Means → Cluster Labels                 │
│  Step 3: SVM (RBF + GridSearchCV) → Pricing State                │
│  Step 4A: SHAP KernelExplainer → Feature Importance              │
│  Step 4B: SARIMA (pmdarima) → 7-Day Forecast + Drift Detection   │
└───────────────────┬──────────────────────────────────────────────┘
                    │
┌───────────────────▼──────────────────────────────────────────────┐
│                         ACT (Alerts)                              │
│  Step 5: Gemini LLM body → SendGrid email + Slack webhook        │
└──────────────────────────────────────────────────────────────────┘
```

---

## Key Technical Decisions

| Design Choice | Rationale |
|---|---|
| DBSCAN before FCM | Removes outliers that would skew fuzzy cluster centroids |
| FCM → SVM (two-stage) | FCM labels data unsupervised; SVM learns a fast decision boundary |
| `random_state=42` everywhere | Full reproducibility |
| SHAP KernelExplainer | Model-agnostic; works with any sklearn estimator |
| `pmdarima.auto_arima(seasonal=True, m=7)` | Automatic order selection with weekly seasonality |
| Gemini → SendGrid | Free LLM + transactional email; swappable via env vars |
| `st.session_state` + `st.rerun()` | Proper Streamlit state management (no deprecated APIs) |

---

## Cluster Labels

| Label | Cluster ID | Description |
|---|---|---|
| Budget | 0 | Lowest mean price tier |
| Mid-Range | 1 | Middle price tier |
| Premium | 2 | Highest mean price tier |

*Labels are assigned by sorting FCM centroids by mean price at cluster time.*

---

## Environment Variables Reference

| Variable | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key for LLM email body |
| `SENDGRID_API_KEY` | SendGrid key for transactional email |
| `ALERT_FROM_EMAIL` | Sender email address |
| `ALERT_TO_EMAIL` | Recipient email address |
| `SLACK_WEBHOOK_URL` | Incoming webhook for #pricing-alerts |
| `PRICE_ALERT_THRESHOLD` | Decimal threshold (default `0.10` = 10%) |
