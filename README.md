# PriceGuard AI 🛡️

**E-Commerce Price Intelligence Agentic System**  
*University Final Project — 5-Step Sense-Reason-Act Pipeline*

---

## Overview

PriceGuard AI monitors competitor prices, clusters products into pricing tiers, classifies new listings in real-time, explains decisions with SHAP, forecasts future prices with SARIMA, and autonomously fires triple-redundant alerts (Mailtrap + Gmail + Slack).

```
Sense (Data) → Reason (Cluster + Classify + Explain + Forecast) → Act (Alert)
```

---

## Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure API keys (.env)
The system uses a **Dual-SMTP** approach for reliability.

```env
# Mailtrap Sandbox (Testing)
SMTP_HOST=sandbox.smtp.mailtrap.io
SMTP_PORT=2525
SMTP_USER=your_user
SMTP_PASS=your_pass

# Gmail Production (Real Alerts)
GMAIL_SMTP_HOST=smtp.gmail.com
GMAIL_SMTP_PORT=587
GMAIL_SMTP_USER=your-email@gmail.com
GMAIL_SMTP_PASS=your-app-password  # (16-char code from Google Security)

ALERT_FROM_EMAIL=alerts@priceguard.ai
ALERT_TO_EMAIL=your-email@gmail.com
GEMINI_API_KEY=your_key
SLACK_WEBHOOK_URL=your_webhook
PRICE_ALERT_THRESHOLD=0.10
```

---

## 🚀 How to Run

### 1. Start the Live Agent
The pipeline orchestrator handles everything from data ingestion to reasoning and alerting.
```bash
python pipeline.py
```
- **First Run**: Automatically trains the clustering and classification models.
- **Subsequent Runs**: Scrapes live data and executes the ReAct reasoning loop.
- **Drift Detection**: Triggers retraining if market conditions change significantly.

### 2. Launch the AI Dashboard
Monitor the agent's internal "brain" in real-world time.
```bash
streamlit run app.py
```
- **Zone 1**: Drift & Training Health.
- **Zone 2**: PCA Cluster Visualization.
- **Zone 3**: Live Inference Reasoning Traces.
- **Zone 4**: SHAP Explainability (Why did the agent act?).
- **Zone 5**: SARIMA Forecast (Where is the price going?).
- **Zone 6**: Recent Action History & Alert Logs.

---

## Project Structure
- `pipeline.py`: Orchestrator (Full system entry point).
- `app.py`: Streamlit Dashboard.
- `react_loop.py`: The 7-path Reasoning Engine.
- `actions.py`: Dual-SMTP & Slack delivery logic.
- `data_loader.py`, `clustering.py`, `classifier.py`: The ML backbone.
- `explainability.py`, `forecasting.py`: Advanced agentic signals.
