"""
actions.py — Step 5: Autonomous API Action Layer
=================================================
Responsibilities:
  • Generate a pricing-strategy email body via the Gemini LLM
  • Deliver the alert email via SendGrid
  • Post a compact summary to Slack via webhook
  • Expose fire_alert() as the single public entry point

All API keys are loaded from the .env file via python-dotenv.
Errors are caught and logged to pipeline.log — never crash the pipeline.
"""

import os
import logging
import warnings
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

warnings.simplefilter("ignore", FutureWarning)
load_dotenv()

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Gemini LLM — pricing strategy recommendation
# ---------------------------------------------------------------------------

def generate_email_body(product_name: str, price_delta: float,
                        top_shap_feature: str, forecast_summary: str) -> str:
    """
    Call the Gemini API to write a 3-paragraph pricing strategy recommendation.

    Parameters
    ----------
    product_name : str
        Name of the product that triggered the alert.
    price_delta : float
        How much cheaper (absolute value) the competitor price is vs reference.
    top_shap_feature : str
        The most influential feature from the SHAP explanation.
    forecast_summary : str
        A short human-readable description of the 7-day SARIMA price forecast.

    Returns
    -------
    str
        A plain-English email body with 3 paragraphs.
    """
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set — returning fallback email body.")
        return (
            f"[ALERT] PriceGuard Alert for '{product_name}'\n\n"
            f"A competitor has undercut the reference price by ${price_delta:.2f}. "
            f"The most influential pricing factor is '{top_shap_feature}'. "
            f"7-day forecast: {forecast_summary}.\n\n"
            "Please review your pricing strategy immediately."
        )

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        prompt = (
            "You are a senior e-commerce pricing strategist. "
            "Write a professional plain-English email body with exactly 3 paragraphs "
            "(no bullet points, no markdown, no bold).\n\n"
            f"Product: {product_name}\n"
            f"Competitor price is ${price_delta:.2f} below our reference price.\n"
            f"Most important pricing factor (from SHAP): {top_shap_feature}\n"
            f"7-day SARIMA price forecast: {forecast_summary}\n\n"
            "Paragraph 1 — Summarise the competitive threat clearly.\n"
            "Paragraph 2 — Explain what the SHAP factor means for pricing.\n"
            "Paragraph 3 — Recommend a concrete 3-step action plan.\n"
        )

        response = model.generate_content(prompt)
        body = response.text.strip()
        logger.info("Gemini email body generated for '%s'.", product_name)
        return body

    except Exception as exc:
        logger.error("Gemini API error: %s", exc)
        return (
            f"[Gemini unavailable] PriceGuard alert for '{product_name}': "
            f"competitor undercut by ${price_delta:.2f}. "
            f"Key factor: {top_shap_feature}. Forecast: {forecast_summary}."
        )


# ---------------------------------------------------------------------------
# SendGrid — email delivery
# ---------------------------------------------------------------------------

def send_alert_email(product_name: str, email_body: str) -> bool:
    """
    Send the pricing alert email via multiple SMTP providers (Mailtrap + Gmail).
    Returns True if at least one provider succeeds.
    """
    # 1. Standard / Mailtrap SMTP
    providers = []
    
    m_host = os.getenv("SMTP_HOST")
    if m_host:
        providers.append({
            "host": m_host,
            "port": int(os.getenv("SMTP_PORT", "2525")),
            "user": os.getenv("SMTP_USER"),
            "pass": os.getenv("SMTP_PASS")
        })
        
    g_host = os.getenv("GMAIL_SMTP_HOST")
    if g_host:
        providers.append({
            "host": g_host,
            "port": int(os.getenv("GMAIL_SMTP_PORT", "587")),
            "user": os.getenv("GMAIL_SMTP_USER"),
            "pass": os.getenv("GMAIL_SMTP_PASS")
        })

    from_email = os.getenv("ALERT_FROM_EMAIL", "alerts@priceguard.com")
    to_email   = os.getenv("ALERT_TO_EMAIL", "manager@priceguard.com")

    if not providers:
        logger.warning("No SMTP providers configured — simulating email send.")
        print(f"\n[SIMULATED EMAIL]\nTo: {to_email}\nSubject: [ALERT] PriceGuard: {product_name}\n\n{email_body}\n")
        return False

    success_any = False
    for p in providers:
        try:
            message = MIMEMultipart("alternative")
            message["Subject"] = f"PriceGuard Alert: {product_name}"
            message["From"] = from_email
            message["To"] = to_email
            
            message.attach(MIMEText(email_body, "plain"))
            message.attach(MIMEText(email_body.replace("\n", "<br>"), "html"))

            with smtplib.SMTP(p["host"], p["port"], timeout=15) as server:
                if p["port"] == 587 or "gmail" in p["host"]:
                    server.starttls()
                server.login(p["user"], p["pass"])
                server.sendmail(from_email, to_email, message.as_string())
            
            logger.info("Email sent successfully via %s", p["host"])
            success_any = True
        except Exception as exc:
            logger.error("SMTP error for %s: %s", p["host"], exc)
            
    return success_any


# ---------------------------------------------------------------------------
# Slack — secondary webhook notification
# ---------------------------------------------------------------------------

def send_slack_notification(product_name: str, price: float, cluster_label: str,
                            shap_score: float, forecast_summary: str) -> bool:
    """
    Post a compact alert message to the #pricing-alerts Slack channel.

    Parameters
    ----------
    product_name : str
    price : float
    cluster_label : str
    shap_score : float
    forecast_summary : str

    Returns
    -------
    bool
        True on success, False on failure.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url or "your/webhook" in webhook_url:
        logger.warning("SLACK_WEBHOOK_URL not configured — skipping Slack notification.")
        print(
            f"[SIMULATED SLACK] 🚨 PriceGuard Alert — {product_name} dropped to "
            f"${price:.2f}. Cluster: {cluster_label}. SHAP score: {shap_score:.4f}. "
            f"Forecast: {forecast_summary}."
        )
        return False

    payload = {
        "text": (
            f"[ALERT] *PriceGuard Alert* — *{product_name}* dropped to *${price:.2f}*.\n"
            f"Cluster: `{cluster_label}` | SHAP score: `{shap_score:.4f}` | "
            f"Forecast: {forecast_summary}"
        )
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Slack notification sent for '%s'.", product_name)
        return True
    except Exception as exc:
        logger.error("Slack webhook error for '%s': %s", product_name, exc)
        return False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fire_alert(product_data: dict, shap_score: float, forecast_data: dict) -> dict:
    """
    Orchestrate the full alert sequence:
      1. Generate LLM email body (Gemini)
      2. Send email (SendGrid)
      3. Post to Slack

    Parameters
    ----------
    product_data : dict
        Must contain: product_name, price, reference_price, top_shap_feature, cluster_label.
    shap_score : float
        confidence_score × max(|shap_values|).
    forecast_data : dict
        Must contain: forecast_7d (list of 7 floats) and category (str).

    Returns
    -------
    dict
        Keys: email_status (bool), slack_status (bool), email_body (str).
    """
    product_name = product_data.get("product_name", "Unknown Product")
    price = float(product_data.get("price", 0))
    reference_price = float(product_data.get("reference_price", price * 1.1))
    top_shap_feature = product_data.get("top_shap_feature", "price")
    cluster_label = product_data.get("cluster_label", "budget")

    price_delta = max(reference_price - price, 0)

    # Build a human-readable forecast summary
    forecast_7d = forecast_data.get("forecast_7d", [])
    if forecast_7d:
        trend = "rising" if forecast_7d[-1] > forecast_7d[0] else "falling"
        forecast_summary = (
            f"prices are forecast to be {trend} over the next 7 days "
            f"(from ${forecast_7d[0]:.2f} to ${forecast_7d[-1]:.2f})"
        )
    else:
        forecast_summary = "no forecast available"

    # 1. Generate email body via Gemini
    email_body = generate_email_body(
        product_name=product_name,
        price_delta=price_delta,
        top_shap_feature=top_shap_feature,
        forecast_summary=forecast_summary,
    )

    # 2. Send email via SendGrid
    email_status = send_alert_email(product_name, email_body)

    # 3. Post to Slack
    slack_status = send_slack_notification(
        product_name=product_name,
        price=price,
        cluster_label=cluster_label,
        shap_score=shap_score,
        forecast_summary=forecast_summary,
    )

    logger.info(
        "fire_alert complete for '%s' — email=%s, slack=%s.",
        product_name, email_status, slack_status,
    )

    return {
        "email_status": email_status,
        "slack_status": slack_status,
        "email_body": email_body,
    }


# ---------------------------------------------------------------------------
# Utility re-exported for compatibility with the reference pattern
# ---------------------------------------------------------------------------

def calculate_action_score(svm_confidence: float, shap_score_sum: float,
                           sarima_trend_percent: float) -> float:
    """
    Compute a composite action urgency score.

    Parameters
    ----------
    svm_confidence : float
        Probability from SVM (0–1).
    shap_score_sum : float
        Sum of absolute SHAP values (normalised to 0–1 is ideal).
    sarima_trend_percent : float
        Forecast trend as a fraction (e.g. 0.05 = 5 % price drop expected).

    Returns
    -------
    float
        Weighted composite score.
    """
    return (svm_confidence * 0.4) + (shap_score_sum * 0.3) + (sarima_trend_percent * 0.3)
