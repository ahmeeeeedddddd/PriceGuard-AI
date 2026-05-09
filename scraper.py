"""
scraper.py — Step 1B: Live Web Scraper (Jumia Egypt Integration)
============================================================
Scrapes live product listings from Jumia Egypt.
Supports both autonomous background mode (multi-query) and
targeted API mode (single user-supplied query).
"""

import os
import logging
import random
import requests
import time
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LIVE_CSV = os.path.join(BASE_DIR, "data", "live_products.csv")
BASE_URL = "https://www.jumia.com.eg/catalog/?q="

# ── Background autonomous mode: scrapes all 3 categories (~119 products) ──────
SEARCH_QUERIES = [
    {"query": "gaming laptop",      "category": "Electronics"},
    {"query": "educational toys",   "category": "Toys & Games"},
    {"query": "fitness equipment",  "category": "Sports & Outdoors"},
]

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def scrape_jumia(query: str, category: str) -> list[dict]:
    """Scrapes Jumia Egypt search results for a specific query."""
    if not query:
        return []

    url = f"{BASE_URL}{query.replace(' ', '+')}"
    products = []

    try:
        logger.info(f"Scraping Jumia Egypt for: {query} ({category})")
        resp = requests.get(url, headers=HEADERS, timeout=15)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")
        cards = soup.select("article.prd")

        for card in cards:
            try:
                name_tag = card.select_one("h3.name")
                name = name_tag.get_text(strip=True) if name_tag else None

                price_tag = card.select_one("div.prc")
                price_text = price_tag.get_text(strip=True) if price_tag else "0"
                price = float(''.join(c for c in price_text if c.isdigit() or c == '.'))

                rating = 4.5 if card.select_one("div.stars") else 4.0

                rev_tag = card.select_one("div.rev")
                reviews_text = rev_tag.get_text(strip=True) if rev_tag else "0"
                reviews = int(''.join(c for c in reviews_text if c.isdigit())) if reviews_text else 0

                if name and price > 0:
                    products.append({
                        "product_name": name,
                        "price": price,
                        "rating": rating,
                        "review_count": reviews,
                        "discount_percentage": 0.0,
                        "category": category,
                        "stock_status": 1,
                        "scraped_at": datetime.now().isoformat(),
                    })
            except Exception:
                continue

        logger.info(f"Found {len(products)} products on Jumia for '{query}'")
        return products

    except Exception as exc:
        logger.error(f"Jumia scrape failure for {query}: {exc}")
        return []


def run_scraper(query: str = None, category: str = None) -> list[dict]:
    """
    Orchestrator for the Jumia Egypt scrape.

    - If called with a query (from API / user simulation): targeted single-query mode.
    - If called without args (background pipeline): multi-query mode using SEARCH_QUERIES.
    """
    all_products = []

    if query:
        # ── Targeted mode (API / Simulate ReAct Loop) ─────────────────────────
        logger.info(f"Starting targeted Jumia scrape for '{query}'...")
        all_products = scrape_jumia(query, category or "General")
    else:
        # ── Autonomous background mode (pipeline.py loop) ─────────────────────
        logger.info("Starting Jumia Egypt Live Scraper (multi-query background mode)...")
        for target in SEARCH_QUERIES:
            items = scrape_jumia(target["query"], target["category"])
            all_products.extend(items)
            time.sleep(random.uniform(1, 2))  # polite delay between queries

        # Fallback to cached CSV if scraping fails entirely
        if not all_products:
            logger.warning("Scraper returned 0 results. Falling back to cached CSV.")
            if os.path.exists(LIVE_CSV):
                return pd.read_csv(LIVE_CSV).to_dict("records")
            return []

    if not all_products:
        logger.warning(f"Scraper returned 0 results for query='{query}'.")
        return []

    # Persist to CSV
    df = pd.DataFrame(all_products)
    os.makedirs(os.path.dirname(LIVE_CSV), exist_ok=True)
    df.to_csv(LIVE_CSV, index=False)

    label = f"'{query}'" if query else "all categories"
    print(f"[scraper] ✓ Collected {len(df)} results from Jumia Egypt for {label}.")
    return all_products


if __name__ == "__main__":
    run_scraper()
