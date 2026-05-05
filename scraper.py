"""
scraper.py — Step 1B: Live Web Scraper
=======================================
Scrapes live product listings from Noon.com using BeautifulSoup +
Requests with rotating user-agent headers.

All errors are caught and logged to pipeline.log.  The scraped products
are saved to data/live_products.csv and returned as a list of dicts.

Public API
----------
run_scraper() → list[dict]
"""

import os
import time
import logging
import warnings
import random
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime

warnings.simplefilter("ignore", FutureWarning)

logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
LIVE_CSV  = os.path.join(BASE_DIR, "data", "live_products.csv")

# ── Rotating user-agents ──────────────────────────────────────────────────────
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.4.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64; rv:125.0) "
    "Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36 Edg/122.0.0.0",
]

SCRAPE_TARGETS = [
    {
        "url":      "https://www.noon.com/egypt-en/sports-and-outdoors/",
        "category": "Sports & Outdoors",
    },
    {
        "url":      "https://www.noon.com/egypt-en/electronics/",
        "category": "Electronics",
    },
    {
        "url":      "https://www.noon.com/egypt-en/toys-and-games/",
        "category": "Toys & Games",
    },
]

TIMEOUT   = 12   # seconds per request
MAX_PAGES = 2    # pages to scrape per category


# ─────────────────────────────────────────────────────────────────────────────

def _get_headers() -> dict:
    """Return a request header dict with a randomly chosen user-agent."""
    return {
        "User-Agent":      random.choice(USER_AGENTS),
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection":      "keep-alive",
        "DNT":             "1",
    }


def _parse_noon_page(soup: BeautifulSoup, category: str) -> list[dict]:
    """
    Parse a Noon.com search/category page and extract product cards.
    Returns a list of product dicts matching PriceGuard schema.
    """
    products = []

    # Noon wraps each product card in <div data-qa="product-block">
    cards = soup.find_all("div", attrs={"data-qa": "product-block"})
    if not cards:
        # Fallback: try generic card class patterns
        cards = soup.find_all("div", class_=lambda c: c and "product" in c.lower())

    for card in cards:
        try:
            # ── Product name ─────────────────────────────────────────────────
            name_tag = (
                card.find("div", attrs={"data-qa": "product-name"})
                or card.find(class_=lambda c: c and "name" in c.lower())
            )
            name = name_tag.get_text(strip=True) if name_tag else None
            if not name:
                continue

            # ── Price ─────────────────────────────────────────────────────────
            price_tag = (
                card.find("strong", attrs={"data-qa": "product-price"})
                or card.find(class_=lambda c: c and "price" in c.lower())
            )
            price_text = price_tag.get_text(strip=True) if price_tag else "0"
            price = float(
                price_text.replace("EGP", "").replace(",", "").strip() or "0"
            )
            if price <= 0:
                continue

            # ── Rating ────────────────────────────────────────────────────────
            rating_tag = card.find(attrs={"data-qa": "product-rating"})
            try:
                rating = float(rating_tag.get_text(strip=True)) if rating_tag else 0.0
            except ValueError:
                rating = 0.0

            # ── Review count ──────────────────────────────────────────────────
            review_tag = card.find(attrs={"data-qa": "product-review-count"})
            try:
                review_count = int(
                    review_tag.get_text(strip=True).replace("(", "").replace(")", "")
                ) if review_tag else 0
            except ValueError:
                review_count = 0

            # ── Discount ──────────────────────────────────────────────────────
            disc_tag = card.find(class_=lambda c: c and "discount" in c.lower())
            try:
                discount_pct = float(
                    disc_tag.get_text(strip=True).replace("%", "").replace("-", "")
                ) if disc_tag else 0.0
            except ValueError:
                discount_pct = 0.0

            # ── Stock ─────────────────────────────────────────────────────────
            out_tag = card.find(class_=lambda c: c and "out-of-stock" in c.lower())
            stock_status = 0 if out_tag else 1

            products.append({
                "product_name":        name,
                "price":               price,
                "rating":              rating,
                "review_count":        review_count,
                "discount_percentage": discount_pct,
                "category":            category,
                "stock_status":        stock_status,
                "scraped_at":          datetime.now().isoformat(),
            })

        except Exception as exc:
            logger.debug("Skipped card due to parse error: %s", exc)
            continue

    return products


def _scrape_target(url: str, category: str) -> list[dict]:
    """
    Fetch and parse up to MAX_PAGES pages from a single target URL.
    """
    products = []
    for page in range(1, MAX_PAGES + 1):
        page_url = f"{url}?page={page}" if page > 1 else url
        try:
            resp = requests.get(page_url, headers=_get_headers(), timeout=TIMEOUT)
            resp.raise_for_status()
            soup  = BeautifulSoup(resp.text, "html.parser")
            items = _parse_noon_page(soup, category)
            products.extend(items)
            logger.info("Scraped %d products from %s (page %d).", len(items), page_url, page)
            time.sleep(random.uniform(1.5, 3.0))   # polite delay

        except requests.exceptions.Timeout:
            logger.error("Timeout scraping %s — skipping page.", page_url)
            break
        except requests.exceptions.HTTPError as exc:
            logger.error("HTTP error %s scraping %s — skipping.", exc.response.status_code, page_url)
            break
        except Exception as exc:
            logger.error("Unexpected error scraping %s: %s", page_url, exc)
            break

    return products


def _generate_fallback_products(n: int = 10) -> list[dict]:
    """
    If scraping fails completely, generate synthetic products to keep
    the pipeline running for demonstration purposes.
    """
    rng = __import__("numpy").random.default_rng(99)
    cats = ["Electronics", "Toys & Games", "Sports & Outdoors"]
    return [
        {
            "product_name":        f"Live Product #{i+1}",
            "price":               round(float(rng.uniform(5, 300)), 2),
            "rating":              round(float(rng.uniform(2.5, 5.0)), 1),
            "review_count":        int(rng.integers(0, 2000)),
            "discount_percentage": round(float(rng.uniform(0, 40)), 1),
            "category":            rng.choice(cats),
            "stock_status":        int(rng.integers(0, 2)),
            "scraped_at":          datetime.now().isoformat(),
        }
        for i in range(n)
    ]


def run_scraper() -> list[dict]:
    """
    Entry-point called by pipeline.py on inference runs.

    Scrapes live product listings from Noon.com, saves them to
    data/live_products.csv, and returns them as a list of dicts.

    Returns
    -------
    list[dict] — PriceGuard schema products ready for SVM inference.
    """
    logger.info("Starting live scraper …")
    all_products: list[dict] = []

    for target in SCRAPE_TARGETS:
        items = _scrape_target(target["url"], target["category"])
        all_products.extend(items)

    if not all_products:
        logger.warning("Scraper returned 0 products — using synthetic fallback data.")
        print("[scraper] ⚠  Live scraping returned no results; using synthetic fallback.")
        all_products = _generate_fallback_products(20)

    df = pd.DataFrame(all_products)
    os.makedirs(os.path.dirname(LIVE_CSV), exist_ok=True)
    df.to_csv(LIVE_CSV, index=False)

    logger.info("Scraper finished — %d products saved to %s.", len(all_products), LIVE_CSV)
    print(f"[scraper] ✓  {len(all_products)} live products saved to {LIVE_CSV}")
    return all_products


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    products = run_scraper()
    print(f"Scraped {len(products)} products.")
    if products:
        print(products[0])
