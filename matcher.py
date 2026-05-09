"""
matcher.py — Validation & Semantic Matching Layer
================================================
Filters scraped products to ensure they match the user's intent.
Uses TF-IDF + Cosine Similarity with strict keyword validation.
"""

import logging
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# Minimum similarity threshold to accept a product (Production standard)
MIN_SIMILARITY = 0.75 

def extract_keywords(text: str) -> set:
    """Extract important keywords from a string, ignoring small words."""
    words = re.findall(r'\b\w{3,}\b', text.lower())
    return set(words)

def match_products(target_name: str, scraped_products: list[dict], target_category: str = None) -> list[dict]:
    """
    Filters and ranks scraped products based on semantic similarity and keyword presence.
    
    Returns:
    --------
    List of matched products with an added 'similarity_score' field.
    """
    if not scraped_products:
        return []

    logger.info(f"Starting product matching for: '{target_name}'")
    target_keywords = extract_keywords(target_name)
    
    # 1. Category & Keyword Pre-filtering
    filtered_list = []
    for p in scraped_products:
        # Category check (if provided)
        if target_category and p.get("category") and p.get("category") != target_category:
            continue
            
        # Hard keyword check: At least 30% of target keywords must be present in the name
        p_name = p.get("product_name", "").lower()
        matched_kw = [kw for kw in target_keywords if kw in p_name]
        
        # If the target name is short (e.g. 1-2 words), require at least one match
        if len(target_keywords) <= 2:
            if len(matched_kw) >= 1:
                filtered_list.append(p)
        else:
            # For longer queries, require at least 40% keyword overlap
            if (len(matched_kw) / len(target_keywords)) >= 0.4:
                filtered_list.append(p)

    if not filtered_list:
        logger.warning(f"No products passed keyword pre-filter for '{target_name}'")
        return []

    # 2. Semantic Matching using Intent Coverage (Subset Matching)
    # Cosine similarity penalizes long product names too harshly. 
    # Instead, we measure how much of the target query is covered in the product name.
    matched_products = []
    
    for p in filtered_list:
        p_name = p.get("product_name", "").lower()
        # Find exactly which target keywords are in the product name
        matched_kw = [kw for kw in target_keywords if kw in p_name]
        
        # Base score: percentage of target keywords found in the product name
        coverage_score = len(matched_kw) / len(target_keywords) if target_keywords else 0.0
        
        # Minor penalty for extremely long titles to prioritize cleaner matches
        p_words = extract_keywords(p_name)
        length_penalty = 0.0
        if len(p_words) > len(target_keywords):
            # Penalize slightly (-0.02 per extra word, max -0.15 penalty)
            extra_words = len(p_words) - len(target_keywords)
            length_penalty = min(extra_words * 0.02, 0.15)
            
        final_score = coverage_score - length_penalty
        
        # Ensure score is between 0 and 1
        final_score = max(0.0, min(1.0, final_score))
        
        p["similarity_score"] = float(final_score)
        
        if final_score >= MIN_SIMILARITY:
            logger.info(f"[MATCH] '{p['product_name']}' matched with confidence {final_score:.2f}")
            matched_products.append(p)
        else:
            logger.info(f"[REJECTED] '{p['product_name']}' similarity {final_score:.2f} (Below {MIN_SIMILARITY})")

    # Sort by similarity score descending
    matched_products = sorted(matched_products, key=lambda x: x["similarity_score"], reverse=True)
    
    logger.info(f"Matching complete. Found {len(matched_products)} relevant products.")
    return matched_products

def get_market_stats(matched_products: list[dict]) -> dict:
    """Computes market statistics for a list of matched products."""
    if not matched_products:
        return {
            "avg_price": 0.0,
            "min_price": 0.0,
            "max_price": 0.0,
            "count": 0
        }
        
    prices = [p["price"] for p in matched_products]
    return {
        "avg_price": float(np.mean(prices)),
        "min_price": float(np.min(prices)),
        "max_price": float(np.max(prices)),
        "count": len(matched_products)
    }
