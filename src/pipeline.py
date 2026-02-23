"""
Main Analysis Pipeline
Orchestrates: Retrieval → Sentiment Analysis (with hybrid mode) → DB Storage
With caching, deduplication, and error resilience.

Modes:
  'transformers' — fast local sentiment, no aspects
  'hybrid'       — Transformers + Groq aspects on every 3rd item (default)
  'llm'          — Groq aspects on every item
"""

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from .retriever import retrieve_all
from .analysis import analyze_batch, analyze_batch_hybrid, extract_word_frequencies
from .database import MongoManager

logger = logging.getLogger(__name__)


async def run_pipeline(
    product_name: str,
    force_refresh: bool = False,
    mode: str = "hybrid",
    db: Optional[MongoManager] = None,
) -> Dict[str, Any]:
    """
    Full pipeline: check cache → retrieve → analyze → store → return summary.

    mode: 'transformers' | 'hybrid' | 'llm'

    Returns a dict with:
        - product_id
        - sentiment_dist: {positive, negative, neutral}
        - total_reviews
        - source_dist: {source: count}
        - reviews: list of review dicts
        - word_frequencies: list of {text, value}
        - from_cache: bool
        - errors: list of error strings
        - mode: str
    """
    if db is None:
        db = MongoManager()
        await db.connect()

    errors = []

    #  Check for fresh cached data 
    if not force_refresh and await db.is_data_fresh(product_name, max_age_hours=24):
        product = await db.get_product(product_name)
        product_id = product["product_id"]
        reviews = await db.get_reviews(product_id, limit=200)
        sentiment_dist = await db.get_sentiment_distribution(product_id)
        source_dist = await db.get_source_distribution(product_id)
        word_freq = extract_word_frequencies([r["review_text"] for r in reviews])
        logger.info(f"📦 Loaded '{product_name}' from cache ({len(reviews)} reviews).")
        return {
            "product_id": product_id,
            "product_name": product_name,
            "sentiment_dist": sentiment_dist,
            "total_reviews": len(reviews),
            "source_dist": source_dist,
            "reviews": reviews,
            "word_frequencies": word_freq,
            "from_cache": True,
            "errors": [],
            "mode": mode,
        }

    #  Retrieve raw data ─
    try:
        raw_items = await retrieve_all(product_name, max_per_source=25)
    except Exception as e:
        error_msg = f"Data retrieval failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        await db.log_error("retrieval", "pipeline", error_msg)
        raw_items = []

    if not raw_items:
        return {
            "product_id": None,
            "product_name": product_name,
            "sentiment_dist": {"positive": 0, "negative": 0, "neutral": 0},
            "total_reviews": 0,
            "source_dist": {},
            "reviews": [],
            "word_frequencies": [],
            "from_cache": False,
            "errors": errors + ["No reviews found for this product. Try a different name or check your API keys."],
            "mode": mode,
        }

    #  Batch sentiment + aspect analysis 
    try:
        if mode == "transformers":
            texts = [item["text"] for item in raw_items]
            base_results = analyze_batch(texts)
            analysis_results = [{**r, "aspects": []} for r in base_results]
        else:
            # hybrid or llm — use async batch with aspect extraction
            analysis_results = await analyze_batch_hybrid(raw_items, mode=mode)
    except Exception as e:
        error_msg = f"Sentiment analysis failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        await db.log_error("sentiment_analysis", "pipeline", error_msg)
        # Fallback: neutral for all
        analysis_results = [
            {"sentiment": "neutral", "score": 0.5, "mode": "error_fallback", "aspects": []}
            for _ in raw_items
        ]

    #  Build review documents 
    product_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, product_name.lower()))
    review_docs = []
    for item, result in zip(raw_items, analysis_results):
        review_docs.append({
            "product_id": product_id,
            "review_text": item["text"][:2000],
            "sentiment": result["sentiment"],
            "confidence_score": result["score"],
            "analysis_mode": result.get("mode", mode),
            "aspects": result.get("aspects", []),
            "source": item.get("source", "unknown"),
            "url": item.get("url"),
            "created_at": item.get("timestamp", datetime.now(timezone.utc)),
        })

    #  Compute distributions 
    sentiment_dist = {"positive": 0, "negative": 0, "neutral": 0}
    source_dist: Dict[str, int] = {}
    for r in review_docs:
        s = r["sentiment"]
        if s in sentiment_dist:
            sentiment_dist[s] += 1
        src = r["source"]
        source_dist[src] = source_dist.get(src, 0) + 1

    total = len(review_docs) if review_docs else 1  # Avoid division by zero
    sentiment_summary = {
        "positive_count": sentiment_dist["positive"],
        "negative_count": sentiment_dist["negative"],
        "neutral_count": sentiment_dist["neutral"],
        "positive_pct": round((sentiment_dist["positive"] / total) * 100, 1),
        "negative_pct": round((sentiment_dist["negative"] / total) * 100, 1),
        "neutral_pct": round((sentiment_dist["neutral"] / total) * 100, 1),
    }

    try:
        await db.save_reviews(review_docs)
        await db.upsert_product(product_name, sentiment_summary, len(review_docs))
    except Exception as e:
        error_msg = f"DB storage failed: {e}"
        logger.error(error_msg)
        errors.append(error_msg)
        await db.log_error("db_write", "pipeline", error_msg)

    texts = [item["text"] for item in raw_items]
    word_freq = extract_word_frequencies(texts)

    # Also extract from aspects for a richer word cloud
    aspect_words = []
    for r in review_docs:
        for asp in r.get("aspects", []):
            if isinstance(asp, dict) and asp.get("aspect"):
                aspect_words.append(asp["aspect"])
    if aspect_words:
        from collections import Counter
        aspect_counts = Counter(aspect_words)
        existing_words = {w["text"] for w in word_freq}
        for word, count in aspect_counts.most_common(20):
            if word not in existing_words:
                word_freq.append({"text": word, "value": count * 3})  # weight aspects higher

    logger.info(
        f"✅ Pipeline complete [{mode}]: {len(review_docs)} reviews for '{product_name}'. "
        f"Pos:{sentiment_summary['positive_count']} Neg:{sentiment_summary['negative_count']} Neu:{sentiment_summary['neutral_count']}"
    )
    return {
        "product_id": product_id,
        "product_name": product_name,
        "sentiment_dist": sentiment_summary,
        "total_reviews": len(review_docs),
        "source_dist": source_dist,
        "reviews": review_docs,
        "word_frequencies": word_freq,
        "from_cache": False,
        "errors": errors,
        "mode": mode,
    }
