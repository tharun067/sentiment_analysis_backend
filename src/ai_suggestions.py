import os
import logging
import asyncio
import json
from typing import List, Dict, Any, Optional
from collections import Counter
import re

logger = logging.getLogger(__name__)



async def generate_ai_suggestions(
    product_name: str,
    reviews: List[dict],
    sentiment_dist: Dict[str, int],
) -> Dict[str, Any]:
    """
    Generate structured product improvement suggestions.
    Tries Groq first, falls back to rule-based analysis.
    """
    negative_reviews = [r for r in reviews if r.get("sentiment") == "negative"]
    neutral_reviews = [r for r in reviews if r.get("sentiment") == "neutral"]
    positive_reviews = [r for r in reviews if r.get("sentiment") == "positive"]

    # Handle both old format {negative: X} and new format {negative_count: X, negative_pct: Y}
    if "negative_count" in sentiment_dist:
        # New format with counts and percentages
        neg_count = sentiment_dist.get("negative_count", 0)
        neu_count = sentiment_dist.get("neutral_count", 0)
        pos_count = sentiment_dist.get("positive_count", 0)
        total = neg_count + neu_count + pos_count or 1
        neg_pct = sentiment_dist.get("negative_pct", 0)
        neu_pct = sentiment_dist.get("neutral_pct", 0)
    else:
        # Old format with just sentiment counts
        total = sum(sentiment_dist.values()) or 1
        neg_pct = (sentiment_dist.get("negative", 0) / total) * 100
        neu_pct = (sentiment_dist.get("neutral", 0) / total) * 100

    # Only generate suggestions if thresholds are met
    if neg_pct <= 30 and neu_pct <= 40:
        return {
            "triggered": False,
            "reason": f"Sentiment is healthy (negative: {neg_pct:.0f}%, neutral: {neu_pct:.0f}%)",
            "neg_pct": neg_pct,
            "neu_pct": neu_pct,
        }

    # Prepare sample texts
    neg_samples = [r["review_text"] for r in negative_reviews[:15]]
    neu_samples = [r["review_text"] for r in neutral_reviews[:10]]
    all_samples = neg_samples + neu_samples

    api_key = os.getenv("GROQ_API_KEY", "")
    if api_key:
        try:
            result = await _groq_suggestions(product_name, all_samples, neg_pct, neu_pct, api_key)
            result["triggered"] = True
            result["neg_pct"] = neg_pct
            result["neu_pct"] = neu_pct
            result["mode"] = "ai"
            return result
        except Exception as e:
            logger.warning(f"Groq failed: {e}. Using rule-based fallback.")
            await _log_fallback("groq_suggestions", str(e))

    # Rule-based fallback
    result = _rule_based_suggestions(product_name, all_samples, neg_pct, neu_pct)
    result["triggered"] = True
    result["neg_pct"] = neg_pct
    result["neu_pct"] = neu_pct
    result["mode"] = "rule_based_fallback"
    return result


async def _groq_suggestions(
    product_name: str,
    review_samples: List[str],
    neg_pct: float,
    neu_pct: float,
    api_key: str,
) -> Dict[str, Any]:
    """Call Groq API for structured suggestions."""
    from groq import Groq  # type: ignore

    sample_text = "\n".join([f"- {r[:200]}" for r in review_samples[:20]])
    prompt = f"""You are a product strategy consultant analyzing customer feedback for "{product_name}".

Sentiment: {neg_pct:.0f}% negative, {neu_pct:.0f}% neutral
Review samples:
{sample_text}

Provide a JSON response with EXACTLY this structure:
{{
  "key_complaints": ["list of 3-5 main customer complaints"],
  "root_causes": ["list of 3-5 underlying root causes"],
  "feature_improvements": ["list of 3-5 specific feature improvements"],
  "ux_improvements": ["list of 2-3 UX improvements"],
  "pricing_suggestions": ["list of 1-3 pricing recommendations"],
  "marketing_suggestions": ["list of 2-3 marketing strategy suggestions"],
  "executive_summary": "2-3 sentence high-level summary"
}}

Respond with ONLY the JSON, no markdown."""

    client = Groq(api_key=api_key)
    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            temperature=0.3,
        )
    )
    raw = response.choices[0].message.content.strip()
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    return json.loads(raw)


def _rule_based_suggestions(
    product_name: str,
    review_samples: List[str],
    neg_pct: float,
    neu_pct: float,
) -> Dict[str, Any]:
    """Generate rule-based suggestions from keyword frequency."""
    full_text = " ".join(review_samples).lower()
    words = re.findall(r"\b[a-zA-Z]{4,}\b", full_text)
    word_freq = Counter(words)

    # Domain-specific keyword categories
    ISSUE_KEYWORDS = {
        "quality": ["quality", "defect", "broken", "damage", "poor", "bad", "faulty"],
        "delivery": ["delivery", "shipping", "late", "delay", "arrived", "package"],
        "support": ["support", "customer", "service", "help", "response", "refund"],
        "price": ["price", "expensive", "cost", "overpriced", "worth", "value"],
        "usability": ["hard", "difficult", "confusing", "complicated", "unclear", "bug"],
        "performance": ["slow", "performance", "speed", "crash", "lag", "freeze"],
    }

    detected_issues = []
    for category, kws in ISSUE_KEYWORDS.items():
        score = sum(word_freq.get(kw, 0) for kw in kws)
        if score > 0:
            detected_issues.append((category, score))

    detected_issues.sort(key=lambda x: x[1], reverse=True)
    top_issues = [i[0] for i in detected_issues[:5]]

    # Build suggestions based on detected issues
    suggestions_map = {
        "quality": {
            "complaint": "Customers report product quality issues and defects",
            "feature": "Implement stricter quality control processes and incoming inspection",
            "ux": "Add quality guarantees and easy return process to product pages",
        },
        "delivery": {
            "complaint": "Shipping delays and packaging issues mentioned frequently",
            "feature": "Integrate real-time order tracking and improve packaging",
            "ux": "Add estimated delivery dates prominently during checkout",
        },
        "support": {
            "complaint": "Poor customer support experience and slow response times",
            "feature": "Build comprehensive self-service FAQ and chatbot support",
            "ux": "Streamline support ticket system with priority escalation",
        },
        "price": {
            "complaint": "Product perceived as overpriced relative to value",
            "feature": "Introduce tiered pricing or subscription models",
            "ux": "Highlight value propositions and ROI calculators",
        },
        "usability": {
            "complaint": "Users find the product difficult to use or configure",
            "feature": "Simplify onboarding with interactive tutorials",
            "ux": "Redesign key user flows based on usability testing",
        },
        "performance": {
            "complaint": "Performance issues including slowness and crashes reported",
            "feature": "Profile and optimize critical code paths, add load testing",
            "ux": "Implement progress indicators and offline graceful degradation",
        },
    }

    key_complaints = []
    feature_improvements = []
    ux_improvements = []

    for issue in top_issues:
        info = suggestions_map.get(issue, {})
        if info.get("complaint"):
            key_complaints.append(info["complaint"])
        if info.get("feature"):
            feature_improvements.append(info["feature"])
        if info.get("ux"):
            ux_improvements.append(info["ux"])

    if not key_complaints:
        key_complaints = ["General dissatisfaction detected in customer reviews"]
    if not feature_improvements:
        feature_improvements = ["Conduct user interviews to identify specific pain points"]
    if not ux_improvements:
        ux_improvements = ["Run usability tests to identify friction points"]

    return {
        "key_complaints": key_complaints[:5],
        "root_causes": [f"Systemic issues in {i} processes need review" for i in top_issues[:3]] or ["Process gaps identified"],
        "feature_improvements": feature_improvements[:5],
        "ux_improvements": ux_improvements[:3],
        "pricing_suggestions": [
            "Consider value-based pricing aligned with customer ROI",
            "Test introductory offers to lower purchase barrier",
        ],
        "marketing_suggestions": [
            "Focus messaging on resolving known pain points",
            "Leverage satisfied customers as case studies and testimonials",
            "Address top complaints proactively in product descriptions",
        ],
        "executive_summary": (
            f"{product_name} shows {neg_pct:.0f}% negative sentiment, primarily around "
            f"{', '.join(top_issues[:3]) if top_issues else 'general satisfaction'}. "
            "Immediate focus on quality and support improvements is recommended."
        ),
    }



async def generate_competitor_analysis(
    main_product: str,
    competitor_products: List[str],
    all_reviews: Dict[str, List[dict]],
) -> Dict[str, Any]:
    """Generate competitor comparison using Groq or rule-based fallback."""
    api_key = os.getenv("GROQ_API_KEY", "")

    summaries = {}
    for product, reviews in all_reviews.items():
        pos = sum(1 for r in reviews if r.get("sentiment") == "positive")
        neg = sum(1 for r in reviews if r.get("sentiment") == "negative")
        neu = sum(1 for r in reviews if r.get("sentiment") == "neutral")
        total = pos + neg + neu or 1
        summaries[product] = {
            "positive_pct": round(pos / total * 100, 1),
            "negative_pct": round(neg / total * 100, 1),
            "neutral_pct": round(neu / total * 100, 1),
            "total": total,
        }

    if api_key:
        try:
            return await _groq_competitor_analysis(main_product, competitor_products, summaries, api_key)
        except Exception as e:
            logger.warning(f"Groq competitor analysis failed: {e}")

    # Rule-based competitor summary
    insights = []
    main_pos = summaries.get(main_product, {}).get("positive_pct", 0)
    for comp in competitor_products:
        comp_pos = summaries.get(comp, {}).get("positive_pct", 0)
        if comp_pos > main_pos:
            insights.append(f"{comp} has higher positive sentiment ({comp_pos:.0f}% vs {main_pos:.0f}%)")
        else:
            insights.append(f"{main_product} outperforms {comp} in positive sentiment")

    return {
        "summaries": summaries,
        "insights": insights,
        "market_advantages": [f"{main_product} strengths identified from positive reviews"],
        "mode": "rule_based_fallback",
    }


async def _groq_competitor_analysis(
    main_product: str,
    competitors: List[str],
    summaries: Dict[str, Any],
    api_key: str,
) -> Dict[str, Any]:
    from groq import Groq  # type: ignore

    prompt = f"""Analyze this competitive sentiment landscape and provide insights.

Product: {main_product}
Competitors: {', '.join(competitors)}

Sentiment data:
{json.dumps(summaries, indent=2)}

Return JSON with:
{{
  "summaries": <the same summaries dict>,
  "insights": ["3-5 key competitive insights"],
  "market_advantages": ["2-3 areas where {main_product} has advantage"],
  "recommendations": ["2-3 strategic recommendations"],
  "mode": "ai"
}}

Respond ONLY with JSON."""

    client = Groq(api_key=api_key)
    response = await asyncio.to_thread(
        lambda: client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=800,
            temperature=0.3,
        )
    )
    raw = response.choices[0].message.content.strip()
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    result = json.loads(raw)
    result["summaries"] = summaries
    result["mode"] = "ai"
    return result


async def _log_fallback(source: str, error: str):
    """Helper to log AI fallback events."""
    logger.warning(f"[FALLBACK] {source}: {error}")
