
import logging
import os
import asyncio
import json
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_ASPECTS = """
You are an expert Aspect-Based Sentiment Analysis (ABSA) system.
Extract specific product/service aspects from the user's text.

Rules:
1. Only extract explicitly mentioned features/attributes (e.g. "battery life", "customer support").
2. Output ONLY a valid JSON object with one key: "aspects".
3. Each aspect object has: "aspect" (noun), "sentiment" (positive/negative/neutral), "quote" (exact minimal snippet).

Example output:
{"aspects": [{"aspect": "camera", "sentiment": "positive", "quote": "camera is amazing"}]}
"""


class SentimentAnalyzer:
    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_ready"):
            return
        self._ready = False
        self._pipeline = None
        self._mode = "uninitialized"
        logger.info("SentimentAnalyzer created (lazy init).")

    def _load(self):
        """Try loading primary model, then fallbacks."""
        if self._ready:
            return

        # Try Transformers
        try:
            from transformers import pipeline  # type: ignore
            self._pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                truncation=True,
                max_length=512,
            )
            self._mode = "transformers"
            self._ready = True
            logger.info("✅ Transformers model loaded.")
            return
        except Exception as e:
            logger.warning(f"Transformers load failed: {e}. Trying VADER...")

        # Try VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore
            self._vader = SentimentIntensityAnalyzer()
            self._mode = "vader"
            self._ready = True
            logger.info("✅ VADER fallback loaded.")
            return
        except Exception:
            pass

        # Try TextBlob
        try:
            from textblob import TextBlob  # type: ignore
            self._mode = "textblob"
            self._ready = True
            logger.info("✅ TextBlob fallback loaded.")
            return
        except Exception:
            pass

        logger.error("❌ All sentiment engines failed to load!")
        self._mode = "none"
        self._ready = True

    def analyze(self, text: str) -> Dict[str, Any]:
        """Analyze a single text. Returns sentiment + score."""
        self._load()
        text = text.strip()
        if not text:
            return {"sentiment": "neutral", "score": 0.5, "mode": self._mode}

        try:
            if self._mode == "transformers":
                return self._analyze_transformers(text)
            elif self._mode == "vader":
                return self._analyze_vader(text)
            elif self._mode == "textblob":
                return self._analyze_textblob(text)
        except Exception as e:
            logger.error(f"Analysis error: {e}")

        return {"sentiment": "neutral", "score": 0.5, "mode": "error_fallback"}

    def _analyze_transformers(self, text: str) -> Dict[str, Any]:
        result = self._pipeline(text[:512])[0]
        label_map = {
            "positive": "positive",
            "negative": "negative",
            "neutral": "neutral",
            "label_0": "negative",
            "label_1": "neutral",
            "label_2": "positive",
        }
        label = label_map.get(result["label"].lower(), "neutral")
        return {"sentiment": label, "score": round(result["score"], 4), "mode": "transformers"}

    def _analyze_vader(self, text: str) -> Dict[str, Any]:
        scores = self._vader.polarity_scores(text)
        compound = scores["compound"]
        if compound >= 0.05:
            sentiment = "positive"
        elif compound <= -0.05:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        score = (compound + 1) / 2  # normalize to [0,1]
        return {"sentiment": sentiment, "score": round(score, 4), "mode": "vader_fallback"}

    def _analyze_textblob(self, text: str) -> Dict[str, Any]:
        from textblob import TextBlob  # type: ignore
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            sentiment = "positive"
        elif polarity < -0.1:
            sentiment = "negative"
        else:
            sentiment = "neutral"
        score = (polarity + 1) / 2
        return {"sentiment": sentiment, "score": round(score, 4), "mode": "textblob_fallback"}

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Batch analyze a list of texts."""
        self._load()
        if self._mode == "transformers":
            try:
                # Truncate texts for transformer
                truncated = [t[:512] for t in texts if t.strip()]
                results = self._pipeline(truncated, batch_size=16)
                label_map = {
                    "positive": "positive",
                    "negative": "negative",
                    "neutral": "neutral",
                    "label_0": "negative",
                    "label_1": "neutral",
                    "label_2": "positive",
                }
                out = []
                for r in results:
                    label = label_map.get(r["label"].lower(), "neutral")
                    out.append({"sentiment": label, "score": round(r["score"], 4), "mode": "transformers"})
                return out
            except Exception as e:
                logger.warning(f"Batch transformers failed: {e}, falling back to single.")

        return [self.analyze(t) for t in texts]

    @property
    def mode(self) -> str:
        self._load()
        return self._mode


class AspectExtractor:
    """Uses Groq LLM to extract product aspects from review text (with LRU cache)."""

    _instance = None

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_ready"):
            return
        self._ready = False
        self._client = None
        self._cache: Dict[str, List[Dict]] = {}

    def _ensure_client(self):
        if self._ready:
            return
        api_key = os.getenv("GROQ_API_KEY", "")
        if api_key:
            try:
                from groq import Groq  # type: ignore
                self._client = Groq(api_key=api_key)
                logger.info("✅ Groq aspect extractor ready.")
            except ImportError:
                logger.warning("groq package not installed; aspect extraction disabled.")
        else:
            logger.warning("GROQ_API_KEY not set; aspect extraction disabled.")
        self._ready = True

    def _cache_key(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    async def extract_aspects(self, text: str) -> List[Dict[str, Any]]:
        """Extract aspects via Groq with retry and cache."""
        self._ensure_client()
        if not self._client:
            return []

        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        for attempt in range(1, 4):
            try:
                from groq import RateLimitError  # type: ignore
                response = await asyncio.to_thread(
                    lambda: self._client.chat.completions.create(
                        model="openai/gpt-oss-120b",
                        messages=[
                            {"role": "system", "content": _SYSTEM_PROMPT_ASPECTS},
                            {"role": "user", "content": text[:800]},
                        ],
                        max_tokens=400,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                )
                raw = response.choices[0].message.content.strip()
                # Use json.loads (not eval — security fix from original)
                data = json.loads(raw)
                aspects = data.get("aspects", [])
                self._cache[key] = aspects
                return aspects
            except Exception as e:
                if "rate_limit" in str(e).lower() and attempt < 3:
                    await asyncio.sleep(2 ** attempt)
                    continue
                logger.warning(f"Aspect extraction failed (attempt {attempt}): {e}")
                break
        return []


_analyzer = SentimentAnalyzer()
_aspect_extractor = AspectExtractor()



def analyze_text(text: str) -> Dict[str, Any]:
    """Analyze a single text (transformers/fallback)."""
    return _analyzer.analyze(text)


def analyze_batch(texts: List[str]) -> List[Dict[str, Any]]:
    """Batch analyze a list of texts."""
    return _analyzer.analyze_batch(texts)


async def analyze_hybrid(text: str) -> Dict[str, Any]:
    """
    Hybrid analysis: Transformers sentiment + Groq aspect extraction.
    Falls back gracefully if Groq is unavailable.
    """
    result = _analyzer.analyze(text)
    aspects = await _aspect_extractor.extract_aspects(text) if len(text) > 80 else []
    result["aspects"] = aspects
    return result


async def analyze_batch_hybrid(
    items: List[Dict[str, Any]],
    mode: str = "hybrid",
    llm_sample_rate: int = 3,
) -> List[Dict[str, Any]]:
    """
    Batch analyze with configurable mode (mirrors original backend):
      'transformers' — fast local, no aspects
      'hybrid'       — Transformers for all + Groq aspects every Nth item
      'llm'          — Groq aspects attempted on every item

    Returns list of result dicts with 'sentiment', 'score', 'mode', 'aspects'.
    """
    results = []
    total = len(items)

    for idx, item in enumerate(items):
        text = item.get("text", "")
        if not text:
            results.append({"sentiment": "neutral", "score": 0.5, "mode": "skip", "aspects": []})
            continue

        base = _analyzer.analyze(text)

        if mode == "transformers":
            base["aspects"] = []
            results.append(base)
            continue

        # Decide whether to call LLM for this item
        use_llm = (
            mode == "llm"
            or (mode == "hybrid" and idx % llm_sample_rate == 0 and len(text) > 100)
        )

        if use_llm:
            aspects = await _aspect_extractor.extract_aspects(text)
            base["aspects"] = aspects
        else:
            base["aspects"] = []

        results.append(base)

    return results


def get_analyzer_mode() -> str:
    return _analyzer.mode



def extract_word_frequencies(texts: List[str], top_n: int = 50) -> List[Dict[str, Any]]:
    """Extract word frequencies for word cloud, filtering stop words."""
    import re
    from collections import Counter

    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "it", "this", "that", "was", "are", "be", "been",
        "have", "has", "had", "do", "does", "did", "i", "my", "you", "your",
        "we", "our", "they", "their", "he", "she", "his", "her", "its",
        "not", "no", "can", "will", "would", "could", "should", "may",
        "also", "just", "from", "by", "about", "up", "so", "as", "more",
        "very", "if", "when", "then", "than", "all", "any", "some", "there",
        "what", "which", "who", "how", "get", "got", "like", "use", "one",
    }

    word_counts: Dict[str, int] = Counter()
    for text in texts:
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        for w in words:
            if w not in STOP_WORDS:
                word_counts[w] += 1

    return [{"text": w, "value": c} for w, c in word_counts.most_common(top_n)]
