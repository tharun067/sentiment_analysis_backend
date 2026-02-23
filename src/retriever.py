"""
Multi-API Data Retrieval with Retry, Fallback, and Caching.
Sources: SerpAPI (Google News), YouTube, Reddit, Twitter/Tweepy, Firecrawl.

Retry logic: 3 attempts with exponential back-off per source.
Caching:     1-hour TTL keyed on (source, query).
Fallback:    Each source fails independently; others continue.
"""

import os
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)



_response_cache: Dict[str, Dict] = {}
CACHE_TTL = 3600  # 1 hour


def _cache_key(source: str, query: str) -> str:
    return f"{source}::{query.lower().strip()}"


def _get_cached(source: str, query: str) -> Optional[List[dict]]:
    key = _cache_key(source, query)
    entry = _response_cache.get(key)
    if entry and (time.time() - entry["ts"] < CACHE_TTL):
        logger.info(f"📦 Cache hit: {key}")
        return entry["data"]
    return None


def _set_cache(source: str, query: str, data: List[dict]):
    key = _cache_key(source, query)
    _response_cache[key] = {"ts": time.time(), "data": data}




async def _retry(coro_fn, max_attempts: int = 3, delay: float = 2.0, label: str = ""):
    """Retry an async callable up to max_attempts times."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await coro_fn()
        except Exception as e:
            logger.warning(f"⚠️  {label} attempt {attempt}/{max_attempts} failed: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(delay * attempt)
    raise RuntimeError(f"{label} failed after {max_attempts} attempts")




async def fetch_serpapi(query: str, max_results: int = 20) -> List[dict]:
    """Fetch Google News snippets via SerpAPI."""
    cached = _get_cached("serpapi", query)
    if cached is not None:
        return cached

    api_key = os.getenv("SERPAPI_API_KEY", "")
    if not api_key:
        logger.warning("⚠️  SERPAPI_API_KEY not set. Skipping SerpAPI. Set it in your .env file to enable Google News retrieval.")
        return []

    async def _call():
        import serpapi  # type: ignore
        params = {
            "engine": "google_news",
            "q": query,
            "api_key": api_key,
            "num": max_results,
        }
        logger.debug(f"SerpAPI params: engine={params['engine']}, q={params['q']}, num={params['num']}")
        
        result = await asyncio.to_thread(serpapi.search, params)
        
        # Log error fields if SerpAPI returns a failure payload.
        if isinstance(result, dict):
            if result.get("error"):
                logger.error(f"❌ SerpAPI error: {result.get('error')}")
                return []
            if result.get("error_message"):
                logger.error(f"❌ SerpAPI error_message: {result.get('error_message')}")
                return []
            if result.get("search_metadata", {}).get("status") != "Success":
                status = result.get("search_metadata", {}).get("status", "Unknown")
                logger.warning(f"⚠️  SerpAPI status: {status}")
        
        items = result.get("news_results", [])
        if not items:
            logger.warning(
                f"⚠️  SerpAPI returned 0 results for query '{query}'. Possible reasons:\n"
                f"    - Invalid API key or no credits\n"
                f"    - Query not indexable in Google News\n"
                f"    - Plan quota exhausted"
            )
            return []
        
        logger.info(f"✅ SerpAPI returned {len(items)} news results")
        data = [
            {
                "text": item.get("snippet", ""),
                "source": "Google News",
                "url": item.get("link"),
                "timestamp": datetime.now(timezone.utc),
            }
            for item in items
            if item.get("snippet")
        ]
        _set_cache("serpapi", query, data)
        return data

    try:
        return await _retry(_call, label="SerpAPI")
    except Exception as e:
        logger.error(f"❌ SerpAPI permanently failed: {e}")
        return []


async def fetch_youtube(query: str, max_results: int = 20) -> List[dict]:
    """Fetch YouTube comments for a query."""
    cached = _get_cached("youtube", query)
    if cached is not None:
        return cached

    api_key = os.getenv("YOUTUBE_API_KEY", "")
    if not api_key:
        logger.warning("YOUTUBE_API_KEY not set, skipping.")
        return []

    async def _call():
        from googleapiclient.discovery import build  # type: ignore

        youtube = build("youtube", "v3", developerKey=api_key)
        search_resp = await asyncio.to_thread(
            lambda: youtube.search()
            .list(q=query, part="id", maxResults=max(1, max_results // 5), type="video")
            .execute()
        )
        video_ids = [item["id"]["videoId"] for item in search_resp.get("items", [])]
        if not video_ids:
            return []

        comments = []
        for vid_id in video_ids:
            try:
                c_resp = await asyncio.to_thread(
                    lambda v=vid_id: youtube.commentThreads()
                    .list(part="snippet", videoId=v, maxResults=5, textFormat="plainText")
                    .execute()
                )
                for item in c_resp.get("items", []):
                    snip = item["snippet"]["topLevelComment"]["snippet"]
                    comments.append({
                        "text": snip["textDisplay"],
                        "source": "YouTube",
                        "timestamp": datetime.strptime(
                            snip["publishedAt"], "%Y-%m-%dT%H:%M:%SZ"
                        ).replace(tzinfo=timezone.utc),
                    })
            except Exception as e:
                logger.warning(f"YouTube comment fetch failed for {vid_id}: {e}")

        data = comments[:max_results]
        _set_cache("youtube", query, data)
        return data

    try:
        return await _retry(_call, label="YouTube")
    except Exception as e:
        logger.error(f"YouTube permanently failed: {e}")
        return []


async def fetch_reddit(query: str, max_results: int = 20) -> List[dict]:
    """Fetch Reddit posts via asyncpraw."""
    cached = _get_cached("reddit", query)
    if cached is not None:
        return cached

    config = {
        "client_id": os.getenv("REDDIT_CLIENT_ID", ""),
        "client_secret": os.getenv("REDDIT_CLIENT_SECRET", ""),
        "user_agent": os.getenv("REDDIT_USER_AGENT", "sentiment-bot/1.0"),
        "username": os.getenv("REDDIT_USERNAME", ""),
        "password": os.getenv("REDDIT_PASSWORD", ""),
    }
    if not config["client_id"] or not config["client_secret"]:
        logger.warning("Reddit credentials not set, skipping.")
        return []

    try:
        import asyncpraw  # type: ignore

        async def _call():
            reddit = asyncpraw.Reddit(**config)
            results = []
            try:
                sub = await reddit.subreddit("all")
                async for post in sub.search(query=query, limit=max_results):
                    text = f"{post.title}. {post.selftext}".strip()
                    if text:
                        results.append({
                            "text": text[:1000],
                            "source": "Reddit",
                            "timestamp": datetime.fromtimestamp(
                                post.created_utc, tz=timezone.utc
                            ),
                        })
            finally:
                await reddit.close()
            _set_cache("reddit", query, results)
            return results

        return await _retry(_call, label="Reddit")
    except Exception as e:
        logger.error(f"Reddit permanently failed: {e}")
        return []


async def fetch_twitter(query: str, max_results: int = 20) -> List[dict]:
    """Fetch recent tweets via Tweepy (Twitter Bearer Token)."""
    cached = _get_cached("twitter", query)
    if cached is not None:
        return cached

    bearer_token = os.getenv("TWITTER_BEARER_TOKEN", "")
    if not bearer_token:
        logger.warning("TWITTER_BEARER_TOKEN not set, skipping Twitter.")
        return []

    async def _call():
        import tweepy  # type: ignore

        client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=False)
        response = await asyncio.to_thread(
            lambda: client.search_recent_tweets(
                query=f"{query} -is:retweet lang:en",
                max_results=max(10, min(max_results, 100)),
                tweet_fields=["created_at", "text"],
            )
        )
        data = response.data or []
        results = [
            {
                "text": t.text,
                "source": "Twitter",
                "timestamp": t.created_at or datetime.now(timezone.utc),
            }
            for t in data
        ]
        _set_cache("twitter", query, results)
        return results

    try:
        return await _retry(_call, label="Twitter")
    except Exception as e:
        logger.error(f"Twitter permanently failed: {e}")
        return []


async def fetch_firecrawl(urls: List[str]) -> List[dict]:
    """Scrape article text via Firecrawl."""
    api_key = os.getenv("FIRECRAWL_API_KEY", "")
    if not api_key or not urls:
        if api_key and not urls:
            logger.warning("Firecrawl skipped: no URLs provided from SerpAPI.")
        return []

    results = []
    try:
        from firecrawl import FirecrawlApp  # type: ignore

        app = FirecrawlApp(api_key=api_key)
        for url in urls[:5]:  # limit to 5 URLs per call
            try:
                # firecrawl-py ≥ 0.0.16 uses scrape_url(); older versions use scrape()
                try:
                    scraped = await asyncio.to_thread(
                        app.scrape_url, url, {"formats": ["markdown"]}
                    )
                except AttributeError:
                    scraped = await asyncio.to_thread(app.scrape, url)

                content = scraped.get("markdown", "") or scraped.get("content", "")
                if content:
                    results.append({
                        "text": content[:2000],
                        "source": "Firecrawl",
                        "timestamp": datetime.now(timezone.utc),
                        "url": url,
                    })
                else:
                    logger.warning(f"Firecrawl returned empty content for {url}")
            except Exception as e:
                logger.warning(f"Firecrawl scrape failed for {url}: {e}")
    except ImportError:
        logger.warning("firecrawl-py not installed, skipping Firecrawl.")
    return results




async def retrieve_all(query: str, max_per_source: int = 20) -> List[dict]:
    """
    Retrieve reviews/text from all configured sources with graceful fallback.
    Sources: SerpAPI (Google News) → YouTube → Reddit → Firecrawl
    """
    logger.info(f"🔍 Starting retrieval for: '{query}'")

    # Run ALL primary sources concurrently using gather on coroutines directly
    # This avoids event loop attachment issues with create_task()
    serp_results, yt_results, reddit_results, twitter_results = await asyncio.gather(
        fetch_serpapi(query, max_per_source),
        fetch_youtube(query, max_per_source),
        fetch_reddit(query, max_per_source),
        fetch_twitter(query, max_per_source),
        return_exceptions=True
    )

    # Handle any exceptions that were caught by gather
    serp_results = serp_results if not isinstance(serp_results, Exception) else []
    yt_results = yt_results if not isinstance(yt_results, Exception) else []
    reddit_results = reddit_results if not isinstance(reddit_results, Exception) else []
    twitter_results = twitter_results if not isinstance(twitter_results, Exception) else []

    # Use Firecrawl on URLs from SerpAPI
    urls = [r["url"] for r in serp_results if r.get("url")]
    firecrawl_results = await fetch_firecrawl(urls)

    all_results = serp_results + yt_results + reddit_results + twitter_results + firecrawl_results

    # Deduplicate by text prefix
    seen = set()
    deduped = []
    for item in all_results:
        key = item["text"][:80].strip().lower()
        if key and key not in seen:
            seen.add(key)
            deduped.append(item)

    logger.info(f"✅ Retrieved {len(deduped)} unique items across all sources.")
    return deduped


def get_cache_status() -> Dict[str, Any]:
    """Return cache statistics."""
    now = time.time()
    valid = sum(1 for e in _response_cache.values() if now - e["ts"] < CACHE_TTL)
    return {
        "total_entries": len(_response_cache),
        "valid_entries": valid,
        "expired_entries": len(_response_cache) - valid,
        "ttl_seconds": CACHE_TTL,
    }
