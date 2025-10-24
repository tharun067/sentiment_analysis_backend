from typing import List, Dict, Any
import tweepy
import asyncpraw
from googleapiclient.discovery import build
import serpapi
from datetime import datetime, timezone
import asyncio
import os
import logging
from .firecrawl_retriever import FirecrawlScraper

# --- Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Individual Scraper Classes ---

class TwitterScraper:
    def __init__(self, api_key: str):
        self.client = tweepy.Client(bearer_token=api_key) if api_key else None
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not self.client:
            logger.warning("Twitter client not initialized. Skipping search.")
            return []
        logger.info(f"Searching Twitter for '{query}'")
        try:
            response = self.client.search_recent_tweets(
                query=f"{query} -is:retweet lang:en",
                max_results=max(10, min(max_results, 100)),
                tweet_fields=['created_at', 'text']
            )
            return [{"text": t.text, "source": "Twitter", "timestamp": t.created_at} for t in response.data or []]
        except Exception as e:
            logger.error(f"Error searching tweets: {e}")
            return []

class RedditScraper:
    def __init__(self, **kwargs):
        self.client = asyncpraw.Reddit(**kwargs) if all(kwargs.values()) else None
    
    async def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not self.client:
            logger.warning("Reddit client not initialized. Skipping search.")
            return []
        logger.info(f"Searching Reddit for '{query}'")
        results = []
        try:
            subreddit = await self.client.subreddit("all")
            async for s in subreddit.search(query=query, limit=max_results):
                results.append({
                    "text": f"{s.title}. {s.selftext}", "source": "Reddit",
                    "timestamp": datetime.fromtimestamp(s.created_utc, tz=timezone.utc)
                })
            return results
        except Exception as e:
            logger.error(f"Error searching Reddit: {e}")
            return []
        finally:
            if self.client: await self.client.close()

class GoogleNewsScraper:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not self.api_key:
            logger.warning("SerpApi client not initialized. Skipping Google News search.")
            return []
        logger.info(f"Searching Google News for '{query}'")
        try:
            params = {"engine": "google_news", "q": query, "api_key": self.api_key, "num": max_results}
            search = serpapi.search(params)
            items = search.get("news_results", [])
            return [
                {"text": item.get("snippet", ""), "source": "Google News", "timestamp": datetime.now(timezone.utc), "url": item.get("link")}
                for item in items if item.get("snippet") and item.get("link")
            ]
        except Exception as e:
            logger.error(f"Error searching Google News: {e}")
            return []

class YouTubeScraper:
    def __init__(self, api_key: str):
        self.client = build("youtube", "v3", developerKey=api_key) if api_key else None
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        if not self.client:
            logger.warning("YouTube client not initialized. Skipping search.")
            return []
        logger.info(f"Searching YouTube for '{query}'")
        try:
            s_res = self.client.search().list(q=query, part='id', maxResults=max(1, max_results//5), type='video').execute()
            video_ids = [item['id']['videoId'] for item in s_res.get('items', [])]
            if not video_ids: return []
            
            comments = []
            comments_per_video = max(1, max_results // len(video_ids))
            for video_id in video_ids:
                try:
                    c_res = self.client.commentThreads().list(part='snippet', videoId=video_id, maxResults=comments_per_video, textFormat='plainText').execute()
                    for item in c_res.get('items', []):
                        comment = item['snippet']['topLevelComment']['snippet']
                        comments.append({
                            "text": comment['textDisplay'], "source": "YouTube",
                            "timestamp": datetime.strptime(comment['publishedAt'], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
                        })
                except Exception: continue
            return comments[:max_results]
        except Exception as e:
            logger.error(f"Error searching YouTube: {e}")
            return []

# --- Main Retriever Orchestrator ---

class MultiAPIRetriever:
    def __init__(self):
        # Store credentials but don't initialize clients yet
        self._twitter_token = os.getenv('TWITTER_BEARER_TOKEN')
        self._reddit_config = {
            'client_id': os.getenv('REDDIT_CLIENT_ID'),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET'),
            'user_agent': os.getenv('REDDIT_USER_AGENT'),
            'username': os.getenv('REDDIT_USERNAME'),
            'password': os.getenv('REDDIT_PASSWORD')
        }
        self._serpapi_key = os.getenv('SERPAPI_API_KEY')
        self._youtube_key = os.getenv('YOUTUBE_API_KEY')
        self._firecrawl_key = os.getenv('FIRECRAWL_API_KEY')
        
        # Lazy-loaded clients
        self._twitter = None
        self._reddit = None
        self._google_news = None
        self._youtube = None
        self._firecrawl = None
        
        logger.info("MultiAPIRetriever initialized (clients will be created on first use).")
    
    @property
    def twitter(self):
        if self._twitter is None:
            self._twitter = TwitterScraper(self._twitter_token)
        return self._twitter
    
    @property
    def reddit(self):
        if self._reddit is None:
            self._reddit = RedditScraper(**self._reddit_config)
        return self._reddit
    
    @property
    def google_news(self):
        if self._google_news is None:
            self._google_news = GoogleNewsScraper(self._serpapi_key)
        return self._google_news
    
    @property
    def youtube(self):
        if self._youtube is None:
            self._youtube = YouTubeScraper(self._youtube_key)
        return self._youtube
    
    @property
    def firecrawl(self):
        if self._firecrawl is None:
            self._firecrawl = FirecrawlScraper(self._firecrawl_key)
        return self._firecrawl
    
    async def retrieve(self, query: str, max_results_per_api: int = 20) -> List[Dict[str, Any]]:
        # CORRECTED: Uncommented the scrapers to ensure they run.
        scrapers = [
            self.reddit.search(query, max_results_per_api),
            asyncio.to_thread(self.twitter.search, query, max_results_per_api),
            asyncio.to_thread(self.youtube.search, query, max_results_per_api),
        ]

        google_results = await asyncio.to_thread(self.google_news.search, query, 5)
        
        firecrawl_tasks = []
        for result in google_results:
            if result.get('url'):
                firecrawl_tasks.append(asyncio.to_thread(self.firecrawl.scrape, result['url']))

        try:
            all_tasks = scrapers + firecrawl_tasks
            all_results = await asyncio.gather(*all_tasks, return_exceptions=True)

            final_data = google_results
            for res in all_results:
                if isinstance(res, list):
                    final_data.extend(res)
                elif isinstance(res, Exception):
                    logger.error(f"Error during data retrieval: {res}")
            
            logger.info(f"Retrieved total {len(final_data)} items from all APIs for query '{query}'.")
            return final_data
        finally:
            # Ensure reddit client is closed to avoid unclosed session warnings
            try:
                await self.reddit.close()
            except Exception:
                pass

