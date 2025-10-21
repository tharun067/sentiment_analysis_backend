from motor.motor_asyncio import AsyncIOMotorClient,AsyncIOMotorDatabase
from typing import List, Dict, Optional
from datetime import datetime, timedelta, timezone
from .models import FeedItem, TrendData, SentimentData, WordCloudData, SentimentRecord, ProductTrend
import uuid
from dotenv import load_dotenv
import os
import logging
import asyncio

load_dotenv()

class MongoManager:
    """ Manages all interactions with MongoDB database."""
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(MongoManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self, uri: Optional[str] = None, db_name: str = 'sentimental_analysis'):
        if hasattr(self, 'client'):
            return  # Already initialized
        mongo_uri = uri or os.getenv("MONGO_URI")
        if not mongo_uri:
            raise ValueError("MongoDB URI environment variable not set.")
        self.client: AsyncIOMotorClient = AsyncIOMotorClient(mongo_uri)
        self.db: AsyncIOMotorDatabase = self.client[db_name]
        self.collection = self.db['feed_items']
        logging.info("MongoDB connection Initialized.")
    
    @staticmethod
    def get_time_range_filter(time_range: str) -> Dict:
        """ Helper function to get the time range filter for MongoDB queries. """
        now = datetime.now(timezone.utc)
        if time_range == '1h':
            start_time = now - timedelta(hours=1)
        elif time_range == '7d':
            start_time = now - timedelta(days=7)
        else:
            start_time = now - timedelta(hours=24) # Default to last 24 hours
        return {"timestamp": {"$gte": start_time}}

    async def save_feed_item(self, items: List[SentimentRecord]):
        """ Saves a batch of analyzed feed items to the database. 
        
        Args:
            items (List[dict]): List of feed items to save.
        
        Returns: None
        """
        if not items:
            return
        try:
            documents = [item.model_dump(by_alias=True) for item in items]
            for doc in documents:
                doc.setdefault("_id", str(uuid.uuid4()))
            await self.collection.insert_many(documents, ordered=False)
            logging.info(f"Inserted {len(documents)} feed items into the database.")
        except Exception as e:
            logging.error(f"Error inserting feed items: {e}")


    async def get_sentiment_distribution(self, query: str, time_range: str) -> SentimentData:
        """ Retrieves sentiment distribution for a given query over the last 24 hours. 
        
        Args:
            query (str): The search query.
        Returns:
            SentimentData: Sentiment distribution data.
        """
        time_filter = self.get_time_range_filter(time_range)
        pipeline = [
            {"$match": {"query": query, **time_filter}},
            {"$group":{"_id":"$analysis.sentiment","count":{"$sum":1}}},
        ]
        results = await self.collection.aggregate(pipeline).to_list(None)
        dist = {"positive": 0, "negative": 0, "neutral": 0}
        for res in results:
            if res["_id"] in dist:
                dist[res["_id"]] = res["count"]
        return SentimentData(**dist)
    
    async def get_sentiment_trends(self, query: str, time_range: str) -> List[TrendData]:
        """ Retrieves sentiment trends for a given query over a specified time range. 
        
        Args:
            query (str): The search query.
            time_range (str): Time range filter ('1h', '24h', '7d').
        Returns:
            List[TrendData]: List of sentiment trend data.
        """
        time_filter = self.get_time_range_filter(time_range)
        if time_range == '1h':
            date_format = "%Y-%m-%dT%H:%M:00Z"
        else:
            date_format = "%Y-%m-%dT%H:00:00Z"

        pipeline = [
            {"$match": {"query": query, **time_filter}},
            {
                "$group": {
                    "_id": {
                        "time_bucket": {
                            "$dateToString": {"format": date_format, "date": "$timestamp"}
                        },
                        "sentiment": "$analysis.sentiment",
                    },
                    "count": {"$sum": 1},
                }
            },
            {
                "$group": {
                    "_id": "$_id.time_bucket",
                    "counts": {"$push": {"k": "$_id.sentiment", "v": "$count"}},
                }
            },
            {"$addFields": {"sentiments": {"$arrayToObject": "$counts"}}},
            {"$sort": {"_id": 1}},
            {
                "$project": {
                    "_id": 0,
                    "timestamp": "$_id",
                    "positive": {"$ifNull": ["$sentiments.positive", 0]},
                    "negative": {"$ifNull": ["$sentiments.negative", 0]},
                    "neutral": {"$ifNull": ["$sentiments.neutral", 0]},
                }
            },
        ]
        results = await self.collection.aggregate(pipeline).to_list(None)
        return [TrendData(**res) for res in results]
    
    async def get_competitor_trends(self, products: List[str], time_range:str) -> List[ProductTrend]:
        """ Retrieves sentiment trends for multiple products over a specified time range. 
        
        Args:
            products (List[str]): List of product names.
            time_range (str): Time range filter ('1h', '24h', '7d').
        Returns:
            List[ProductTrend]: List of product trend data.
        """
        tasks = [self.get_sentiment_trends(product, time_range) for product in products]
        results = await asyncio.gather(*tasks)

        comparsion_data = []
        for product, trends in zip(products, results):
            comparsion_data.append(ProductTrend(product_name=product, trends=trends))

        return comparsion_data

    async def get_recent_feed(self, query: str, limit: int = 50) -> List[FeedItem]:
        """ Retrieves the most recent feed items for a given query. 
        
        Args:
            query (str): The search query.
            limit (int): Number of recent items to retrieve.
        Returns:
            List[FeedItem]: List of recent feed items.
        """
        cursor = self.collection.find({"query": query}).sort("timestamp", -1).limit(limit)
        items = await cursor.to_list(length=limit)
        processed_items = []
        for item in items:
            analysis = item.get("analysis", {})
            processed_items.append(
                {
                    "_id": str(item['_id']),
                    "text": item.get("text"),
                    "sentiment": analysis.get("sentiment"),
                    "score": analysis.get("score"),
                    "timestamp": item.get("timestamp"),
                    "source": item.get("source"),
                    "query": item.get("query"),
                }
            )
        return [FeedItem(**item) for item in processed_items]
    
    async def get_word_cloud_data(self, query: str, time_range: str = '24h') -> List[WordCloudData]:
        """ Retrieves word cloud data for a given query over the last 24 hours. 
        
        Args:
            query (str): The search query.
            time_range (str): The time range for the data (default is '24h').

        Returns:
            List[WordCloudData]: List of word cloud data.
        """
        time_filter = self.get_time_range_filter(time_range)
        pipeline = [
            {"$match": {"query": query, **time_filter}},
            {"$unwind": "$analysis.aspects"},
            {"$group": {"_id": "$analysis.aspects.aspect", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 50},
            {"$project": {"text": "$_id", "value": "$count", "_id": 0}},
        ]

        results = await self.collection.aggregate(pipeline).to_list(None)
        return [WordCloudData(**res) for res in results]
    
    async def get_documents_for_summary(self, query: str, sample_size: int = 25, time_range: str = '24h') -> Dict[str, List[str]]:
        """ Retrieves documents for positive and negative summary generation for a given query over the last 24 hours. 
        
        Args:
            query (str): The search query.
            sample_size (int): Number of documents to sample for summary.
        Returns:
            Dict[str, List[str]]: Dictionary with list of document texts.
        """
        time_filter = self.get_time_range_filter(time_range)
        positive_docs_cursor = self.collection.find(
            {"query": query, "analysis.sentiment": "positive", **time_filter},
            {"text": 1, "_id": 0}
        ).limit(sample_size)

        negative_docs_cursor = self.collection.find(
            {"query": query, "analysis.sentiment": "negative", **time_filter},
            {"text": 1, "_id": 0}
        ).limit(sample_size)

        positive_docs = [
            doc['text'] for doc in await positive_docs_cursor.to_list(length=sample_size)
        ]

        negative_docs = [
            doc['text'] for doc in await negative_docs_cursor.to_list(length=sample_size)
        ]

        return {
            "positive": positive_docs,
            "negative": negative_docs
        }
    

    async def delete_old_records(self, days: int = 30):
        """ Deletes records older than the specified number of days. 
        
        Args:
            days (int): Number of days to retain records.
        
        Returns: None
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            result = await self.collection.delete_many({"timestamp": {"$lt": cutoff_date}})
            logging.info(f"Deleted {result.deleted_count} old records from the database.")
        except Exception as e:
            logging.error(f"Error deleting old records: {e}")