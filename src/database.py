from motor.motor_asyncio import AsyncIOMotorClient
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta, timezone
import uuid
import os
import logging
import asyncio
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError

logger = logging.getLogger(__name__)

_memory_cache: Dict[str, List[dict]] = {
    "reviews": [],
    "products": [],
    "ai_reports": [],
    "system_logs": []
}


class MongoManager:
    """Manages all MongoDB Atlas interactions with resilience and fallback."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, uri: Optional[str] = None, db_name: str = "sentiment_platform"):
        if hasattr(self, "_initialized"):
            return
        self._initialized = False
        self._mongo_uri = uri or os.getenv("MONGO_URI", "")
        self._db_name = db_name
        self._client = None
        self._db = None
        self._connected = False
        self._last_connect_attempt = None
        self._event_loop = None  # Track the event loop for Streamlit compatibility
        logger.info("MongoManager created (lazy connection).")


    async def connect(self) -> bool:
        """Attempt to establish MongoDB connection. Returns True if successful."""
        if self._connected:
            return True
        if not self._mongo_uri:
            logger.warning("No MONGO_URI set. Using in-memory fallback.")
            return False
        try:
            # Track the event loop for later checks
            try:
                self._event_loop = asyncio.get_running_loop()
            except RuntimeError:
                pass
            
            self._client = AsyncIOMotorClient(
                self._mongo_uri,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000,
            )
            # Ping to verify connection
            await self._client.admin.command("ping")
            self._db = self._client[self._db_name]
            self._connected = True
            self._initialized = True
            await self._ensure_indexes()
            logger.info("✅ MongoDB connection established.")
            return True
        except (ConnectionFailure, ServerSelectionTimeoutError, Exception) as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            # Don't call log_error during connection phase to avoid recursion
            self._connected = False
            return False

    async def _ensure_indexes(self):
        """Create indexes for performance."""
        try:
            reviews = self._db["reviews"]
            await reviews.create_index([("product_id", 1), ("created_at", -1)])
            await reviews.create_index([("sentiment", 1)])
            await reviews.create_index([("source", 1)])
        except Exception as e:
            logger.warning(f"Index creation warning: {e}")

    @property
    def is_connected(self) -> bool:
        return self._connected

    def _is_event_loop_changed(self) -> bool:
        """Check if the event loop has changed (common in Streamlit reruns)."""
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No running loop
            return self._event_loop is not None
        
        if self._event_loop is None:
            return False
        return current_loop != self._event_loop

    async def _reset_connection(self):
        """Reset the connection when event loop changes."""
        if self._client:
            try:
                self._client.close()
            except Exception as e:
                logger.debug(f"Error closing client: {e}")
        self._client = None
        self._db = None
        self._connected = False
        logger.info("Connection reset due to event loop change.")

    async def ensure_connected(self) -> bool:
        """Re-attempt connection if not connected."""
        # Check if event loop has changed (Streamlit compatibility)
        if self._is_event_loop_changed():
            await self._reset_connection()
        
        if not self._connected:
            return await self.connect()
        return True


    async def get_product(self, product_name: str) -> Optional[dict]:
        """Get product record by name."""
        if await self.ensure_connected():
            try:
                return await self._db["products"].find_one(
                    {"product_name": {"$regex": product_name, "$options": "i"}}
                )
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in get_product: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_product error: {e}")
        # Fallback
        for p in _memory_cache["products"]:
            if p["product_name"].lower() == product_name.lower():
                return p
        return None

    async def upsert_product(self, product_name: str, sentiment_summary: dict, total_reviews: int) -> str:
        """Create or update a product record. Returns product_id."""
        product_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, product_name.lower()))
        doc = {
            "product_id": product_id,
            "product_name": product_name,
            "last_updated": datetime.now(timezone.utc),
            "sentiment_summary": sentiment_summary,
            "total_reviews": total_reviews,
        }
        if await self.ensure_connected():
            try:
                await self._db["products"].update_one(
                    {"product_id": product_id},
                    {"$set": doc},
                    upsert=True,
                )
                return product_id
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in upsert_product: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"upsert_product error: {e}")
        # Fallback
        _memory_cache["products"] = [p for p in _memory_cache["products"] if p["product_id"] != product_id]
        _memory_cache["products"].append(doc)
        return product_id

    async def is_data_fresh(self, product_name: str, max_age_hours: int = 24) -> bool:
        """Check if product data is fresh (< max_age_hours old)."""
        product = await self.get_product(product_name)
        if not product:
            return False
        last_updated = product.get("last_updated")
        if not last_updated:
            return False
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        if last_updated.tzinfo is None:
            last_updated = last_updated.replace(tzinfo=timezone.utc)
        age = datetime.now(timezone.utc) - last_updated
        return age < timedelta(hours=max_age_hours)

    async def get_all_products(self) -> List[dict]:
        """Get all stored products with their metadata."""
        if await self.ensure_connected():
            try:
                cursor = self._db["products"].find({}, {"_id": 0}).sort("last_updated", -1)
                return await cursor.to_list(length=None)
            except (RuntimeError, asyncio.InvalidStateError) as e:
                # Handle event loop attachment errors
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in get_all_products: {e}. Resetting connection...")
                    await self._reset_connection()
                    return _memory_cache["products"]
                raise
            except Exception as e:
                logger.error(f"get_all_products error: {e}")
        # Fallback to memory cache
        return _memory_cache["products"]

    async def delete_product(self, product_id: str) -> bool:
        """Delete a product and all its associated reviews."""
        if await self.ensure_connected():
            try:
                # Delete product
                await self._db["products"].delete_one({"product_id": product_id})
                # Delete associated reviews
                result = await self._db["reviews"].delete_many({"product_id": product_id})
                logger.info(f"Deleted product {product_id} and {result.deleted_count} reviews.")
                return True
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in delete_product: {e}. Resetting connection...")
                    await self._reset_connection()
                    return False
                else:
                    raise
            except Exception as e:
                logger.error(f"delete_product error: {e}")
                return False
        # Fallback: remove from memory cache
        _memory_cache["products"] = [p for p in _memory_cache["products"] if p.get("product_id") != product_id]
        _memory_cache["reviews"] = [r for r in _memory_cache["reviews"] if r.get("product_id") != product_id]
        return True


    async def save_reviews(self, reviews: List[dict]) -> int:
        """Batch save reviews. Returns count saved."""
        if not reviews:
            return 0
        for r in reviews:
            r.setdefault("_id", str(uuid.uuid4()))
            r.setdefault("created_at", datetime.now(timezone.utc))

        if await self.ensure_connected():
            try:
                result = await self._db["reviews"].insert_many(reviews, ordered=False)
                return len(result.inserted_ids)
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in save_reviews: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"save_reviews error: {e}")
        # Fallback
        _memory_cache["reviews"].extend(reviews)
        return len(reviews)

    async def get_reviews(
        self,
        product_id: str,
        sentiment: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[dict]:
        """Fetch reviews with optional filters."""
        query: dict = {"product_id": product_id}
        if sentiment:
            query["sentiment"] = sentiment
        if source:
            query["source"] = source

        if await self.ensure_connected():
            try:
                cursor = (
                    self._db["reviews"]
                    .find(query, {"_id": 0})
                    .sort("created_at", -1)
                    .skip(skip)
                    .limit(limit)
                )
                return await cursor.to_list(length=limit)
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in get_reviews: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_reviews error: {e}")

        # Fallback
        results = [r for r in _memory_cache["reviews"] if r.get("product_id") == product_id]
        if sentiment:
            results = [r for r in results if r.get("sentiment") == sentiment]
        if source:
            results = [r for r in results if r.get("source") == source]
        return results[skip : skip + limit]

    async def get_review_count(self, product_id: str) -> int:
        """Count reviews for a product."""
        if await self.ensure_connected():
            try:
                return await self._db["reviews"].count_documents({"product_id": product_id})
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in get_review_count: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_review_count error: {e}")
        return len([r for r in _memory_cache["reviews"] if r.get("product_id") == product_id])

    async def get_sentiment_distribution(self, product_id: str) -> Dict[str, int]:
        """Get sentiment counts for a product."""
        if await self.ensure_connected():
            try:
                pipeline = [
                    {"$match": {"product_id": product_id}},
                    {"$group": {"_id": "$sentiment", "count": {"$sum": 1}}},
                ]
                results = await self._db["reviews"].aggregate(pipeline).to_list(None)
                dist = {"positive": 0, "negative": 0, "neutral": 0}
                for r in results:
                    if r["_id"] in dist:
                        dist[r["_id"]] = r["count"]
                return dist
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop error in get_sentiment_distribution: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_sentiment_distribution error: {e}")

        # Fallback
        reviews = [r for r in _memory_cache["reviews"] if r.get("product_id") == product_id]
        dist = {"positive": 0, "negative": 0, "neutral": 0}
        for r in reviews:
            s = r.get("sentiment", "neutral")
            if s in dist:
                dist[s] += 1
        return dist

    async def get_source_distribution(self, product_id: str) -> Dict[str, int]:
        """Get source counts for a product."""
        if await self.ensure_connected():
            try:
                pipeline = [
                    {"$match": {"product_id": product_id}},
                    {"$group": {"_id": "$source", "count": {"$sum": 1}}},
                ]
                results = await self._db["reviews"].aggregate(pipeline).to_list(None)
                return {r["_id"]: r["count"] for r in results if r["_id"]}
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop error in get_source_distribution: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_source_distribution error: {e}")

        # Fallback
        reviews = [r for r in _memory_cache["reviews"] if r.get("product_id") == product_id]
        dist: Dict[str, int] = {}
        for r in reviews:
            s = r.get("source", "unknown")
            dist[s] = dist.get(s, 0) + 1
        return dist

    async def get_trend_data(self, product_id: str, days: int = 7) -> List[dict]:
        """Get daily sentiment counts for trend chart."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        if await self.ensure_connected():
            try:
                pipeline = [
                    {"$match": {"product_id": product_id, "created_at": {"$gte": cutoff}}},
                    {
                        "$group": {
                            "_id": {
                                "date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
                                "sentiment": "$sentiment",
                            },
                            "count": {"$sum": 1},
                        }
                    },
                    {"$sort": {"_id.date": 1}},
                ]
                results = await self._db["reviews"].aggregate(pipeline).to_list(None)
                return results
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop error in get_trend_data: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_trend_data error: {e}")
        return []


    async def save_ai_report(self, product_id: str, report: dict) -> str:
        """Save AI-generated report."""
        report_id = str(uuid.uuid4())
        doc = {
            "_id": report_id,
            "product_id": product_id,
            "created_at": datetime.now(timezone.utc),
            **report,
        }
        if await self.ensure_connected():
            try:
                await self._db["ai_reports"].insert_one(doc)
                return report_id
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in save_ai_report: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"save_ai_report error: {e}")
        _memory_cache["ai_reports"].append(doc)
        return report_id

    async def get_latest_ai_report(self, product_id: str) -> Optional[dict]:
        """Get most recent AI report for a product."""
        if await self.ensure_connected():
            try:
                return await self._db["ai_reports"].find_one(
                    {"product_id": product_id},
                    {"_id": 0},
                    sort=[("created_at", -1)],
                )
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop mismatch in get_latest_ai_report: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_latest_ai_report error: {e}")
        # Fallback
        reports = [r for r in _memory_cache["ai_reports"] if r.get("product_id") == product_id]
        return reports[-1] if reports else None


    async def log_error(
        self,
        error_type: str,
        source: str,
        error_message: str,
        resolved: bool = False,
    ):
        """Log a system error."""
        doc = {
            "_id": str(uuid.uuid4()),
            "error_type": error_type,
            "source": source,
            "error_message": error_message,
            "resolved": resolved,
            "timestamp": datetime.now(timezone.utc),
        }
        if await self.ensure_connected():
            try:
                await self._db["system_logs"].insert_one(doc)
                return
            except (RuntimeError, asyncio.InvalidStateError):
                # Connection error - just use memory cache
                await self._reset_connection()
            except Exception:
                pass
        _memory_cache["system_logs"].append(doc)

    async def get_recent_logs(self, limit: int = 50) -> List[dict]:
        """Get recent system logs."""
        try:
            if await self.ensure_connected():
                try:
                    cursor = self._db["system_logs"].find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
                    return await cursor.to_list(length=limit)
                except (RuntimeError, asyncio.InvalidStateError) as e:
                    error_msg = str(e)
                    if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                        logger.warning(f"Event loop error in get_recent_logs: {e}. Resetting connection...")
                        await self._reset_connection()
                    else:
                        raise
        except Exception as e:
            logger.error(f"get_recent_logs error: {e}")
        return list(reversed(_memory_cache["system_logs"][-limit:]))

    async def delete_old_records(self, days: int = 30):
        """Delete reviews older than N days to maintain DB performance."""
        from datetime import timedelta
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        try:
            if await self.ensure_connected():
                try:
                    result = await self._db["reviews"].delete_many({"created_at": {"$lt": cutoff}})
                    logger.info(f"Deleted {result.deleted_count} reviews older than {days} days.")
                    return result.deleted_count
                except (RuntimeError, asyncio.InvalidStateError) as e:
                    error_msg = str(e)
                    if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                        logger.warning(f"Event loop error in delete_old_records: {e}. Resetting connection...")
                        await self._reset_connection()
                    else:
                        raise
        except Exception as e:
            logger.error(f"delete_old_records error: {e}")
        # Fallback
        before = len(_memory_cache["reviews"])
        _memory_cache["reviews"] = [
            r for r in _memory_cache["reviews"]
            if r.get("created_at", datetime.now(timezone.utc)) >= cutoff
        ]
        return before - len(_memory_cache["reviews"])

    async def get_aspects_word_cloud(self, product_id: str) -> List[Dict[str, Any]]:
        """
        Get word cloud data from extracted aspects (requires hybrid/llm mode data).
        Falls back to review text frequency if no aspects found.
        """
        if await self.ensure_connected():
            try:
                pipeline = [
                    {"$match": {"product_id": product_id, "aspects": {"$exists": True, "$ne": []}}},
                    {"$unwind": "$aspects"},
                    {"$group": {"_id": "$aspects.aspect", "count": {"$sum": 1}}},
                    {"$sort": {"count": -1}},
                    {"$limit": 50},
                    {"$project": {"text": "$_id", "value": "$count", "_id": 0}},
                ]
                results = await self._db["reviews"].aggregate(pipeline).to_list(None)
                if results:
                    return results
            except (RuntimeError, asyncio.InvalidStateError) as e:
                error_msg = str(e)
                if "attached to a different loop" in error_msg or "Event loop is closed" in error_msg or "InvalidStateError" in error_msg:
                    logger.warning(f"Event loop error in get_aspects_word_cloud: {e}. Resetting connection...")
                    await self._reset_connection()
                else:
                    raise
            except Exception as e:
                logger.error(f"get_aspects_word_cloud error: {e}")
        return []

    async def get_log_summary(self) -> Dict[str, Any]:
        """Get a summary of recent system logs."""
        logs = await self.get_recent_logs(100)
        summary: Dict[str, Any] = {
            "total": len(logs),
            "unresolved": sum(1 for l in logs if not l.get("resolved")),
            "by_type": {},
            "by_source": {},
        }
        for log in logs:
            t = log.get("error_type", "unknown")
            s = log.get("source", "unknown")
            summary["by_type"][t] = summary["by_type"].get(t, 0) + 1
            summary["by_source"][s] = summary["by_source"].get(s, 0) + 1
        return summary
