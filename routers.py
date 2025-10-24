from fastapi import APIRouter, BackgroundTasks, HTTPException, Query
from typing import List
from src.models import (
    StartResponse, SentimentData, TrendData, SummaryResponse,
    FeedItem, WordCloudData, AnalysisRequest, ComparisonRequest,
    CompetitorComparisonResponse
)
from src.database import MongoManager
from src.analysis import GroqAnalysis
from src.services import AnalysisPipeline
from datetime import datetime
import logging

# Initialize router
router = APIRouter(
    prefix="/api",
    tags=["Sentiment Analysis"]
)

# Lazy initialization - services will be created on first use
_db_manager = None
__analysis_pipeline = None
_groq_analyzer = None

def get_db_manager():
    """Lazy initialization of database manager."""
    global _db_manager
    if _db_manager is None:
        logging.info("Initializing MongoManager...")
        _db_manager = MongoManager()
    return _db_manager

def get_analysis_pipeline():
    """Lazy initialization of analysis pipeline."""
    global _analysis_pipeline
    if _analysis_pipeline is None:
        logging.info("Initializing AnalysisPipeline...")
        _analysis_pipeline = AnalysisPipeline(max_concurrent_llm=5)
    return _analysis_pipeline

def get_groq_analyzer():
    """Lazy initialization of Groq analyzer."""
    global _groq_analyzer
    if _groq_analyzer is None:
        logging.info("Initializing GroqAnalysis...")
        _groq_analyzer = GroqAnalysis()
    return _groq_analyzer

logging.info("API Router initialized (services will load on first request).")

@router.post("/start_analysis", response_model=StartResponse)
async def start_analysis(request: AnalysisRequest, background_tasks: BackgroundTasks, mode: str = Query('hybrid', description="Analysis mode: 'transformers', 'llm', or 'hybrid'", regex="^(transformers|llm|hybrid)$")):
    """
    Starts sentiment analysis in the background with configurable mode.
    
    **Modes:**
    - `hybrid`: Smart mix of Transformers + LLM (recommended, 70-80% cost reduction)
    - `transformers`: Fast, local analysis only (no rate limits, no aspects)
    - `llm`: Full LLM analysis (detailed but may hit rate limits)
    
    **Process:**
    1. Retrieves data from multiple sources
    2. Analyzes sentiment using selected mode
    3. Saves results to database
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must be provided.")
    logging.info(f"Starting analysis for request for query: {query} in mode: {mode}")
    background_tasks.add_task(_analysis_pipeline.run, query, mode)
    return StartResponse(
        status="success",
        query=query,
        message="Analysis has been started and is running in the background."
    )

@router.post("/compare_competitors", response_model=CompetitorComparisonResponse)
async def get_competitors_comparison(request: ComparisonRequest, background_tasks: BackgroundTasks, mode: str = Query('hybrid', description="Analysis mode: 'transformers', 'llm', or 'hybrid'", regex="^(transformers|llm|hybrid)$")):
    """
    Compares sentiment trends for multiple products.
    
    **Features:**
    - Retrieves existing trend data immediately
    - Triggers background analysis for fresh data
    - Supports 2-10 products for comparison
    """
    products = request.Products
    time_range = request.time_range
    if len(products) < 2 or len(products) > 10:
        raise HTTPException(status_code=400, detail="Please provide between 2 and 10 products for comparison.")
    
    logging.info(f"Starting competitor comparison for products: {products} in mode: {mode}")
    # Trigger background analysis for each product
    for product in products:
        background_tasks.add_task(_analysis_pipeline.run, product, mode)
    
    # Retrieve existing trend data
    comparison_data = await _db_manager.get_competitor_trends(products=products,time_range=time_range)

    return CompetitorComparisonResponse(comparison=comparison_data)

@router.get("/distribution/{query}", response_model=SentimentData)
async def get_distribution(query: str, time_range: str = Query('24h', description="Time range for sentiment distribution", regex="^(1h|24h|7d)$")):
    """
    Gets sentiment distribution (positive, negative, neutral counts).
    
    **Returns:**
    - Count of positive sentiments
    - Count of negative sentiments  
    - Count of neutral sentiments
    
    **Note:** Returns all zeros if no data exists for the query.
    """
    try:
        db = get_db_manager()
        distribution = await db.get_sentiment_distribution(query=query, time_range=time_range)

        # check if any data exists
        total = distribution.positive + distribution.negative + distribution.neutral
        if total == 0:
            logging.warning(f"No sentiment data found for query: {query} in time range: {time_range}")
        
        return distribution
    except Exception as e:
        logging.error(f"Error retrieving sentiment distribution for query: {query} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving sentiment distribution.")
    
@router.get("/trends/{query}", response_model=List[TrendData])
async def get_trends(query: str, time_range: str = Query('24h', description="Time range for sentiment trends", regex="^(1h|24h|7d)$")):
    """
    Gets sentiment trends over time for visualization.
    
    **Returns time-series data with:**
    - Timestamp buckets
    - Positive count per bucket
    - Negative count per bucket
    - Neutral count per bucket
    
    **Use for:** Line charts, area charts, trend visualization
    """
    try:
        db = get_db_manager()
        trends = await db.get_sentiment_trends(query=query, time_range=time_range)
        if not trends:
            logging.warning(f"No trend data found for query: {query} in time range: {time_range}")
            return []
        return trends
    except Exception as e:
        logging.error(f"Error retrieving sentiment trends for query: {query} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving sentiment trends.")
    
@router.get("/summary/{query}", response_model=SummaryResponse)
async def get_summary(query: str, sample_size: int = Query(25, ge=5, le=100, description="Number of documents to use for summary (5-100)"), time_range: str = Query('24h', description="Time range for summary", regex="^(1h|24h|7d)$")):
    """
    Generates intelligent summaries using hybrid approach.
    
    **Process:**
    1. Fetches recent positive and negative documents from database
    2. Uses Transformers for quick text summarization
    3. Uses LLM for structured insights and key points (when available)
    4. Falls back to Transformers-only if LLM fails
    
    **Returns:**
    - Positive summary with overview and key insights
    - Negative summary with overview and key insights
    - Overall sentiment for each
    """
    try:
        db = get_db_manager()
        docs = await db.get_documents_for_summary(query=query, sample_size=sample_size, time_range=time_range)

        if not docs['positive'] and not docs['negative']:
            logging.warning(f"No documents found for summary for query: {query} in time range: {time_range}")
            raise HTTPException(status_code=404, detail="No documents available for summary.")
        
        logging.info(f"Generating summary for query: {query} using {len(docs['positive'])} positive and {len(docs['negative'])} negative documents.")

        analyzer = get_groq_analyzer()
        positive_summary = await analyzer.generate_structured_summary(docs['positive'], sentiment_context='positive')
        negative_summary = await analyzer.generate_structured_summary(docs['negative'], sentiment_context='negative')

        return SummaryResponse(
            positive_summary=positive_summary,
            negative_summary=negative_summary
        )
    except Exception as e:
        logging.error(f"Error generating summary for query: {query} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error while generating summary.")

@router.get(("/feed/{query}"), response_model=List[FeedItem])
async def get_feed(query: str, limit: int = Query(50, ge=1, le=500, description="Number of feed items to retrieve (1-500)")):
    """
    Gets the most recent feed items for a given query.
    
    **Returns:**
    - Text content
    - Sentiment (positive/negative/neutral)
    - Confidence score
    - Source (reddit/twitter/youtube/news)
    - Timestamp
    
    **Sorted by:** Most recent first
    """
    try:
        db = get_db_manager()
        feed = await db.get_recent_feed(query=query, limit=limit)

        if not feed:
            logging.warning(f"No feed items found for query: {query}")
            return []
        return feed
    except Exception as e:
        logging.error(f"Error retrieving feed items for query: {query} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving feed items.")
    
@router.get("/wordcloud/{query}", response_model=List[WordCloudData])
async def get_wordcloud(query: str, time_range: str = Query('24h', description="Time range for word cloud", regex="^(1h|24h|7d)$")):
    """
    Gets word cloud data from analyzed aspects.
    
    **Features:**
    - Extracts product aspects (camera, battery, screen, etc.)
    - Shows frequency of each aspect
    - Only includes aspects from hybrid/llm mode analysis
    
    **Returns:** List of {text, value} pairs for word cloud visualization
    
    **Note:** Empty list if no aspects were extracted (transformers-only mode doesn't extract aspects)
    """
    try:
        db = get_db_manager()
        word_data = await db.get_word_cloud_data(query=query, time_range=time_range)

        if not word_data:
            logging.warning(f"No word cloud data found for query: {query} in time range: {time_range}")
            return []
        return word_data
    except Exception as e:
        logging.error(f"Error retrieving word cloud data for query: {query} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error while retrieving word cloud data.")
    

@router.post("/delete_data/{query}")
async def delete_data(query: str, days: int = Query(30, ge=1, le=365, description="Delete data older than this number of days (1-365)")):
    """
    Deletes old data to maintain database performance.
    
    **Warning:** This deletes ALL records older than specified days, not just for this query.
    
    **Use cases:**
    - Regular maintenance
    - Clean up old test data
    - Free up database space
    """
    try:
        db = get_db_manager()
        await db.delete_old_records(days=days)

        logging.info(f"Deleted records older than {days} days as requested for query: {query}")
        return {"status": "success", "message": f"Records older than {days} days have been deleted."}
    except Exception as e:
        logging.error(f"Error deleting old records for query: {query} - {e}")
        raise HTTPException(status_code=500, detail="Internal server error while deleting old records.")
    

# Health check endpoint
@router.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring service status.
    
    **Checks:**
    - Database connection
    - Service initialization
    
    **Returns:** Status and timestamp
    """
    try:
        # Simple check - try to access database
        db = get_db_manager()
        await db.collection.find_one({})
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": {
                "database": "connected",
                "analysis_pipeline": "ready",
                "groq_analyzer": "ready"
            }
        }
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


@router.get("/info")
async def api_info():
    """
    Returns API information and usage guide.
    
    **Explains:**
    - Available analysis modes
    - Cost/speed tradeoffs
    - Feature availability
    - Endpoint usage
    """
    return {
        "api_version": "2.0",
        "name": "Hybrid Sentiment Analysis API",
        "description": "Intelligent sentiment analysis combining local Transformers with LLM",
        
        "analysis_modes": {
            "hybrid": {
                "description": "Smart mix of Transformers + LLM (RECOMMENDED)",
                "speed": "medium",
                "cost": "low (70-80% API reduction)",
                "features": ["sentiment", "aspects", "emotions", "intent"],
                "use_when": "You need aspects but want to minimize API costs"
            },
            "transformers": {
                "description": "Fast local analysis only",
                "speed": "very fast",
                "cost": "free (no API calls)",
                "features": ["sentiment", "emotions"],
                "use_when": "You need quick results and don't need aspect extraction"
            },
            "llm": {
                "description": "Full LLM analysis for everything",
                "speed": "slow",
                "cost": "high (maximum API usage)",
                "features": ["sentiment", "aspects", "emotions", "intent"],
                "use_when": "You need maximum detail and don't mind API costs"
            }
        },
        
        "workflow": {
            "step_1": "POST /api/start-analysis with query and mode",
            "step_2": "Wait 2-5 minutes for background processing",
            "step_3": "GET /api/distribution/{query} to see results",
            "step_4": "GET /api/summary/{query} for insights",
            "step_5": "GET /api/feed/{query} for detailed items"
        },
        
        "endpoints": {
            "analysis": {
                "start": "POST /api/start-analysis",
                "compare": "POST /api/compare_competitors"
            },
            "results": {
                "distribution": "GET /api/distribution/{query}",
                "trends": "GET /api/trends/{query}",
                "summary": "GET /api/summary/{query}",
                "feed": "GET /api/feed/{query}",
                "wordcloud": "GET /api/wordcloud/{query}"
            },
            "maintenance": {
                "delete": "POST /api/delete-data/{query}",
                "health": "GET /api/health"
            }
        },
        
        "time_ranges": {
            "supported": ["1h", "24h", "7d"],
            "default": "24h",
            "note": "Use '1h' for real-time monitoring, '7d' for trends"
        },
        
        "best_practices": {
            "1": "Start with 'hybrid' mode for best balance",
            "2": "Use 'transformers' mode for testing/development",
            "3": "Run analysis during off-peak hours to avoid rate limits",
            "4": "Set up regular cleanup (30 days) to maintain performance",
            "5": "Monitor feed size - limit to 50-100 for better performance"
        },
        
        "rate_limits": {
            "groq_api": "Depends on your API key tier",
            "transformers": "No limits (runs locally)",
            "concurrent_llm": "Limited to 5 by default in pipeline"
        }
    }