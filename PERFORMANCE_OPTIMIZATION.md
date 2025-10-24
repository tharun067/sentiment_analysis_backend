# Performance Optimization Guide

## 🚀 Startup Latency Reduction

### Problem Identified
Your application was taking **~5-6 seconds** to start due to:
1. **MongoDB connection** (~1 second)
2. **Transformer models loading** (~3-4 seconds) - Two large PyTorch models
3. **API client initialization** (~1 second)
4. **Eager singleton initialization** (all services loaded at startup)

### Solution Implemented: Lazy Initialization

All heavy components now use **lazy initialization** - they only load when first used, not at startup.

#### Changed Components:

1. **TransformersAnalysis** (`src/analysis.py`)
   - Models load only on first `analyze_sentiment()` or `summarize_text()` call
   - Saves ~3-4 seconds at startup
   
2. **MongoManager** (`src/database.py`)
   - Database connection established on first query
   - Saves ~1 second at startup

3. **MultiAPIRetriever** (`src/retriever.py`)
   - API clients (Twitter, Reddit, YouTube, etc.) created on demand
   - Saves ~1 second at startup

4. **Router Services** (`routers.py`)
   - Services instantiated on first API request
   - Uses getter functions instead of global initialization

### Expected Results:

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Startup Time | ~5-6 seconds | **<1 second** | **83-90%** faster |
| First Request | Fast | +3-4 seconds (model load) | One-time cost |
| Subsequent Requests | Fast | Fast | Same performance |
| Memory at Startup | High | **Low** | Deferred allocation |

## 📊 Additional Optimization Recommendations

### 1. Use Model Caching with Persistent Storage

```python
# Add to analysis.py
import os
from transformers import pipeline

CACHE_DIR = "./model_cache"
os.environ['TRANSFORMERS_CACHE'] = CACHE_DIR

self._sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
    model_kwargs={"cache_dir": CACHE_DIR}
)
```

**Benefit:** First load downloads models, subsequent loads are much faster.

### 2. Reduce Model Size (Optional)

Consider using smaller, quantized models for even faster loading:

```python
# Alternative lightweight model
self._sentiment_analyzer = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"  # 3x smaller
)
```

**Trade-off:** Slightly lower accuracy for 3x faster loading.

### 3. Use Async Database Connection Pool

```python
# Update database.py initialization
self._client = AsyncIOMotorClient(
    self._mongo_uri,
    maxPoolSize=10,  # Connection pool
    minPoolSize=2,
    serverSelectionTimeoutMS=5000  # Fail fast
)
```

**Benefit:** Better concurrent request handling.

### 4. Enable FastAPI Lifespan Events (Recommended)

Add warmup on first request automatically:

```python
# Add to main.py
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Do nothing (lazy init)
    logging.info("Server started (services will load on demand)")
    yield
    # Shutdown: Cleanup
    logging.info("Server shutting down")

app = FastAPI(
    title="Sentiment Analysis API",
    lifespan=lifespan
)
```

### 5. Preload Models in Production (Optional)

For production environments where first-request latency matters:

```python
# Add environment variable check
if os.getenv("PRELOAD_MODELS", "false").lower() == "true":
    logging.info("Preloading models for production...")
    _ = get_analysis_pipeline()  # Force initialization
```

**Usage:**
```bash
# Development: Fast startup (default)
uvicorn main:app --reload

# Production: Preload everything
PRELOAD_MODELS=true uvicorn main:app --workers 4
```

### 6. Use Docker Multi-Stage Builds

If using Docker, optimize image:

```dockerfile
# Pre-download models during build
RUN python -c "from transformers import pipeline; \
    pipeline('sentiment-analysis', model='cardiffnlp/twitter-roberta-base-sentiment-latest'); \
    pipeline('summarization', model='facebook/bart-large-cnn')"
```

### 7. Database Indexing

Ensure MongoDB indexes for fast queries:

```python
# Run once after deployment
async def create_indexes():
    await collection.create_index([("query", 1), ("timestamp", -1)])
    await collection.create_index([("timestamp", 1)])
    await collection.create_index([("analysis.sentiment", 1)])
```

### 8. Enable HTTP/2 and Compression

```python
# Add to main.py
from fastapi.middleware.gzip import GZipMiddleware

app.add_middleware(GZipMiddleware, minimum_size=1000)
```

## 🔧 Configuration for Different Environments

### Development (Fast Iteration)
```bash
# Lazy loading (current implementation)
uvicorn main:app --reload --log-level info
```

### Production (Minimize First Request Latency)
```bash
# Preload + multiple workers
PRELOAD_MODELS=true uvicorn main:app --workers 4 --host 0.0.0.0
```

### Testing (No Model Loading)
```python
# Mock the heavy components
@pytest.fixture
def mock_analysis():
    with patch('src.analysis.TransformersAnalysis'):
        yield
```

## 📈 Monitoring Recommendations

Add timing middleware to track performance:

```python
import time
from starlette.middleware.base import BaseHTTPMiddleware

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start = time.time()
        response = await call_next(request)
        duration = time.time() - start
        response.headers["X-Process-Time"] = str(duration)
        logging.info(f"{request.method} {request.url.path} - {duration:.2f}s")
        return response

app.add_middleware(TimingMiddleware)
```

## 🎯 Summary of Changes Made

1. ✅ **TransformersAnalysis**: Lazy model loading with property accessors
2. ✅ **MongoManager**: Lazy database connection
3. ✅ **MultiAPIRetriever**: Lazy API client creation
4. ✅ **Router Services**: Lazy service initialization with getter functions

## 🚦 Testing Your Changes

```bash
# Test startup speed
time uvicorn main:app

# Should see:
# - "API Router initialized (services will load on demand)" immediately
# - No "Loading transformer models..." until first API call
# - Server starts in <1 second

# Test first request (will take longer due to model loading)
curl http://localhost:8000/api/distribution/test

# Test subsequent requests (should be fast)
curl http://localhost:8000/api/distribution/test
```

## 💡 Trade-offs

| Approach | Pros | Cons |
|----------|------|------|
| **Lazy Loading (Current)** | ✅ Fast startup<br>✅ Low memory initially<br>✅ Better dev experience | ⚠️ First request slower |
| **Eager Loading** | ✅ First request fast | ❌ Slow startup<br>❌ High memory always |
| **Hybrid (Recommended for Prod)** | ✅ Balanced | Requires environment config |

## 🔍 Further Optimization Ideas

1. **Use FastAPI Background Tasks** for async processing (already implemented ✅)
2. **Implement request caching** with Redis for repeated queries
3. **Use ONNX Runtime** for faster model inference (advanced)
4. **Implement circuit breakers** for external API failures
5. **Add rate limiting** to protect against abuse

## 📞 Need Help?

If you experience issues:
1. Check logs for initialization timing
2. Monitor first vs subsequent request times
3. Consider preloading for production environments
4. Profile memory usage before/after changes
