# Sentiment Analysis API

A comprehensive FastAPI-based backend service for real-time sentiment analysis of web content from multiple sources including Twitter, Reddit, YouTube, and Google News. The system uses a hybrid approach combining fast local Transformers models with LLM-powered aspect-based sentiment analysis for optimal performance and accuracy.

## 🚀 Features

### Core Capabilities
- **Multi-Source Data Retrieval**: Aggregates content from Twitter, Reddit, YouTube, and Google News
- **Hybrid Sentiment Analysis**: Smart combination of Transformers (fast) and LLM (detailed)
- **Aspect-Based Sentiment Analysis**: Extracts and analyzes specific product features (camera, battery, UI, etc.)
- **Real-time Processing**: Background task execution with MongoDB persistence
- **Competitor Comparison**: Compare sentiment trends across multiple products
- **Intelligent Summarization**: AI-generated insights from positive and negative feedback
- **Word Cloud Generation**: Visualize frequently mentioned aspects
- **Trend Analysis**: Time-series sentiment data with configurable time ranges

### Analysis Modes
1. **Hybrid Mode** (Recommended): 70-80% cost reduction by combining Transformers + selective LLM
2. **Transformers Mode**: Fast, local analysis only - no rate limits, no aspect extraction
3. **LLM Mode**: Full LLM analysis - detailed but may hit rate limits

## 📋 Table of Contents
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [API Endpoints](#-api-endpoints)
- [Usage Examples](#-usage-examples)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Deployment](#-deployment)
- [Performance Optimization](#-performance-optimization)
- [Troubleshooting](#-troubleshooting)

## 🏗 Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Application                     │
│                     (main.py, routers.py)                    │
└────────────────────┬────────────────────────────────────────┘
                     │
        ┌────────────┼────────────┐
        │            │            │
   ┌────▼────┐  ┌───▼────┐  ┌───▼──────┐
   │ Retriever│  │Analysis│  │ Database │
   │  Layer   │  │ Layer  │  │  Layer   │
   └────┬─────┘  └───┬────┘  └────┬─────┘
        │            │             │
   Multi-API    Hybrid AI      MongoDB
   Scraping    (Transformers    Storage
               + Groq LLM)
```

### Data Flow
1. **Retrieval**: MultiAPIRetriever fetches data from multiple sources concurrently
2. **Analysis**: AnalysisPipeline processes data using hybrid strategy
3. **Storage**: MongoManager persists analyzed results with timestamps
4. **Serving**: API endpoints provide various views and aggregations

## 🔧 Installation

### Prerequisites
- Python 3.11+
- MongoDB instance (local or cloud)
- API keys for data sources (see [Configuration](#-configuration))

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd sentiment_analysis_backend
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create `.env` file** in the project root:
```env
# MongoDB
MONGO_URI=mongodb://localhost:27017/

# LLM API
GROQ_API_KEY=your_groq_api_key

# Twitter
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# Reddit
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=your_reddit_user_agent
REDDIT_USERNAME=your_reddit_username
REDDIT_PASSWORD=your_reddit_password

# Google Services
YOUTUBE_API_KEY=your_youtube_api_key
SERPAPI_API_KEY=your_serpapi_key

# Web Scraping
FIRECRAWL_API_KEY=your_firecrawl_api_key
```

4. **Run the application**
```bash
uvicorn main:app --host 0.0.0.0 --port 8080 --reload
```

5. **Access API documentation**
   - Swagger UI: `http://localhost:8080/docs`
   - ReDoc: `http://localhost:8080/redoc`

## ⚙ Configuration

### API Keys Required

| Service | Purpose | Get API Key |
|---------|---------|-------------|
| MongoDB | Data persistence | [MongoDB Atlas](https://www.mongodb.com/cloud/atlas) |
| Groq | LLM analysis | [Groq Console](https://console.groq.com/) |
| Twitter | Tweet retrieval | [Twitter Developer Portal](https://developer.twitter.com/) |
| Reddit | Reddit post/comment retrieval | [Reddit Apps](https://www.reddit.com/prefs/apps) |
| YouTube | Comment retrieval | [Google Cloud Console](https://console.cloud.google.com/) |
| SerpAPI | Google News scraping | [SerpAPI](https://serpapi.com/) |
| Firecrawl | Web content scraping | [Firecrawl](https://firecrawl.dev/) |

### Optional Configuration

All services are optional and will gracefully skip if credentials are not provided. The system will log warnings but continue functioning with available services.

## 📡 API Endpoints

### Core Analysis Endpoints

#### `POST /api/start_analysis`
Starts background sentiment analysis for a query.

**Request Body:**
```json
{
  "query": "iPhone 15"
}
```

**Query Parameters:**
- `mode`: Analysis mode (`hybrid` | `transformers` | `llm`) - default: `hybrid`

**Response:**
```json
{
  "status": "success",
  "query": "iPhone 15",
  "message": "Analysis has been started and is running in the background."
}
```

#### `POST /api/compare_competitors`
Compare sentiment trends for multiple products.

**Request Body:**
```json
{
  "Products": ["iPhone 15", "Samsung Galaxy S24", "Google Pixel 8"],
  "time_range": "24h"
}
```

**Response:**
```json
{
  "comparison": [
    {
      "product_name": "iPhone 15",
      "trends": [
        {
          "timestamp": "2025-12-15T10:00:00Z",
          "positive": 45,
          "negative": 12,
          "neutral": 8
        }
      ]
    }
  ]
}
```

### Data Retrieval Endpoints

#### `GET /api/distribution/{query}`
Get sentiment distribution (positive/negative/neutral counts).

**Query Parameters:**
- `time_range`: `1h` | `24h` | `7d` (default: `24h`)

**Response:**
```json
{
  "positive": 150,
  "negative": 45,
  "neutral": 30
}
```

#### `GET /api/trends/{query}`
Get time-series sentiment trend data.

**Query Parameters:**
- `time_range`: `1h` | `24h` | `7d` (default: `24h`)

**Response:**
```json
[
  {
    "timestamp": "2025-12-15T10:00:00Z",
    "positive": 45,
    "negative": 12,
    "neutral": 8
  }
]
```

#### `GET /api/summary/{query}`
Generate AI-powered summaries of positive and negative feedback.

**Query Parameters:**
- `sample_size`: 5-100 documents (default: 25)
- `time_range`: `1h` | `24h` | `7d` (default: `24h`)

**Response:**
```json
{
  "positive_summary": {
    "overview": "Users praise the camera quality and battery life",
    "keyInsights": [
      "Camera performance exceeds expectations",
      "Battery lasts all day with heavy use"
    ],
    "overallSentiment": "positive"
  },
  "negative_summary": {
    "overview": "Complaints focus on price and software bugs",
    "keyInsights": [
      "Price considered too high compared to competitors",
      "Software updates cause occasional crashes"
    ],
    "overallSentiment": "negative"
  }
}
```

#### `GET /api/feed/{query}`
Get recent feed items with sentiment scores.

**Query Parameters:**
- `limit`: 1-500 items (default: 50)

**Response:**
```json
[
  {
    "_id": "unique-id",
    "text": "The new iPhone camera is amazing!",
    "sentiment": "positive",
    "score": 0.95,
    "timestamp": "2025-12-15T12:00:00Z",
    "source": "Twitter",
    "query": "iPhone 15"
  }
]
```

#### `GET /api/wordcloud/{query}`
Get word cloud data from extracted aspects.

**Query Parameters:**
- `time_range`: `1h` | `24h` | `7d` (default: `24h`)

**Response:**
```json
[
  {"text": "camera", "value": 45},
  {"text": "battery", "value": 32},
  {"text": "screen", "value": 28}
]
```

### Administrative Endpoints

#### `POST /api/delete_data/{query}`
Delete old data to maintain database performance.

**Query Parameters:**
- `days`: Delete data older than N days (1-365, default: 30)

## 💡 Usage Examples

### Basic Sentiment Analysis

```python
import requests

# Start analysis
response = requests.post(
    "http://localhost:8080/api/start_analysis",
    json={"query": "iPhone 15"},
    params={"mode": "hybrid"}
)

# Wait for processing (runs in background)
# Then retrieve results

# Get sentiment distribution
distribution = requests.get(
    "http://localhost:8080/api/distribution/iPhone 15",
    params={"time_range": "24h"}
).json()

print(f"Positive: {distribution['positive']}")
print(f"Negative: {distribution['negative']}")
print(f"Neutral: {distribution['neutral']}")
```

### Competitor Comparison

```python
# Compare products
comparison = requests.post(
    "http://localhost:8080/api/compare_competitors",
    json={
        "Products": ["iPhone 15", "Samsung Galaxy S24"],
        "time_range": "7d"
    }
).json()

for product in comparison['comparison']:
    print(f"Product: {product['product_name']}")
    for trend in product['trends']:
        print(f"  {trend['timestamp']}: +{trend['positive']} -{trend['negative']}")
```

### Get AI Summary

```python
# Get intelligent summary
summary = requests.get(
    "http://localhost:8080/api/summary/iPhone 15",
    params={"sample_size": 50, "time_range": "24h"}
).json()

print("Positive Insights:")
for insight in summary['positive_summary']['keyInsights']:
    print(f"  - {insight}")

print("\nNegative Insights:")
for insight in summary['negative_summary']['keyInsights']:
    print(f"  - {insight}")
```

## 📁 Project Structure

```
sentiment_analysis_backend/
│
├── main.py                      # FastAPI application entry point
├── routers.py                   # API route definitions
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Docker configuration
├── README.md                    # This file
│
├── src/
│   ├── analysis.py              # Sentiment analysis engines
│   │   ├── TransformersAnalysis # Local transformer models
│   │   └── GroqAnalysis         # LLM-based analysis
│   │
│   ├── database.py              # MongoDB operations
│   │   └── MongoManager         # Database manager singleton
│   │
│   ├── retriever.py             # Multi-source data retrieval
│   │   ├── TwitterScraper       # Twitter API integration
│   │   ├── RedditScraper        # Reddit API integration
│   │   ├── YouTubeScraper       # YouTube API integration
│   │   ├── GoogleNewsScraper    # Google News via SerpAPI
│   │   └── MultiAPIRetriever    # Orchestrator
│   │
│   ├── firecrawl_retriever.py  # Web content scraper
│   │   └── FirecrawlScraper     # Firecrawl API wrapper
│   │
│   ├── services.py              # Business logic layer
│   │   └── AnalysisPipeline     # Main processing pipeline
│   │
│   └── models.py                # Pydantic data models
│       ├── API Request/Response # API schemas
│       ├── Database Models      # MongoDB schemas
│       └── Analysis Models      # LLM output schemas
```

## 🛠 Technologies Used

### Backend Framework
- **FastAPI**: Modern, high-performance web framework
- **Uvicorn**: ASGI server for async operations
- **Pydantic**: Data validation and serialization

### AI/ML Models
- **Transformers** (Hugging Face):
  - `cardiffnlp/twitter-roberta-base-sentiment-latest` - Sentiment classification
  - `facebook/bart-large-cnn` - Text summarization
- **Groq API**: LLM-powered aspect extraction and insights
  - Model: `llama-3.3-70b-versatile`

### Database
- **MongoDB**: Document storage with Motor (async driver)

### Data Sources
- **Twitter API**: Recent tweets
- **Reddit API** (PRAW): Posts and comments
- **YouTube API**: Video comments
- **SerpAPI**: Google News results
- **Firecrawl**: Web content extraction

### Key Libraries
- `motor`: Async MongoDB driver
- `langchain`: LLM orchestration
- `backoff`: Exponential backoff for rate limiting
- `asyncio`: Concurrent processing
- `python-dotenv`: Environment management

## 🐳 Deployment

### Docker Deployment

1. **Build the image**
```bash
docker build -t sentiment-analysis-api .
```

2. **Run the container**
```bash
docker run -d \
  --name sentiment-api \
  -p 8080:8080 \
  --env-file .env \
  sentiment-analysis-api
```

### Docker Compose

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "8080:8080"
    env_file:
      - .env
    depends_on:
      - mongodb
    restart: unless-stopped

  mongodb:
    image: mongo:7
    ports:
      - "27017:27017"
    volumes:
      - mongo-data:/data/db
    restart: unless-stopped

volumes:
  mongo-data:
```

### Production Considerations

1. **Environment Variables**: Use secrets management (AWS Secrets Manager, Azure Key Vault)
2. **Scaling**: Deploy behind load balancer with multiple instances
3. **Monitoring**: Integrate with logging services (Datadog, New Relic)
4. **Rate Limiting**: Implement API rate limiting for public endpoints
5. **CORS**: Configure specific origins instead of wildcard `*`

## ⚡ Performance Optimization

### Hybrid Analysis Strategy

The system intelligently chooses between fast local models and detailed LLM analysis:

**When LLM is Used:**
- Text length > 100 characters
- Every 3rd item for representative sampling
- Texts containing aspect keywords (battery, camera, price, etc.)

**Benefits:**
- 70-80% reduction in LLM API calls
- 5-10x faster processing
- Maintains high-quality aspect extraction

### Lazy Initialization

All heavy components (models, API clients, database) use lazy initialization:
- Transformers models load on first analysis
- API clients connect on first request
- MongoDB connection on first query

**Benefit:** Fast startup time, reduced memory footprint

### Caching

- **Analysis Cache**: SHA-256 based text caching prevents duplicate LLM calls
- **Connection Pooling**: MongoDB connection reuse
- **Singleton Pattern**: Single instances of analyzers and database managers

### Concurrent Processing

- **Async Operations**: All I/O operations use asyncio
- **Parallel Retrieval**: Multiple data sources fetched concurrently
- **Semaphore Control**: Max 5 concurrent LLM calls to avoid rate limits
- **Background Tasks**: Analysis runs without blocking API responses

## 🔍 Troubleshooting

### Common Issues

#### 1. MongoDB Connection Error
```
ValueError: MongoDB URI environment variable not set.
```
**Solution:** Ensure `MONGO_URI` is set in `.env` file

#### 2. Rate Limit Errors (Groq/Twitter)
```
groq.RateLimitError: Rate limit exceeded
```
**Solution:** 
- Use `transformers` mode for unlimited processing
- Reduce concurrent LLM calls (adjust `max_concurrent_llm`)
- Wait for rate limit reset

#### 3. No Data Retrieved
```
WARNING: No items retrieved.
```
**Solution:**
- Check API keys are valid
- Verify network connectivity
- Check API quotas haven't been exceeded
- Try different query terms

#### 4. Transformer Model Download
Models auto-download on first use. Ensure internet connectivity.

#### 5. Empty Aspects in Results
Aspects are only extracted in `hybrid` or `llm` modes. `transformers` mode doesn't extract aspects.

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Health Check

Check service status:
```bash
curl http://localhost:8080/
```

Expected response:
```json
{
  "message": "Welcome to the Sentiment Analysis API. Visit /docs for documentation."
}
```

## 📊 Database Schema

### Collection: `feed_items`

```javascript
{
  "_id": "uuid",
  "query": "string",
  "text": "string",
  "source": "Twitter|Reddit|YouTube|Google News|WebApp",
  "timestamp": ISODate,
  "analysis": {
    "sentiment": "positive|negative|neutral",
    "score": 0.95,
    "emotions": ["satisfaction", "joy"],
    "intent": "user_feedback",
    "aspects": [
      {
        "aspect": "camera",
        "sentiment": "positive",
        "quote": "The camera is absolutely amazing"
      }
    ]
  }
}
```

### Indexes (Recommended)

```javascript
db.feed_items.createIndex({ "query": 1, "timestamp": -1 })
db.feed_items.createIndex({ "timestamp": 1 })
db.feed_items.createIndex({ "query": 1, "analysis.sentiment": 1 })
```

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 👨‍💻 Support

For issues and questions:
- Open an issue on GitHub
- Check existing documentation at `/docs`
- Review logs for detailed error messages

## 🔮 Future Enhancements

- [ ] Real-time WebSocket support for live updates
- [ ] Multi-language sentiment analysis
- [ ] Custom model fine-tuning interface
- [ ] Advanced visualization dashboard
- [ ] Sentiment prediction/forecasting
- [ ] Export functionality (CSV, JSON, PDF reports)
- [ ] User authentication and API keys
- [ ] Webhook notifications for analysis completion

---

**Built with ❤️ using FastAPI, Transformers, and Groq LLM**