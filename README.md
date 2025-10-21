# Real-Time Sentiment Analysis & Competitor Tracking Platform

This application is a powerful, real-time sentiment analysis and market intelligence platform. It automatically gathers public opinion on any topic, product, or brand from a wide range of online sources. Using a cutting-edge Large Language Model (LLM), it analyzes the collected data to provide deep insights into public sentiment, key discussion themes, and competitive positioning.

The entire system is exposed via a clean, robust FastAPI backend, making it easy to integrate with a front-end dashboard for visualization.

---

## Key Features

- **Multi-Source Data Aggregation:** Gathers data from Twitter, Reddit, YouTube, and Google News, and performs deep scrapes of full web articles using Firecrawl.
- **Advanced AI Analysis:** Leverages the high-speed Groq LPU™ Inference Engine to perform detailed, structured sentiment and aspect analysis on every piece of collected text.
- **Real-Time Sentiment Tracking:** Provides endpoints to track sentiment distribution (positive, negative, neutral) and monitor trends over various timeframes (1h, 24h, 7d).
- **Competitor Analysis:** Directly compare sentiment trends between multiple products or brands over time to gauge market positioning and public perception shifts.
- **AI-Generated Summaries:** Automatically generates concise overviews and key insights for both positive and negative feedback, allowing for quick comprehension of public opinion.
- **Dynamic Word Clouds:** Identifies the most frequently discussed topics and aspects related to a query.
- **Robust & Scalable:** Built with a modern Python stack including FastAPI for asynchronous performance and MongoDB for scalable data storage.

---

## System Architecture

The application follows a modular, pipeline-based architecture that ensures a clear separation of concerns, from data collection to analysis and presentation.

```mermaid
graph TD
    subgraph User Interaction
        A[User's Browser/Client]
    end

    subgraph Backend API (FastAPI)
        B[API Endpoints]
        C[Background Analysis Pipeline]
    end

    subgraph Data Collection Layer
        D[Multi-API Retriever]
        E[Twitter, Reddit, YouTube, Google News, Firecrawl]
    end

    subgraph AI & Storage Layer
        F[Groq LLM Analysis Service]
        G[MongoDB Database]
    end

    %% --- Data Flows ---

    A -- 1. POST /api/start-analysis (Body: {query: "Product X"}) --> B
    A -- 2. POST /api/compare (Body: {products: ["X", "Y"], time_range: "7d"}) --> B

    B -- 3. Triggers async task --> C
    C -- 4. Fetches raw text data --> D
    D -- 5. Scrapes content from --> E
    E --> D
    D -- 6. Returns collected text --> C
    C -- 7. Analyzes each text item --> F
    F -- 8. Returns structured JSON analysis --> C
    C -- 9. Stores records in --> G

    A -- 10. GET /api/trends/Product%20X --> B
    B -- 11. Queries processed data from --> G
    G -- 12. Returns aggregated results --> B
    B -- 13. Sends JSON response to Client --> A
```

---

## Technology Stack

- **Backend:** FastAPI  
- **AI/LLM:** Groq (Llama 3)  
- **Database:** MongoDB  
- **Data Validation:** Pydantic  
- **Web Scraping:** Firecrawl, SerpApi  
- **Social Media APIs:** Tweepy (Twitter), AsyncPRAW (Reddit)  
- **Async HTTP:** httpx  
- **Concurrency:** asyncio  

---

## Setup and Installation

### 1. Prerequisites

- Python 3.10+  
- MongoDB instance (local or cloud-hosted, e.g., on MongoDB Atlas)

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd sentiment_analysis_app
```

### 3. Create and Populate the .env File

```bash
cp .env.example .env
```

Now, open the `.env` file and fill in your API keys and the MongoDB connection string:

```bash
# .env
GROQ_API_KEY="gsk_..."
TWITTER_BEARER_TOKEN="your_twitter_bearer_token"
REDDIT_CLIENT_ID="your_reddit_client_id"
REDDIT_CLIENT_SECRET="your_reddit_client_secret"
REDDIT_USER_AGENT="your_reddit_user_agent"
REDDIT_USERNAME="your_reddit_username"
REDDIT_PASSWORD="your_reddit_password"
SERPAPI_API_KEY="your_serpapi_key"
YOUTUBE_API_KEY="your_youtube_api_key"
FIRECRAWL_API_KEY="fc-..."
MONGO_URI="mongodb+srv://..."
```

### 4. Install Dependencies

It's recommended to use a virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`

pip install -r requirements.txt

or

uv init .
uv add -r requiremnts.txt # can use in place of pip for fast.
```

### 5. Run the Application

```bash
uvicorn main:app --reload
or
uv run uvicorn main:app --reload
```

The server will start, and you can access the interactive API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

---

## How to Use

### Start an Analysis
Send a POST request to `/api/start-analysis` with a query. This will trigger a background task to collect and analyze data.

```json
{
  "query": "Openai Sora"
}
```

### Fetch Insights
Once the background task is complete (monitor the server logs), you can use the GET endpoints (`/api/distribution/{query}`, `/api/trends/{query}`, `/api/summary/{query}`, etc.) to retrieve the processed insights.

### Compare Competitors
To compare multiple products, ensure you have already run an analysis for each one. Then, send a POST request to `/api/compare_competitors` with the list of product names.

---

## API Endpoints

(Access the interactive Swagger UI at `/docs` for a full list of endpoints and schemas.)

- **POST /api/start-analysis:** Kicks off a new analysis for a given query.  
- **GET /api/distribution/{query}:** Gets the sentiment distribution (positive/negative/neutral counts).  
- **GET /api/trends/{query}:** Gets sentiment trend data over time.  
- **GET /api/summary/{query}:** Retrieves AI-generated summaries for positive and negative feedback.  
- **GET /api/feed/{query}:** Fetches the latest raw feed items with their analysis.  
- **GET /api/word-cloud/{query}:** Gets data for generating a word cloud.  
- **POST /api/compare_competitors:** Compares sentiment trends for a list of products.
