from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import router
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(
    title="Sentiment Analysis API",
    description="An API to retrieve, analyze, and serve sentiment data from various web sources.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins
    allow_credentials=True,
    allow_methods=["*"], # Allow all methods(GET, POST, etc.)
    allow_headers=["*"], # Allow all headers
)

## API Router
# Include the router from the routers module
app.include_router(router)

@app.get("/", tags=["Root"])
async def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    return {"message": "Welcome to the Sentiment Analysis API. Visit /docs for documentation."}
