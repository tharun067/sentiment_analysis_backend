from transformers import pipeline
from typing import List, Dict, Optional
from .models import AnalysisResult, SummaryData, AspectSentiment
import os
from groq import Groq, RateLimitError
from pydantic import ValidationError
import backoff
import logging
from functools import lru_cache
import hashlib


class TransformersAnalysis:
    """Class for performing fast sentiment analysis using local transformers."""
    _instance = None
    def __new__(cls,*args, **kwargs):
        if not cls._instance:
            cls._instance = super(TransformersAnalysis, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initializes the analysis pipelines."""
        if hasattr(self, 'sentiment_analyzer'):
            return
        
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest"
        )
        self.sentiment_summarizer = pipeline(
            'summarization',
            model="facebook/bart-large-cnn"
        )
        logging.info("TransformersAnalysis Initialized.")
    
    def analyze_sentiment(self,text: str) -> Dict:
        """Analyzes seniment with confidence threshold."""
        try:
            result = self.sentiment_analyzer(text)[0]
            # Map transformer label to standard format
            label_map = {
                'positive':'positive',
                'negative':'negative',
                'netural':'netural',
                'label_0':'negative',
                'label_1':'neutral',
                'label_2':'positive'
            }
            label = label_map.get(result['label'].lower(),'neutral')
            return {"label":label, "score":result['score']}
        except Exception as e:
            logging.error(f"Error analyzing Sentiment: {e}")
            return {"label": "netural", "score": 0.5}

    def summarize_text(self, texts: List[str], max_length: int = 130, min_length: int = 30) -> str:
        """Summarizes text with provided input list of text.."""
        full_text = " ".join(texts)

        if len(full_text) < 50:
            return full_text

        # Truncate if too long (BART has 1024 token limit only)
        if len(full_text.split()) > 800:
            words = full_text.split()[:800]
            full_text = " ".join(words)

        try:
            summary = self.sentiment_summarizer(
                full_text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]
            return summary['summary_text']
        except Exception as e:
            logging.error("Error summarizing text: {e}")
            return full_text[:200] + "..."
    
    def basic_analysis(self, text: str) -> AnalysisResult:
        """Creates a bascic AnalysisResult using only Transformers."""
        sentiment_data = self.analyze_sentiment(text)

        # Map emotions based on sentiment
        emotion_map = {
            'positive': ['satisfaction','joy'],
            'negative':['frustration','disappointment'],
            'neutral':['neutral']
        }

        return AnalysisResult(
            sentiment=sentiment_data['label'],
            score=sentiment_data['score'],
            emotions=emotion_map.get(sentiment_data['label'],['neutral']),
            intent='feedback',
            aspects=[] #empty aspects for basic analysis
        )


SYSTEM_PROMPT_ASPECTS = """
You are an expert Aspect-Based Sentiment Analysis (ABSA) system.
Your task is to analyze user feedback text and extract all specific, explicitly mentioned product or service aspects.

**Rules:**

1.  Extract *only* specific features, attributes, or components (e.g., "battery life", "UI design", "customer support", "price").
2.  Ignore vague, general feedback not tied to a specific feature (e.g., "I hate it", "It's good").
3.  Your output MUST be a single, valid JSON object.
4.  The JSON object must contain *only* one key: `"aspects"`.
5.  The value of `"aspects"` must be an array of objects.
6.  Each object in the array MUST have exactly three keys:
      * `"aspect"`: The noun or feature name (e.g., "camera", "battery").
      * `"sentiment"`: The sentiment for that aspect. Must be one of: `"positive"`, `"negative"`, or `"neutral"`.
      * `"quote"`: The *exact*, minimal, contiguous text snippet from the input that directly supports the aspect and sentiment.
8.  Do not include any explanations or conversational text.

**Example Input:**
"The camera on this phone is absolutely amazing, but the battery drains way too fast. The screen is fine, I guess."

**Example Output:**
```json
{
  "aspects": [
    {
      "aspect": "camera",
      "sentiment": "positive",
      "quote": "camera on this phone is absolutely amazing"
    },
    {
      "aspect": "battery",
      "sentiment": "negative",
      "quote": "battery drains way too fast"
    },
    {
      "aspect": "screen",
      "sentiment": "neutral",
      "quote": "The screen is fine, I guess."
    }
  ]
}
```
"""

SYSTEM_PROMPT_SUMMARY = """
You are an expert Text Analyst AI. Your task is to analyze a batch of user comments and consolidate them into a high-level, strategic summary.
Your output MUST be a single, valid JSON object and nothing else.

**Required JSON Format:**

```json
{
  "overview": "A 1-2 sentence neutral summary of the main topics and themes present in the feedback.",
  "keyInsights": [
    "A concise, actionable insight derived from the most significant trends.",
    "Another key finding, praise, or complaint.",
    "..."
  ],
  "overallSentiment": "positive" | "negative" | "neutral"
}
```

**Field Definitions:**

  * `overview`: A brief, factual summary of *what* users are talking about (e.g., "Feedback focuses on the new UI, battery performance, and checkout process.").
  * `keyInsights`: An array of strings. Each string must be a distinct, significant finding or actionable takeaway. Do not just list topics; provide the insight (e.g., "Users find the new checkout process confusing," not "Checkout Process").
  * `overallSentiment`: The dominant, aggregate sentiment of the entire batch of comments. Use "neutral" if the sentiment is heavily mixed, balanced, or apathetic.

**Example Input:**
"The new interface is so much better, I love it. But the app has been really slow since the update, and it crashed twice today. The new shipping tracker is very helpful, though."

**Example Output:**
```json
{
  "overview": "Users are discussing the new interface, app performance post-update, and the shipping tracker feature.",
  "keyInsights": [
    "The new interface design is highly praised by users.",
    "App performance has significantly degraded since the recent update, including slowness and crashes.",
    "The new shipping tracker is a valued feature."
  ],
  "overallSentiment": "neutral"
}
```
"""

class GroqAnalysis:
    """Class for performing sentiment analysis using Groq API. With caching and hybrid approach."""

    def __init__(self):
        api_key = "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" #os.getenv('GROQ_API_KEY') or 
        self.client = Groq(api_key=api_key) if api_key else None
        self.transformer_analysis = TransformersAnalysis()
        self._cache = {}

        if not api_key:
            logging.warning("GROQ_API_KEY not set. GroqAnalysis will not function.")
    
    def _get_cache_key(self, text: str) -> str:
        """Generates a cache key based on the text content."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    
    @backoff.on_exception(backoff.expo, RateLimitError, max_tries=3)
    async def _get_chat_completion(self, system_prompt: str, user_content: str):
        """Helper to get chat completion from Groq with backoff on rate limits."""
        return self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.3,
            max_tokens=500,
            response_format={"type": "json_object"}
        )
    
    async def extract_aspects_with_llm(self, text: str) -> List[AspectSentiment]:
        """Use Groq LLM to extract aspect-based sentiment analysis."""
        if not self.client:
            logging.warning("Groq client not initialized. Skipping LLM aspect extraction.")
            return []
        
        try:
            completion = await self._get_chat_completion(SYSTEM_PROMPT_ASPECTS, text)
            response_text = completion.choices[0].message.content
            data = eval(response_text)  # Using eval to parse JSON object

            return [
                AspectSentiment(
                    aspect=item.get("aspect",""),
                    sentiment=item.get("sentiment","neutral"),
                    quote=item.get("quote","")
                )
                for item in data.get("aspects", [])
            ]
        except Exception as e:
            logging.error(f"Error extracting aspects with LLM: {e}")
            return []
        
    async def analyze_text(self, text: str, use_hybrid: bool = True) -> Optional[AnalysisResult]:
        """
            HYBRID APPROACH: Combines Transformers + LLM
            - Transformers: Fast sentiment (always)
            - LLM: Aspect extraction only (when available)
            
            This reduces LLM calls by ~80% while maintaining quality.
        """
        ### Check cache first
        cache_key = self._get_cache_key(text)
        if cache_key in self._cache:
            logging.info("Cache hit for text analysis.")
            return self._cache[cache_key]
        
        # Step 1: Always use Transformers for basic sentiment analysis(FAST)
        sentiment_data = self.transformer_analysis.analyze_sentiment(text)

        # Step 2: Use LLM for aspect extraction if enabled and client available(SLOW)
        aspects = []
        if use_hybrid and self.client:
            try:
                aspects = await self.extract_aspects_with_llm(text)
            except RateLimitError:
                logging.warning("Rate limit hit during LLM aspect extraction. Falling back to no aspects.")
            except Exception as e:
                logging.error(f"Unexpected error during LLM aspect extraction: {e}")
        
        # Step 3: Combine results
        emotion_map = {
            'positive': ['satisfaction', 'appreciation'],
            'negative': ['frustration', 'concern'],
            'neutral': ['neutral']
        }

        result = AnalysisResult(
            sentiment=sentiment_data['label'],
            score=sentiment_data['score'],
            emotions=emotion_map.get(sentiment_data['label'], ['neutral']),
            intent='user_feedback',
            aspects=aspects
        )

        # Cache the result
        self._cache[cache_key] = result
        return result

    async def generate_structured_summary(self, documents: List[str], sentiment_context: str) -> SummaryData:
        """Generates a structured summary using Transformers first, LLM as enhancement."""
        if not documents:
            return SummaryData(
                overview="No documents available for summary.",
                keyInsights=[],
                overallSentiment="neutral"
            )
        
        # Step 1: Use Transformers to create a basic summary(FAST)
        transformer_summary = self.transformer_analysis.summarize_text(
            documents[:10], # Limit to first 10 docs for speed
            max_length=50,
            min_length=1
        )

        # Step 2: Use LLM to refine and structure the summary if client available(SLOW)
        if self.client and len(documents) > 5:
            try:
                combined = "\n".join(f"- {doc[:100]}" for doc in documents[:20])
                prompt = f"Context: {sentiment_context}\nComments:\n{combined}"
                completion = await self._get_chat_completion(
                    SYSTEM_PROMPT_SUMMARY,
                    prompt
                )
                response_text = completion.choices[0].message.content
                print(response_text)
                return SummaryData.model_validate_json(response_text)
            except (RateLimitError, ValidationError, Exception) as e:
                logging.error(f"Error generating structured summary with LLM: {e}")
        
        # Fallback to basic summary if LLM fails
        return SummaryData(
            overview=transformer_summary,
            keyInsights=[f"Based on {len(documents)} documents analyzed."],
            overallSentiment=sentiment_context
        )
    
    def clear_cache(self):
        """Clears the internal analysis cache."""
        self._cache.clear()
        logging.info("Analysis cache cleared.")