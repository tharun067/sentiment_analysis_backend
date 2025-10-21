import asyncio
from typing import List, Dict, Any, Optional
from .retriever import MultiAPIRetriever
from .analysis import TransformersAnalysis, GroqAnalysis
from .database import MongoManager
from .models import SentimentRecord
from datetime import datetime, timezone
from collections import defaultdict
import logging

class AnalysisPipeline:
    """
    Hybrid sentiment analysis pipeline that intelligently uses:
    - Transformers: For fast, local sentiment analysis (primary)
    - LLM: For detailed aspect extraction (secondary, rate-limited)
    """

    def __init__(self, max_concurrent_llm: int = 5):
        self.retriever = MultiAPIRetriever()
        self.transformers_analyzer = TransformersAnalysis()
        self.groq_analyzer = GroqAnalysis()
        self.db_manager = MongoManager()

        # Reduced conncurrency for LLM to avoid rate limits
        self.analysis_semaphore = asyncio.Semaphore(max_concurrent_llm)

        # Track LLM usage
        self.llm_call_count = 0
        self.transformer_call_count = 0

        logging.info(f"AnalysisPipeline initialized with max {max_concurrent_llm} concurrent LLM calls.")
    
    def _should_use_llm(self, text: str, index: int, total: int) -> bool:
        """
        Intelligent decision: when to use expensive LLM vs fast Transformers.
        
        Use LLM only for:
        - Longer texts (>100 chars) that likely have aspects
        - Representative sample (e.g., every 3rd item)
        - High-priority items"""
        # Always use Transformers for short texts
        if len(text) < 100:
            return False
        
        # Use LLM for every Nth item to get a representative sample
        if index % 3 == 0:
            return True
        
        # Use LLM if text contains product-specific keywords (example logic)
        aspect_keywords = [
            "battery", "screen", "performance", "design", "price", "quality",
            "durability", "camera", "software", "features", "support"
        ]
        if any(keyword in text.lower() for keyword in aspect_keywords):
            return True
        
        return False
    
    async def _analyze_with_hybrid_strategy(self, item: dict, query: str, index: int, total: int) -> Optional[SentimentRecord]:
        """
        Strategy:
        1. Quick sentiment check with Transformers (always)
        2. Aspect extraction with LLM (selectively)
        3. Fallback to Transformers-only if LLM fails
        """

        text  = item.get('text', '')
        if not text:
            return None
        
        # Decide whether to use LLM for this item
        use_llm = self._should_use_llm(text, index, total)

        if use_llm:
            # Try hybrid approach
            async with self.analysis_semaphore:
                try:
                    analysis_result = await self.groq_analyzer.analyze_text(
                        text=text,
                        use_hybrid=True
                    )
                    self.llm_call_count += 1

                    if analysis_result:
                        return SentimentRecord(
                            query=query,
                            text=text,
                            source=item.get('source', 'unknown'),
                            timestamp=item.get('timestamp', datetime.now(timezone.utc)),
                            analysis=analysis_result
                        )
                except Exception as e:
                    logging.error(f"Hybrid analysis failed: {e}, falling back to Transformers.")
        # Fallback to Transformers-only analysis
        try:
            analysis_result = self.transformers_analyzer.basic_analysis(text)
            self.transformer_call_count += 1

            return SentimentRecord(
                query=query,
                text=text,
                source=item.get('source', 'unknown'),
                timestamp=item.get('timestamp', datetime.now(timezone.utc)),
                analysis=analysis_result
            )
        except Exception as e:
            logging.error(f"Transformers analysis failed: {e}")
            return None
    
    async def _batch_analyze_transformers(self, items: List[dict], query: str) -> List[SentimentRecord]:
        """
        Batch analyze a list of items using only Transformers for speed.
        Use this for simple sentiment analysis without aspects.
        """
        records = []
        for item in items:
            text = item.get('text', '')
            if not text:
                continue
            try:
                analysis = self.transformers_analyzer.basic_analysis(text)
                record = SentimentRecord(
                    query=query,
                    text=text,
                    source=item.get('source', 'unknown'),
                    timestamp=item.get('timestamp', datetime.now(timezone.utc)),
                    analysis=analysis
                )
                records.append(record)
                self.transformer_call_count += 1
            except Exception as e:
                logging.error(f"Transformers batch analysis failed: {e}")
        return records
    
    async def run(self, query: str, mode: str = 'hybrid') -> Dict[str, Any]:
        """
        xecute the pipeline with configurable modes:
        
        - 'hybrid': Smart mix of Transformers + LLM (recommended)
        - 'transformers': Transformers only (fastest, no rate limits)
        - 'llm': LLM for all (slowest, may hit rate limits)
        
        Returns statistics about the run.
        """
        logging.info(f"Starting analysis pipeline in '{mode}' mode for query: {query}")

        # Reset counters
        self.llm_call_count = 0
        self.transformer_call_count = 0

        # 1. Retrieve data
        retrieved_items = await self.retriever.retrieve(query=query)
        if not retrieved_items:
            logging.warning("No items retrieved.")
            return self._build_stats(0,0,mode)
        logging.info(f"Retrieved {len(retrieved_items)} items.")

        # 2. Analyze data
        if mode == 'transformers':
            successful_records = await self._batch_analyze_transformers(retrieved_items, query)
        
        elif mode == 'llm':
            tasks = [
                self._analyze_with_hybrid_strategy(item, query, idx, len(retrieved_items))
                for idx, item in enumerate(retrieved_items)
            ]
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_records = [
                res for res in analysis_results if isinstance(res, SentimentRecord)
            ]
        else: # hybrid (default)
            tasks = [
                self._analyze_with_hybrid_strategy(item, query, idx, len(retrieved_items))
                for idx, item in enumerate(retrieved_items)
            ]
            analysis_results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_records = [
                res for res in analysis_results if isinstance(res, SentimentRecord)
            ]
        # 3. Save to database
        if successful_records:
            await self.db_manager.save_feed_item(successful_records)
            logging.info(f"Pipeline completed: {len(successful_records)}/{len(retrieved_items)} records saved.")
        else:
            logging.warning("No items could be analyzed successfully.")

    def _build_stats(self, saved: int, total: int, mode: str) -> Dict[str, Any]:
        """Build statistics report."""
        return {
            'query': 'N/A',
            'mode': mode,
            'items_retrieved': total,
            'items_saved': saved,
            'success_rate': (saved / total * 100) if total > 0 else 0,
            'llm_calls': self.llm_call_count,
            'transformer_calls': self.transformer_call_count,
            'llm_reduction': (
                (1 - self.llm_call_count / total) * 100 
                if total > 0 else 0
            )
        }
    
    async def run_with_summary(self, query: str, mode: str = 'hybrid') -> Dict[str, Any]:
        """
        Run the pipeline and return a summary of results.
        """
        stats = await self.run(query=query, mode=mode)
        
        # Fetch recent results for summary
        records = await self.db_manager.get_recent_feed(query=query, limit=50)

        if records:
            sentiment_groups = defaultdict(list)
            for record in records:
                sentiment = record.analysis.seniment
                sentiment_groups[sentiment].append(record.text)

            summaries = {}
            for sentiment, texts in sentiment_groups.items():
                summary = await self.groq_analyzer.generate_structured_summary(
                    texts=texts,
                    sentiment=sentiment
                )
                summaries[sentiment] = summary
            stats['summaries'] = summaries
        logging.info("Summary generation completed.")
        return stats
