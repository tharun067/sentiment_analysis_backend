from firecrawl import FirecrawlApp
from typing import List, Dict, Any
from datetime import datetime, timezone
import os
import logging

class FirecrawlScraper:
    """Scrapes web content using the Firecrawl API."""
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Firecrawl API key is not provided.")
        self.app = FirecrawlApp(api_key=api_key)
        logging.info("FirecrawlScraper initialized with provided API key.")
    
    def scrape(self, url: str) -> List[Dict[str, Any]]:
        """Scrapes a single URL and returns the main content in the standard format.
        
        Args:
            url (str): The URL to scrape.
            
        Returns:
            List[Dict[str, Any]]: A list containing one scraped item dictionary.
        """
        logging.info(f"Scraping URL via Firecrawl: {url}")
        try:
            scrapped_data = self.app.scrape(url)
            if 'markdown' in scrapped_data and scrapped_data['markdown'].strip():
                return [{
                    "text": scrapped_data['markdown'],
                    "source": "WebApp",
                    "timestamp": datetime.now(timezone.utc),
                    "url": url
                }]
            return []
        except Exception as e:
            logging.error(f"Error scraping URL {url} via Firecrawl: {e}")
            return []


# Example usage (for testing)
if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    
    scraper = FirecrawlScraper(api_key=os.getenv('FIRECRAWL_API_KEY'))
    results = scraper.scrape(url='https://blog.google/technology/ai/google-gemini-ai/')
    if results:
        print(results[0]['text'][:500])
        