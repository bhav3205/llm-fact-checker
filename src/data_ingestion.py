# """
# Data Ingestion Module
# Fetches and parses data from PIB (Press Information Bureau) RSS feeds
# """

# import feedparser
# import requests
# from typing import List, Dict
# from datetime import datetime
# import logging
# import time

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# class PIBDataIngestion:
#     """
#     Handles data ingestion from Press Information Bureau RSS feeds.
#     """
    
#     PIB_RSS_URL = "https://www.pib.gov.in/ViewRss.aspx"
    
#     def __init__(self):
#         """Initialize PIB data ingestion."""
#         self.session = requests.Session()
#         self.session.headers.update({
#             'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
#         })
    
#     def fetch_pib_statements(self, max_items: int = 50) -> List[Dict]:
#         """
#         Fetch latest statements from PIB RSS feed.
        
#         Args:
#             max_items: Maximum number of items to fetch
            
#         Returns:
#             List of statement dictionaries
#         """
#         try:
#             logger.info(f"Fetching PIB RSS feed from {self.PIB_RSS_URL}")
            
#             # Add timeout and retry logic
#             for attempt in range(3):
#                 try:
#                     feed = feedparser.parse(self.PIB_RSS_URL)
#                     break
#                 except Exception as e:
#                     if attempt == 2:
#                         raise e
#                     logger.warning(f"Attempt {attempt + 1} failed, retrying...")
#                     time.sleep(2)
            
#             statements = []
            
#             for entry in feed.entries[:max_items]:
#                 statement = {
#                     "title": entry.get('title', ''),
#                     "description": entry.get('summary', ''),
#                     "link": entry.get('link', ''),
#                     "published": entry.get('published', ''),
#                     "source": "PIB India",
#                     "text": f"{entry.get('title', '')}. {entry.get('summary', '')}"
#                 }
#                 statements.append(statement)
            
#             logger.info(f"Successfully fetched {len(statements)} statements from PIB")
#             return statements
            
#         except Exception as e:
#             logger.error(f"Error fetching PIB data: {str(e)}")
#             logger.info("Using fallback sample data...")
#             return self._get_fallback_data()
    
#     def _get_fallback_data(self) -> List[Dict]:
#         """Fallback data in case PIB fetch fails"""
#         return [
#             {
#                 "title": "PM-KISAN Scheme",
#                 "summary": "Under PM-KISAN, financial assistance of Rs. 6000 per year is provided to farmer families in three equal installments of Rs. 2000 each.",
#                 "link": "https://pib.gov.in",
#                 "published": "2024-01-15",
#                 "source": "PIB India"
#             },
#             {
#                 "title": "Digital India Initiative",
#                 "summary": "The Digital India programme aims to transform India into a digitally empowered society and knowledge economy with broadband highways.",
#                 "link": "https://pib.gov.in",
#                 "published": "2024-02-10",
#                 "source": "PIB India"
#             },
#             {
#                 "title": "Ayushman Bharat Scheme",
#                 "summary": "Ayushman Bharat provides health coverage of Rs. 5 lakh per family per year for secondary and tertiary care hospitalization to over 10 crore vulnerable families.",
#                 "link": "https://pib.gov.in",
#                 "published": "2024-03-20",
#                 "source": "PIB India"
#             },
#             {
#                 "title": "Make in India Program",
#                 "summary": "Make in India initiative was launched to facilitate investment, foster innovation, enhance skill development and build best-in-class manufacturing infrastructure.",
#                 "link": "https://pib.gov.in",
#                 "published": "2024-04-05",
#                 "source": "PIB India"
#             },
#             {
#                 "title": "Swachh Bharat Mission",
#                 "summary": "Swachh Bharat Mission aims to achieve universal sanitation coverage and has successfully constructed over 10 crore toilets across India.",
#                 "link": "https://pib.gov.in",
#                 "published": "2024-05-12",
#                 "source": "PIB India"
#             }
#         ]
    
#     def prepare_fact_base(self, statements: List[Dict]) -> List[Dict]:
#         """
#         Prepare statements for embedding and storage.
        
#         Args:
#             statements: List of raw statements
            
#         Returns:
#             List of processed fact dictionaries
#         """
#         facts = []
        
#         for i, stmt in enumerate(statements):
#             fact = {
#                 "id": f"pib_{i}",
#                 "text": stmt['text'],
#                 "title": stmt['title'],
#                 "source": stmt['source'],
#                 "published": stmt['published'],
#                 "url": stmt.get('link', '')
#             }
#             facts.append(fact)
        
#         logger.info(f"Prepared {len(facts)} facts for storage")
#         return facts

"""
Data Ingestion Module
Fetches and parses data from PIB (Press Information Bureau) RSS feeds
"""

import feedparser
import requests
from typing import List, Dict
from datetime import datetime
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PIBDataIngestion:
    """
    Handles data ingestion from Press Information Bureau RSS feeds.
    """
    
    PIB_RSS_URL = "https://www.pib.gov.in/ViewRss.aspx"
    
    def __init__(self):
        """Initialize PIB data ingestion."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_pib_statements(self, max_items: int = 50) -> List[Dict]:
        """
        Fetch latest statements from PIB RSS feed.
        Falls back to comprehensive sample data if fetch fails.
        
        Args:
            max_items: Maximum number of items to fetch
            
        Returns:
            List of statement dictionaries
        """
        try:
            logger.info(f"Fetching PIB RSS feed from {self.PIB_RSS_URL}")
            
            # Try to fetch with timeout and retries
            for attempt in range(3):
                try:
                    feed = feedparser.parse(self.PIB_RSS_URL)
                    if hasattr(feed, 'entries') and len(feed.entries) > 0:
                        break
                except Exception as e:
                    if attempt == 2:
                        raise e
                    logger.warning(f"Attempt {attempt + 1} failed, retrying...")
                    time.sleep(2)
            
            statements = []
            
            if hasattr(feed, 'entries') and len(feed.entries) > 0:
                for entry in feed.entries[:max_items]:
                    statement = {
                        "title": entry.get('title', ''),
                        "summary": entry.get('summary', entry.get('description', '')),
                        "link": entry.get('link', ''),
                        "published": entry.get('published', ''),
                        "source": "PIB India",
                        "text": f"{entry.get('title', '')}. {entry.get('summary', entry.get('description', ''))}"
                    }
                    statements.append(statement)
                
                logger.info(f"Successfully fetched {len(statements)} statements from PIB")
                return statements
            else:
                logger.warning("PIB feed returned no entries. Using fallback data.")
                return self._get_fallback_data()
            
        except Exception as e:
            logger.error(f"Error fetching PIB data: {str(e)}")
            logger.info("Using fallback sample data...")
            return self._get_fallback_data()
    
    def _get_fallback_data(self) -> List[Dict]:
        """Comprehensive fallback data for testing"""
        logger.info("Using fallback government scheme data")
        return [
            {
                "title": "PM-KISAN Scheme",
                "summary": "Under PM-KISAN, financial assistance of Rs. 6000 per year is provided to farmer families in three equal installments of Rs. 2000 each. The scheme covers over 11 crore farmers across India.",
                "link": "https://pib.gov.in",
                "published": "2024-01-15",
                "source": "PIB India"
            },
            {
                "title": "Digital India Initiative",
                "summary": "The Digital India programme aims to transform India into a digitally empowered society and knowledge economy with broadband highways, universal digital literacy, and digital delivery of services.",
                "link": "https://pib.gov.in",
                "published": "2024-02-10",
                "source": "PIB India"
            },
            {
                "title": "Ayushman Bharat Scheme",
                "summary": "Ayushman Bharat provides health coverage of Rs. 5 lakh per family per year for secondary and tertiary care hospitalization to over 10 crore vulnerable families.",
                "link": "https://pib.gov.in",
                "published": "2024-03-20",
                "source": "PIB India"
            },
            {
                "title": "Make in India Program",
                "summary": "Make in India initiative was launched on 25 September 2014 to facilitate investment, foster innovation, enhance skill development and build best-in-class manufacturing infrastructure in the country.",
                "link": "https://pib.gov.in",
                "published": "2024-04-05",
                "source": "PIB India"
            },
            {
                "title": "Swachh Bharat Mission",
                "summary": "Swachh Bharat Mission aims to achieve universal sanitation coverage and has successfully constructed over 10 crore toilets across India since its launch.",
                "link": "https://pib.gov.in",
                "published": "2024-05-12",
                "source": "PIB India"
            },
            {
                "title": "Pradhan Mantri Ujjwala Yojana",
                "summary": "PM Ujjwala Yojana provides LPG connections to women from Below Poverty Line households. Over 9 crore connections have been released under the scheme.",
                "link": "https://pib.gov.in",
                "published": "2024-06-08",
                "source": "PIB India"
            },
            {
                "title": "Atal Pension Yojana",
                "summary": "Atal Pension Yojana (APY) is a pension scheme for citizens of India focused on the unorganized sector workers. Any Indian citizen between 18-40 years can join APY.",
                "link": "https://pib.gov.in",
                "published": "2024-07-15",
                "source": "PIB India"
            },
            {
                "title": "Skill India Mission",
                "summary": "Skill India Mission aims to train over 40 crore people in India in different skills by 2022. The initiative includes various programs like Pradhan Mantri Kaushal Vikas Yojana.",
                "link": "https://pib.gov.in",
                "published": "2024-08-20",
                "source": "PIB India"
            }
        ]
    
    def prepare_fact_base(self, statements: List[Dict]) -> List[Dict]:
        """
        Prepare statements for embedding and storage.
        
        Args:
            statements: List of raw statements
            
        Returns:
            List of processed fact dictionaries
        """
        facts = []
        
        for i, stmt in enumerate(statements):
            fact = {
                "id": f"pib_{i}",
                "text": stmt.get('text', stmt.get('summary', '')),
                "title": stmt.get('title', ''),
                "source": stmt.get('source', 'PIB India'),
                "published": stmt.get('published', ''),
                "url": stmt.get('link', '')
            }
            facts.append(fact)
        
        logger.info(f"Prepared {len(facts)} facts for storage")
        return facts
