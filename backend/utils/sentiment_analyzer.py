"""
Enhanced sentiment analysis module using DistilBERT model for analyzing news headlines.
Includes caching, batch processing, and improved error handling.
"""
from transformers import pipeline
import torch
from functools import lru_cache
import logging
from typing import Dict, List, Union, Optional
import numpy as np
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
                 cache_size: int = 1000,
                 batch_size: int = 8):
        """
        Initialize the sentiment analyzer with configurable parameters.
        
        Args:
            model_name (str): Name of the HuggingFace model to use
            cache_size (int): Number of results to cache
            batch_size (int): Size of batches for processing multiple texts
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.analyzer = None
        self.cache_size = cache_size
        self._setup_cache()

    def _setup_cache(self):
        """Configure the LRU cache for sentiment analysis results."""
        self.analyze_text = lru_cache(maxsize=self.cache_size)(self._analyze_text_impl)

    def initialize(self):
        """
        Lazy initialization of the model with proper error handling.
        """
        if self.analyzer is None:
            try:
                logger.info(f"Initializing sentiment analyzer with model: {self.model_name}")
                self.analyzer = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device='cuda' if torch.cuda.is_available() else 'cpu'
                )
                # Optimize memory usage
                torch.cuda.empty_cache()
                logger.info("Sentiment analyzer initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize sentiment analyzer: {str(e)}")
                raise RuntimeError(f"Model initialization failed: {str(e)}")

    def _analyze_text_impl(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Internal implementation of sentiment analysis for a single text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Enhanced sentiment analysis result
        """
        if not text or not isinstance(text, str):
            return self._get_neutral_sentiment()
            
        if self.analyzer is None:
            self.initialize()
            
        try:
            result = self.analyzer(text)[0]
            
            # Convert binary sentiment to a more nuanced score (-1 to 1)
            sentiment_score = self._calculate_nuanced_score(result)
            
            return {
                'label': result['label'].lower(),
                'score': result['score'],
                'nuanced_score': sentiment_score,
                'timestamp': datetime.now().isoformat(),
                'confidence': self._calculate_confidence(result['score'])
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return self._get_neutral_sentiment()

    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        """
        Analyze sentiment for multiple texts in batches for better performance.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            List[dict]: List of sentiment analysis results
        """
        if not texts:
            return []
            
        if self.analyzer is None:
            self.initialize()
            
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                batch_results = self.analyzer(batch)
                results.extend([
                    self._enhance_sentiment_result(result)
                    for result in batch_results
                ])
            except Exception as e:
                logger.error(f"Error in batch sentiment analysis: {str(e)}")
                results.extend([self._get_neutral_sentiment() for _ in batch])
                
        return results

    def _calculate_nuanced_score(self, result: Dict[str, Union[str, float]]) -> float:
        """
        Calculate a more nuanced sentiment score from -1 (very negative) to 1 (very positive).
        
        Args:
            result (dict): Raw sentiment analysis result
            
        Returns:
            float: Nuanced sentiment score
        """
        if result['label'].lower() == 'positive':
            return result['score']
        return -result['score']

    def _calculate_confidence(self, score: float) -> str:
        """
        Convert the confidence score to a human-readable level.
        
        Args:
            score (float): Raw confidence score
            
        Returns:
            str: Confidence level description
        """
        if score >= 0.9:
            return 'very high'
        elif score >= 0.7:
            return 'high'
        elif score >= 0.5:
            return 'moderate'
        else:
            return 'low'

    def _get_neutral_sentiment(self) -> Dict[str, Union[str, float]]:
        """
        Return a neutral sentiment result for error cases.
        
        Returns:
            dict: Neutral sentiment analysis result
        """
        return {
            'label': 'neutral',
            'score': 0.5,
            'nuanced_score': 0.0,
            'timestamp': datetime.now().isoformat(),
            'confidence': 'low'
        }

    def _enhance_sentiment_result(self, result: Dict[str, Union[str, float]]) -> Dict[str, Union[str, float]]:
        """
        Enhance the raw sentiment result with additional metrics.
        
        Args:
            result (dict): Raw sentiment analysis result
            
        Returns:
            dict: Enhanced sentiment analysis result
        """
        return {
            'label': result['label'].lower(),
            'score': result['score'],
            'nuanced_score': self._calculate_nuanced_score(result),
            'timestamp': datetime.now().isoformat(),
            'confidence': self._calculate_confidence(result['score'])
        }

    def cleanup(self):
        """
        Clean up resources used by the model.
        """
        if self.analyzer is not None:
            # Clear the cache
            self.analyze_text.cache_clear()
            # Free up GPU memory if used
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.analyzer = None
            logger.info("Sentiment analyzer resources cleaned up")

# Create a singleton instance
sentiment_analyzer = SentimentAnalyzer()
