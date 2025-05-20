"""
Enhanced sentiment analysis module using DistilBERT model for analyzing news headlines and technical indicators.
Includes caching, batch processing, technical analysis, and improved error handling.
"""
from transformers import pipeline
import torch
from functools import lru_cache
import logging
from typing import Dict, List, Union, Optional, Any
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
from dataclasses import dataclass

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TechnicalIndicators:
    """Technical indicators for stock analysis"""
    rsi: float
    sma_20: float
    sma_50: float
    sma_200: float
    price_volatility: float
    volume_trend: float
    momentum: float
    price: float

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

    def calculate_sentiment_stats(self, results: List[Dict[str, Union[str, float]]]) -> Dict[str, Union[float, Dict[str, float]]]:
        """
        Calculate aggregate sentiment statistics from a list of sentiment results.
        
        Args:
            results (List[Dict]): List of sentiment analysis results
            
        Returns:
            Dict: Aggregate statistics including percentages and averages
        """
        if not results:
            return {
                'positive_percentage': 0.0,
                'negative_percentage': 0.0,
                'neutral_percentage': 0.0,
                'overall_sentiment': 0.0,
                'average_confidence': 0.0,
                'sentiment_counts': {
                    'positive': 0,
                    'negative': 0,
                    'neutral': 0
                }
            }
            
        total = len(results)
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_score = 0.0
        total_confidence = 0.0
        
        for result in results:
            sentiment_counts[result['label']] += 1
            total_score += result['nuanced_score']
            # Convert confidence level back to score for averaging
            confidence_scores = {'very high': 0.95, 'high': 0.8, 'moderate': 0.6, 'low': 0.4}
            total_confidence += confidence_scores.get(result['confidence'], 0.5)
        
        return {
            'positive_percentage': round((sentiment_counts['positive'] / total) * 100, 2),
            'negative_percentage': round((sentiment_counts['negative'] / total) * 100, 2),
            'neutral_percentage': round((sentiment_counts['neutral'] / total) * 100, 2),
            'overall_sentiment': round(total_score / total, 3),
            'average_confidence': round(total_confidence / total, 3),
            'sentiment_counts': sentiment_counts
        }

    def analyze_batch(self, texts: List[str]) -> Dict[str, Union[List[Dict[str, Union[str, float]]], Dict[str, Union[float, Dict[str, int]]]]]:
        """
        Analyze sentiment for multiple texts in batches and provide aggregate statistics.
        
        Args:
            texts (List[str]): List of texts to analyze
            
        Returns:
            Dict: Results containing individual sentiments and aggregate statistics
        """
        if not texts:
            return {
                'individual_sentiments': [],
                'statistics': self.calculate_sentiment_stats([])
            }
            
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
                
        return {
            'individual_sentiments': results,
            'statistics': self.calculate_sentiment_stats(results)
        }

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

    def _calculate_rsi(self, prices: np.ndarray, periods: int = 14) -> float:
        """Calculate Relative Strength Index"""
        deltas = np.diff(prices)
        gain = np.where(deltas > 0, deltas, 0)
        loss = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gain[-periods:])
        avg_loss = np.mean(loss[-periods:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_sma(self, data: np.ndarray, window: int) -> float:
        """Calculate Simple Moving Average"""
        return np.mean(data[-window:])

    def _calculate_technical_indicators(self, ticker: str) -> Optional[TechnicalIndicators]:
        """
        Calculate technical indicators without external TA libraries.
        
        Args:
            ticker (str): Stock ticker symbol
            
        Returns:
            Optional[TechnicalIndicators]: Technical indicators if successful, None otherwise
        """
        try:
            # Fetch historical data
            stock = yf.Ticker(ticker)
            hist = stock.history(period="1y")
            
            if len(hist) < 200:
                logger.warning(f"Insufficient historical data for {ticker}")
                return None
                
            prices = hist['Close'].values
            volumes = hist['Volume'].values
            current_price = prices[-1]
            
            # Calculate indicators
            rsi = self._calculate_rsi(prices)
            sma_20 = self._calculate_sma(prices, 20)
            sma_50 = self._calculate_sma(prices, 50)
            sma_200 = self._calculate_sma(prices, 200)
            
            # Calculate volatility (20-day standard deviation)
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:]) * 100
            
            # Calculate volume trend (comparing current volume to 20-day average)
            volume_sma = self._calculate_sma(volumes, 20)
            volume_trend = (volumes[-1] / volume_sma - 1) * 100
            
            # Calculate momentum (percentage change over 14 days)
            momentum = ((prices[-1] / prices[-14]) - 1) * 100
            
            return TechnicalIndicators(
                rsi=rsi,
                sma_20=sma_20,
                sma_50=sma_50,
                sma_200=sma_200,
                price_volatility=volatility,
                volume_trend=volume_trend,
                momentum=momentum,
                price=current_price
            )
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
            return None

    def predict_stock_movement(self, ticker: str, indicators: TechnicalIndicators, sentiment_score: float) -> Dict[str, Any]:
        """
        Predict future price movement based on technical indicators and sentiment.
        
        Args:
            ticker (str): Stock ticker symbol
            indicators (TechnicalIndicators): Technical indicators
            sentiment_score (float): Current sentiment score
            
        Returns:
            Dict[str, Any]: Prediction results
        """
        try:
            # Calculate technical signals
            tech_signals = {
                'rsi_signal': 'oversold' if indicators.rsi < 30 else 'overbought' if indicators.rsi > 70 else 'neutral',
                'trend_signal': 'bullish' if indicators.sma_20 > indicators.sma_50 else 'bearish',
                'long_term_trend': 'bullish' if indicators.price > indicators.sma_200 else 'bearish',
                'volume_signal': 'bullish' if indicators.volume_trend > 10 else 'bearish' if indicators.volume_trend < -10 else 'neutral',
                'momentum_signal': 'bullish' if indicators.momentum > 0 else 'bearish'
            }
            
            # Calculate scores for each component (-1 to 1)
            technical_scores = {
                'rsi_score': (50 - indicators.rsi) / 50,  # Normalized RSI
                'trend_score': (indicators.sma_20 / indicators.sma_50 - 1) * 2,  # Normalized trend
                'volume_score': np.clip(indicators.volume_trend / 50, -1, 1),  # Normalized volume trend
                'momentum_score': np.clip(indicators.momentum / 10, -1, 1)  # Normalized momentum
            }
            
            # Calculate overall technical score
            tech_score = np.mean(list(technical_scores.values()))
            
            # Combine technical and sentiment scores
            combined_score = (tech_score + sentiment_score) / 2
            
            # Calculate confidence based on agreement between signals
            signal_agreement = len(set(tech_signals.values()))  # How many unique signals
            confidence = 1 - (signal_agreement - 1) / (len(tech_signals) - 1)  # Normalize to 0-1
            
            return {
                'technical_score': round(tech_score, 3),
                'combined_score': round(combined_score, 3),
                'prediction': 'strong_buy' if combined_score > 0.6 else
                             'buy' if combined_score > 0.2 else
                             'sell' if combined_score < -0.2 else
                             'strong_sell' if combined_score < -0.6 else
                             'hold',
                'confidence': round(confidence, 3),
                'signals': tech_signals,
                'metrics': {
                    'price': round(indicators.price, 2),
                    'rsi': round(indicators.rsi, 2),
                    'volatility': round(indicators.price_volatility, 2),
                    'volume_trend': round(indicators.volume_trend, 2),
                    'momentum': round(indicators.momentum, 2)
                },
                'moving_averages': {
                    'sma_20': round(indicators.sma_20, 2),
                    'sma_50': round(indicators.sma_50, 2),
                    'sma_200': round(indicators.sma_200, 2)
                }
            }
            
        except Exception as e:
            logger.error(f"Error predicting stock movement for {ticker}: {str(e)}")
            return {
                'prediction': 'hold',
                'confidence': 0,
                'error': str(e)
            }

    def analyze_stock(self, ticker: str, news_titles: List[str]) -> Dict[str, Any]:
        """
        Comprehensive stock analysis combining sentiment and technical indicators.
        
        Args:
            ticker (str): Stock ticker symbol
            news_titles (List[str]): List of news headlines
            
        Returns:
            Dict[str, Any]: Comprehensive analysis results
        """
        # Get sentiment analysis
        sentiment_results = self.analyze_batch(news_titles)
        sentiment_stats = sentiment_results['statistics']
        
        # Get technical indicators
        indicators = self._calculate_technical_indicators(ticker)
        if indicators is None:
            return {
                'error': f'Could not calculate technical indicators for {ticker}',
                'sentiment_analysis': sentiment_results
            }
            
        # Make prediction
        prediction = self.predict_stock_movement(
            ticker,
            indicators,
            sentiment_stats['overall_sentiment']
        )
        
        # Get fundamental data
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            fundamentals = {
                'pe_ratio': info.get('forwardPE', 0),
                'market_cap': info.get('marketCap', 0),
                'beta': info.get('beta', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'revenue_growth': info.get('revenueGrowth', 0),
                'profit_margins': info.get('profitMargins', 0)
            }
        except Exception as e:
            logger.error(f"Error fetching fundamental data for {ticker}: {str(e)}")
            fundamentals = {}
        
        return {
            'ticker': ticker,
            'timestamp': datetime.now().isoformat(),
            'sentiment_analysis': sentiment_results,
            'technical_analysis': prediction,
            'fundamentals': fundamentals,
            'recommendation': {
                'action': prediction['prediction'],
                'confidence': prediction['confidence'],
                'factors': {
                    'sentiment': sentiment_stats['overall_sentiment'],
                    'technical': prediction['technical_score'],
                    'combined': prediction['combined_score']
                }
            }
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
