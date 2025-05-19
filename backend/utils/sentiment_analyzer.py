"""
Sentiment analysis module using DistilBERT model for analyzing news headlines.
"""
from transformers import pipeline
import torch

class SentimentAnalyzer:
    def __init__(self):
        """Initialize the sentiment analyzer with DistilBERT model."""
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"
        self.analyzer = None

    def initialize(self):
        """
        Lazy initialization of the model to avoid loading it before Flask app is ready.
        """
        if self.analyzer is None:
            self.analyzer = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                device=-1  # Use CPU
            )
            # Optimize memory usage
            torch.cuda.empty_cache()
            self.analyzer.model.to('cpu')

    def analyze_text(self, text):
        """
        Analyze the sentiment of given text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            dict: Sentiment analysis result with label and score
        """
        if not text:
            return {'label': 'neutral', 'score': 0.0}
            
        if self.analyzer is None:
            self.initialize()
            
        try:
            result = self.analyzer(text)[0]
            return result
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {'label': 'neutral', 'score': 0.0}

# Create a singleton instance
sentiment_analyzer = SentimentAnalyzer()
