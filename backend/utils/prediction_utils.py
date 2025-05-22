import yfinance as yf
import numpy as np
from datetime import datetime, timedelta

def get_stock_prediction(ticker: str) -> dict:
    """
    Get a simple stock prediction based on historical data.
    Uses a basic approach looking at recent trends and volatility to predict stock movement.
    
    Algorithm Overview:
    1. Fetches 30 days of historical price data using yfinance
    2. Calculates key metrics:
       - Current price (most recent closing price)
       - 30-day moving average
       - Price volatility (standard deviation)
    3. Analyzes momentum by comparing:
       - Last 5 days average price
       - Previous 5 days average price (days 6-10)
    4. Generates confidence score based on:
       - Strength of price movement relative to volatility
       - Score is normalized to 0-100 range
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
        
    Returns:
        dict: Prediction data containing:
            - ticker: Stock symbol
            - current_price: Latest closing price
            - predicted_direction: 'upward' or 'downward'
            - confidence_score: 0-100 score indicating prediction confidence
            - prediction_date: Date when prediction was generated
            - metrics: Additional technical metrics used in prediction
            
    Raises:
        ValueError: If no historical data is found for the ticker
        Exception: For other errors during data fetching or processing
    """
    try:        # Fetch historical data using yfinance API
        # period='30d' gets us exactly 30 calendar days of data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='30d')
        
        if hist.empty:
            raise ValueError(f"No historical data found for ticker {ticker}")

        # Calculate core technical indicators
        current_price = hist['Close'].iloc[-1]  # Most recent closing price
        avg_price = hist['Close'].mean()        # 30-day moving average
        std_dev = hist['Close'].std()           # Price volatility measure
        
        # Momentum Analysis
        # Compare last 5 days vs previous 5 days to determine price momentum
        # Positive value indicates upward momentum, negative indicates downward
        recent_5d_avg = hist['Close'].iloc[-5:].mean()     # Last 5 days
        prev_5d_avg = hist['Close'].iloc[-10:-5].mean()    # Previous 5 days
        recent_change = recent_5d_avg - prev_5d_avg
        
        # Set prediction direction based on momentum
        direction = "upward" if recent_change > 0 else "downward"
            
        # Confidence Score Calculation:
        # 1. Take the absolute magnitude of price change
        # 2. Compare it to the stock's volatility (std_dev)
        # 3. Multiply by 50 to scale to 0-100 range
        # 4. Cap at 100 for very strong movements
        confidence = min(100, abs(recent_change / std_dev * 50))
        
        return {
            'ticker': ticker,
            'current_price': round(current_price, 2),
            'predicted_direction': direction,
            'confidence_score': round(confidence, 2),
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'metrics': {
                'average_price_30d': round(avg_price, 2),
                'volatility_30d': round(std_dev, 2),
                'recent_trend': round(recent_change, 2)
            }
        }
        
    except Exception as e:
        raise Exception(f"Error generating prediction: {str(e)}")
