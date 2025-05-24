import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Calculate MACD"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    return macd

def get_stock_prediction(ticker: str) -> dict:
    """
    Get a stock prediction based on historical data using technical analysis and ML.
    
    Algorithm Overview:
    1. Fetches 60 days of historical price data using yfinance
    2. Calculates technical indicators:
       - RSI (Relative Strength Index) for overbought/oversold conditions
       - MACD (Moving Average Convergence Divergence) for trend strength
       - Volume analysis for trade activity confirmation
    3. Uses Random Forest model to predict price movements based on indicators
    4. Generates confidence score based on:
       - Technical indicator signals (70% weight)
       - ML model prediction accuracy (30% weight)
    """
    try:
        # Fetch historical data using yfinance API
        stock = yf.Ticker(ticker)
        hist = stock.history(period='60d')
        
        if hist.empty:
            raise ValueError(f"No historical data found for ticker {ticker}")

        # Calculate technical indicators using native pandas functions
        # RSI (Relative Strength Index)
        hist['RSI_14'] = calculate_rsi(hist['Close'])
        
        # MACD (Moving Average Convergence Divergence)
        hist['MACD_12_26_9'] = calculate_macd(hist['Close'])
        
        # Volume Analysis
        hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_MA']
        
        # Handle missing values before ML
        hist = hist.fillna(method='ffill').fillna(method='bfill')
        
        # Prepare features for ML prediction
        features = ['RSI_14', 'MACD_12_26_9', 'Volume_Ratio']
        
        # Ensure all required features exist
        missing_features = [f for f in features if f not in hist.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {', '.join(missing_features)}")
        
        X = hist[features].values[-30:]  # Use last 30 days for prediction
        y = hist['Close'].values[-30:]
        
        # Check for any remaining NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Data contains NaN values after preprocessing")
        
        # Train a simple Random Forest model
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        model.fit(X[:-1], y[1:])  # Train on all but last day
        
        # Make prediction
        prediction_value = model.predict(X[-1:].reshape(1, -1))[0]
        current_price = hist['Close'].iloc[-1]
        
        # Get latest technical indicators
        latest_rsi = float(hist['RSI_14'].iloc[-1])
        latest_macd = float(hist['MACD_12_26_9'].iloc[-1])
        latest_vol_ratio = float(hist['Volume_Ratio'].iloc[-1])
        
        # Determine direction and confidence
        direction = "upward" if prediction_value > current_price else "downward"
        
        # Calculate confidence score based on multiple factors
        technical_confidence = min(100, (
            abs(latest_rsi - 50) / 50 * 40 +  # RSI contribution
            abs(latest_macd) * 20 +           # MACD contribution
            (latest_vol_ratio - 1) * 10       # Volume confirmation
        ))
        
        ml_confidence = min(100, (
            abs(prediction_value - current_price) / current_price * 100
        ))
        
        # Combined confidence (70% technical, 30% ML)
        confidence = 0.7 * technical_confidence + 0.3 * ml_confidence
        
        return {
            'ticker': ticker,
            'current_price': round(float(current_price), 2),
            'predicted_direction': direction,
            'confidence_score': round(confidence, 2),
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'metrics': {
                'rsi': round(float(latest_rsi), 2),
                'macd': round(float(latest_macd), 2),
                'volume_ratio': round(float(latest_vol_ratio), 2),
                'ml_predicted_price': round(float(prediction_value), 2)
            }
        }
        
    except Exception as e:
        raise Exception(f"Error generating prediction: {str(e)}")
