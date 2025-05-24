import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, root_mean_squared_error
import xgboost as xgb

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

def calculate_bollinger_bands(data: pd.Series, window: int = 20) -> tuple:
    """Calculate Bollinger Bands"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * 2)
    lower_band = sma - (std * 2)
    return sma, upper_band, lower_band

def calculate_support_resistance(data: pd.Series, window: int = 20) -> tuple:
    """Calculate dynamic support and resistance levels"""
    rolling_min = data.rolling(window=window).min()
    rolling_max = data.rolling(window=window).max()
    return rolling_min, rolling_max

def is_trading_day(date):
    """Check if the given date is a trading day (excluding weekends)"""
    return date.weekday() < 5  # 0-4 are Monday through Friday

def get_next_trading_day(date):
    """Get the next trading day after the given date"""
    next_day = date + timedelta(days=1)
    while not is_trading_day(next_day):
        next_day += timedelta(days=1)
    return next_day

def predict_price_timeline(model, features, current_price, target_price):
    """
    Predict when the target price will be reached, considering only trading days
    Returns the number of trading days until the target is reached
    """
    max_trading_days = 30
    trading_days = 0
    current_date = datetime.now()
    current_features = features.copy()
    current_pred = current_price
    tolerance = 0.0001  # Add small tolerance for floating point comparison
    predictions = []  # Store all predictions to analyze trend
    
    # If target is very close to current price, return 1 day
    if abs(target_price - current_price) / current_price < 0.001:
        return 1
    
    target_change = (target_price - current_price) / current_price
    
    # Loop through trading days
    for i in range(max_trading_days * 2):  # Double the range to account for weekends
        if trading_days >= max_trading_days:
            break
            
        # Get next trading day
        current_date = get_next_trading_day(current_date)
        if not is_trading_day(current_date):
            continue
            
        # Make prediction for this day
        pred = model.predict(current_features.reshape(1, -1))[0]
        predictions.append(pred)
        
        # Check if target price is reached with tolerance
        if target_price > current_price:
            if pred >= target_price - tolerance:
                return trading_days + 1
        else:
            if pred <= target_price + tolerance:
                return trading_days + 1
        
        # Update features for next prediction
        price_change = (pred - current_pred) / current_pred
        current_features[4] = price_change  # Update Price_Change feature
        current_pred = pred
        trading_days += 1
    
    # If we haven't reached the target, analyze the trend
    if len(predictions) >= 2:
        # Calculate average daily change
        daily_changes = [(predictions[i] - predictions[i-1])/predictions[i-1] 
                        for i in range(1, len(predictions))]
        avg_daily_change = sum(daily_changes) / len(daily_changes)
        
        # If moving in right direction, estimate days based on current rate
        if (target_change > 0 and avg_daily_change > 0) or (target_change < 0 and avg_daily_change < 0):
            # Estimate days needed at current rate
            estimated_days = int(abs(target_change / avg_daily_change))
            if estimated_days <= max_trading_days:
                return estimated_days
            elif abs(avg_daily_change * max_trading_days) >= abs(target_change) * 0.5:
                # Return max_days if we can achieve at least 50% of target move
                return max_trading_days
    
    return None

def get_stock_prediction(ticker: str) -> dict:
    """
    Get a comprehensive stock prediction using advanced technical analysis and ensemble ML methods.
    
    Algorithm Overview:
    1. Fetches 120 days of historical price data using yfinance
    2. Calculates advanced technical indicators:
       - RSI (Relative Strength Index)
       - MACD (Moving Average Convergence Divergence)
       - Bollinger Bands for volatility
       - Dynamic Support/Resistance levels
       - Volume analysis and patterns
    3. Uses ensemble of ML models:
       - Random Forest for pattern recognition
       - XGBoost for trend prediction
       - Gradient Boosting for price targets
    4. Generates detailed analysis including:
       - Price targets (short/medium term)
       - Timeline predictions
       - Confidence scores with multiple factors
       - Technical signal strength
    """
    try:
        # Fetch historical data using yfinance API
        stock = yf.Ticker(ticker)
        hist = stock.history(period='120d')  # Extended period for better training
        
        if hist.empty:
            raise ValueError(f"No historical data found for ticker {ticker}")

        # Calculate comprehensive technical indicators
        hist['RSI_14'] = calculate_rsi(hist['Close'])
        hist['MACD'] = calculate_macd(hist['Close'])
        hist['SMA_20'], hist['BB_Upper'], hist['BB_Lower'] = calculate_bollinger_bands(hist['Close'])
        hist['Support'], hist['Resistance'] = calculate_support_resistance(hist['Close'])
        
        # Enhanced volume analysis
        hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_MA']
        hist['Price_Range'] = hist['High'] - hist['Low']
        hist['Price_Change'] = hist['Close'].pct_change()

        # Handle missing values
        hist = hist.fillna(method='ffill').fillna(method='bfill')
        
        # Prepare features for ML prediction
        features = [
            'RSI_14', 'MACD', 'Volume_Ratio', 'Price_Range',
            'Price_Change', 'SMA_20', 'BB_Upper', 'BB_Lower'
        ]
        
        X = hist[features].values[-60:]  # Use last 60 days for prediction
        y = hist['Close'].values[-60:]
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train multiple models
        rf_model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
        gb_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1)
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X_scaled[:-1], y[1:], test_size=0.2, shuffle=False)
        
        # Train models
        rf_model.fit(X_train, y_train)
        xgb_model.fit(X_train, y_train)
        gb_model.fit(X_train, y_train)
        
        # Make ensemble prediction
        latest_features = X_scaled[-1:].reshape(1, -1)
        rf_pred = rf_model.predict(latest_features)[0]
        xgb_pred = xgb_model.predict(latest_features)[0]
        gb_pred = gb_model.predict(latest_features)[0]
        
        # Weighted ensemble prediction
        prediction_value = (rf_pred * 0.4 + xgb_pred * 0.4 + gb_pred * 0.2)
        current_price = hist['Close'].iloc[-1]
        
        # Calculate model accuracies
        rf_accuracy = 1 - root_mean_squared_error(y_test, rf_model.predict(X_test)) / np.mean(y_test)
        xgb_accuracy = 1 - root_mean_squared_error(y_test, xgb_model.predict(X_test)) / np.mean(y_test)
        gb_accuracy = 1 - root_mean_squared_error(y_test, gb_model.predict(X_test)) / np.mean(y_test)
        
        # Get latest technical indicators
        latest_rsi = float(hist['RSI_14'].iloc[-1])
        latest_macd = float(hist['MACD'].iloc[-1])
        latest_vol_ratio = float(hist['Volume_Ratio'].iloc[-1])
        
        # Determine direction and multiple price targets
        direction = "upward" if prediction_value > current_price else "downward"
        price_change_pct = abs(prediction_value - current_price) / current_price * 100
        
        # Predict timeline with trading days
        days_to_target = predict_price_timeline(rf_model, latest_features[0], current_price, prediction_value)
        if days_to_target:
            target_date = datetime.now()
            for _ in range(days_to_target):
                target_date = get_next_trading_day(target_date)
        else:
            target_date = None

        # Calculate enhanced confidence score
        technical_confidence = min(100, (
            abs(latest_rsi - 50) / 50 * 30 +     # RSI contribution
            abs(latest_macd) * 20 +              # MACD contribution
            (latest_vol_ratio - 1) * 20 +        # Volume confirmation
            (price_change_pct) * 30              # Price movement magnitude
        ))
        
        ml_confidence = min(100, (
            rf_accuracy * 40 +                    # Random Forest accuracy
            xgb_accuracy * 40 +                  # XGBoost accuracy
            gb_accuracy * 20                     # Gradient Boosting accuracy
        ))
          # Combined confidence (60% technical, 40% ML)
        confidence = 0.6 * technical_confidence + 0.4 * ml_confidence
        
        return {
            'ticker': ticker,
            'current_price': round(float(current_price), 2),
            'predicted_direction': direction,
            'predicted_price': round(float(prediction_value), 2),
            'price_targets': {
                'conservative': round(float(min(rf_pred, xgb_pred, gb_pred)), 2),
                'moderate': round(float(prediction_value), 2),
                'aggressive': round(float(max(rf_pred, xgb_pred, gb_pred)), 2)
            },
            'confidence_score': round(confidence, 2),
            'prediction_date': datetime.now().strftime('%Y-%m-%d'),
            'technical_analysis': {
                'rsi': {
                    'value': round(float(latest_rsi), 2),
                    'signal': 'oversold' if latest_rsi < 30 else 'overbought' if latest_rsi > 70 else 'neutral'
                },
                'macd': {
                    'value': round(float(latest_macd), 2),
                    'signal': 'bullish' if latest_macd > 0 else 'bearish'
                },
                'volume': {
                    'ratio': round(float(latest_vol_ratio), 2),
                    'signal': 'high' if latest_vol_ratio > 1.5 else 'low' if latest_vol_ratio < 0.5 else 'normal'
                },
                'support_resistance': {
                    'support': round(float(hist['Support'].iloc[-1]), 2),
                    'resistance': round(float(hist['Resistance'].iloc[-1]), 2)
                }
            },
            'model_metrics': {
                'rf_accuracy': round(rf_accuracy * 100, 2),
                'xgb_accuracy': round(xgb_accuracy * 100, 2),
                'gb_accuracy': round(gb_accuracy * 100, 2)
            }
        }
        
    except Exception as e:
        raise Exception(f"Error generating prediction: {str(e)}")
