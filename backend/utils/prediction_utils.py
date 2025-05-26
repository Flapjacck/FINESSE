import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import functools
import threading
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

# Cache for storing predictions
prediction_cache = {}
cache_lock = threading.Lock()

def cache_prediction(func):
    """
    Cache decorator for predictions with a 5-minute expiry.
    """
    @functools.wraps(func)
    def wrapper(ticker: str, *args, **kwargs):
        current_time = datetime.now()
        with cache_lock:
            if ticker in prediction_cache:
                prediction_time, prediction = prediction_cache[ticker]
                # Return cached prediction if less than 5 minutes old
                if current_time - prediction_time < timedelta(minutes=5):
                    return prediction
        
        # Calculate new prediction
        result = func(ticker, *args, **kwargs)
        
        # Cache the new prediction
        with cache_lock:
            prediction_cache[ticker] = (current_time, result)
        
        return result
    return wrapper

# Cache for storing indicators
indicator_cache = {}

def cache_indicators(func):
    """Cache decorator for technical indicators"""
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        # Create a cache key from the function name and data hash
        key = (func.__name__, hash(str(data.index[-1]) + str(data['Close'].iloc[-1])))
        if key in indicator_cache:
            return indicator_cache[key]
        result = func(data, *args, **kwargs)
        indicator_cache[key] = result
        return result
    return wrapper

def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD with signal line"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple:
    """Calculate Bollinger Bands with customizable standard deviation"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    return sma, upper_band, lower_band

def calculate_fibonacci_levels(data: pd.Series, period: int = 20) -> tuple:
    """Calculate Fibonacci retracement levels"""
    high = data.rolling(window=period).max()
    low = data.rolling(window=period).min()
    diff = high - low
    
    level_0 = low  # 0%
    level_23_6 = low + diff * 0.236
    level_38_2 = low + diff * 0.382
    level_50_0 = low + diff * 0.5
    level_61_8 = low + diff * 0.618
    level_100 = high  # 100%
    
    return level_0, level_23_6, level_38_2, level_50_0, level_61_8, level_100

def calculate_momentum_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate various momentum indicators"""
    # Rate of Change (ROC)
    data['ROC'] = data['Close'].pct_change(periods=10) * 100
    
    # Money Flow Index (MFI)
    typical_price = (data['High'] + data['Low'] + data['Close']) / 3
    money_flow = typical_price * data['Volume']
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(14).sum()
    mfi = 100 - (100 / (1 + positive_flow / negative_flow))
    data['MFI'] = mfi
    
    # Stochastic Oscillator
    low_14 = data['Low'].rolling(window=14).min()
    high_14 = data['High'].rolling(window=14).max()
    data['K_percent'] = ((data['Close'] - low_14) / (high_14 - low_14)) * 100
    data['D_percent'] = data['K_percent'].rolling(window=3).mean()
    
    return data

def calculate_trend_strength(data: pd.Series, window: int = 20) -> pd.Series:
    """Calculate trend strength using vectorized linear regression slope"""
    # Create rolling window view of the data
    values = data.values
    x = np.arange(window)
    # Vectorized calculation of slope for each window
    y = np.lib.stride_tricks.sliding_window_view(values, window)
    slopes = np.polyfit(x, y.T, 1)[0]
    # Pad the beginning with the first slope value
    slopes = np.pad(slopes, (window-1, 0), mode='edge')
    return pd.Series(slopes, index=data.index)

def calculate_price_momentum(data: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced price momentum indicators"""
    # Triple EMA for trend confirmation
    data['EMA_5'] = data['Close'].ewm(span=5, adjust=False).mean()
    data['EMA_10'] = data['Close'].ewm(span=10, adjust=False).mean()
    data['EMA_21'] = data['Close'].ewm(span=21, adjust=False).mean()
    
    # Price Rate of Change for multiple timeframes
    for period in [5, 10, 21]:
        data[f'ROC_{period}'] = data['Close'].pct_change(periods=period) * 100
    
    # Average Directional Index (ADX)
    high_diff = data['High'].diff()
    low_diff = data['Low'].diff()
    
    pos_dm = high_diff.where(high_diff > low_diff, 0.0)
    neg_dm = -low_diff.where(low_diff > high_diff, 0.0)
    
    tr = pd.DataFrame({
        'HL': data['High'] - data['Low'],
        'HC': abs(data['High'] - data['Close'].shift(1)),
        'LC': abs(data['Low'] - data['Close'].shift(1))
    }).max(axis=1)
    
    pos_di = 100 * (pos_dm.rolling(14).mean() / tr.rolling(14).mean())
    neg_di = 100 * (neg_dm.rolling(14).mean() / tr.rolling(14).mean())
    data['ADX'] = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    
    return data

def analyze_feature_importance(model, feature_names, X_train, y_train):
    """
    Analyze and select important features using model-based selection.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names (list): List of feature names
        X_train (array): Training features
        y_train (array): Training targets
    
    Returns:
        tuple: Selected feature indices and importance scores dictionary
    """
    # Fit model for feature selection
    selector = SelectFromModel(model, prefit=False)
    selector.fit(X_train, y_train)
    
    # Get feature importance scores
    importance_scores = {}
    if hasattr(model, 'feature_importances_'):
        importance_scores = dict(zip(feature_names, model.feature_importances_))
        importance_scores = dict(sorted(importance_scores.items(), key=lambda x: x[1], reverse=True))
    
    return selector.get_support(), importance_scores

def calculate_prediction_metrics(y_true, y_pred):
    """
    Calculate comprehensive prediction performance metrics.
    
    Args:
        y_true (array): Actual values
        y_pred (array): Predicted values
    
    Returns:
        dict: Dictionary containing various performance metrics
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Directional accuracy
    direction_correct = np.mean((np.diff(y_true) * np.diff(y_pred)) > 0)
    
    return {
        'mse': float(mse),
        'rmse': float(rmse),
        'mape': float(mape),
        'r2': float(r2),
        'directional_accuracy': float(direction_correct)
    }

def train_model_for_timeframe(X: np.ndarray, y: np.ndarray, forecast_period: int) -> tuple:
    """Train models for specific timeframe prediction"""
    # Ensure we have enough data for the forecast period
    if len(X) <= forecast_period:
        raise ValueError(f"Not enough data for {forecast_period} day forecast. Need more than {forecast_period} samples.")
    
    # Split data for time series forecasting, ensuring minimum training size
    min_train_size = max(60, forecast_period * 2)  # At least 60 days or 2x forecast period
    if len(X) < min_train_size + forecast_period:
        raise ValueError(f"Not enough data. Need at least {min_train_size + forecast_period} samples.")
    
    split_idx = len(X) - forecast_period
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Initialize models with parameters adjusted based on data size
    n_estimators = min(100, len(X_train) // 2)  # Adjust number of trees based on data size
    
    models = {
        'rf': RandomForestRegressor(
            n_estimators=n_estimators, 
            max_depth=min(10, len(X_train) // 10),  # Adjust depth based on data size
            min_samples_split=5,
            n_jobs=-1, 
            random_state=42
        ),
        'xgb': xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=min(6, len(X_train) // 20),  # Prevent overfitting on small datasets
            n_jobs=-1,
            random_state=42
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=min(6, len(X_train) // 20),
            min_samples_split=5,
            random_state=42
        )
    }
    
    # Train models and get predictions
    predictions = {}
    mape_scores = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mape = mean_absolute_percentage_error(y_test, pred)
        predictions[name] = pred
        mape_scores[name] = mape
    
    # Calculate weighted ensemble prediction
    weights = np.array([1/score for score in mape_scores.values()])
    weights = weights / weights.sum()
    
    ensemble_pred = np.zeros_like(predictions['rf'])
    for (name, pred), weight in zip(predictions.items(), weights):
        ensemble_pred += pred * weight
    
    confidence = 1 - np.mean(list(mape_scores.values()))  # Convert MAPE to confidence score
    return ensemble_pred[-1], confidence

def determine_trend(current_price: float, predicted_price: float, confidence: float) -> str:
    """Determine if trend is bullish, bearish, or neutral based on prediction"""
    percent_change = ((predicted_price - current_price) / current_price) * 100
    if confidence < 0.4:  # Low confidence threshold
        return "neutral"
    elif percent_change > 2:  # More than 2% increase
        return "bullish"
    elif percent_change < -2:  # More than 2% decrease
        return "bearish"
    else:
        return "neutral"

@cache_prediction
def get_stock_prediction(ticker: str) -> dict:
    """
    Get comprehensive stock predictions for different time periods.
    Args:
        ticker (str): Stock symbol (e.g., 'AAPL', 'GOOGL')
    Returns:
        dict: Predictions for different time periods
    """
    try:
        # Fetch 5 years of historical data for better long-term predictions
        stock = yf.Ticker(ticker)
        hist = stock.history(period="5y")
        
        if hist.empty:
            raise ValueError(f"No historical data found for ticker {ticker}")
            
        # Ensure minimum data requirements
        if len(hist) < 252:  # At least one year of data
            raise ValueError(f"Insufficient historical data for {ticker}. Need at least 1 year of data.")
        
        # Calculate technical indicators
        hist['RSI'] = calculate_rsi(hist['Close'])
        macd, signal = calculate_macd(hist['Close'])
        hist['MACD'] = macd
        hist['Signal'] = signal
        sma20, bb_upper, bb_lower = calculate_bollinger_bands(hist['Close'])
        hist['SMA_20'] = sma20
        hist['BB_Upper'] = bb_upper
        hist['BB_Lower'] = bb_lower
        
        # Add price and volume features
        hist['Price_Change'] = hist['Close'].pct_change()
        hist['Volatility'] = hist['Close'].pct_change().rolling(window=20).std()
        hist['Volume_Change'] = hist['Volume'].pct_change()
        
        # Create feature matrix
        feature_columns = ['RSI', 'MACD', 'Signal', 'SMA_20', 'Price_Change', 'Volatility', 'Volume_Change']
        hist = hist.dropna()
        
        # Prepare features
        scaler = StandardScaler()
        X = scaler.fit_transform(hist[feature_columns])
        y = hist['Close'].values
        
        # Define prediction periods in trading days
        periods = {
            '1d': 1,
            '1w': 5,
            '1m': 21,
            '1y': 252
        }
        
        current_price = hist['Close'].iloc[-1]
        predictions = {}
        
        for period_name, days in periods.items():
            try:
                predicted_price, confidence = train_model_for_timeframe(X, y, days)
                
                trend = determine_trend(current_price, predicted_price, confidence)
                
                predictions[period_name] = {
                    'trend': trend,
                    'confidence': float(confidence),
                    'predicted_price': float(predicted_price),
                    'percent_change': float(((predicted_price - current_price) / current_price) * 100)
                }
                
            except Exception as e:
                predictions[period_name] = {
                    'error': f"Failed to predict {period_name}: {str(e)}"
                }
        
        # Add technical indicators for current state
        current_signals = {
            'rsi_signal': 'oversold' if hist['RSI'].iloc[-1] < 30 else 'overbought' if hist['RSI'].iloc[-1] > 70 else 'neutral',
            'macd_signal': 'bullish' if hist['MACD'].iloc[-1] > hist['Signal'].iloc[-1] else 'bearish',
            'volatility': float(hist['Volatility'].iloc[-1]),
            'volume_trend': 'increasing' if hist['Volume_Change'].iloc[-1] > 0 else 'decreasing'
        }
        
        return {
            'current_price': float(current_price),
            'time_based_predictions': predictions,
            'technical_signals': current_signals,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        return {'error': str(e)}
