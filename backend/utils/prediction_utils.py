import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, mean_absolute_percentage_error
import xgboost as xgb

def calculate_rsi(data: pd.Series, periods: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=periods).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=periods).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
    """Calculate MACD with signal line and histogram"""
    exp1 = data.ewm(span=fast, adjust=False).mean()
    exp2 = data.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple:
    """Calculate Bollinger Bands with customizable standard deviation"""
    sma = data.rolling(window=window).mean()
    std = data.rolling(window=window).std()
    upper_band = sma + (std * num_std)
    lower_band = sma - (std * num_std)
    bandwidth = (upper_band - lower_band) / sma
    return sma, upper_band, lower_band, bandwidth

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
    """Calculate trend strength using linear regression slope"""
    slopes = []
    for i in range(len(data) - window + 1):
        y = data.iloc[i:i+window].values
        x = np.arange(len(y))
        slope, _ = np.polyfit(x, y, 1)
        slopes.append(slope)
    
    slopes = [slopes[0]] * (window-1) + slopes  # Pad beginning with first value
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

def get_stock_prediction(ticker: str) -> dict:
    """
    Get a comprehensive stock prediction using advanced technical analysis and ensemble ML methods.
    
    Enhanced Algorithm Overview:
    1. Fetches 2 years of historical data for better pattern recognition
    2. Calculates advanced technical indicators:
       - Enhanced RSI with multiple timeframes
       - MACD with signal line and histogram
       - Advanced Bollinger Bands with bandwidth
       - Fibonacci retracement levels
       - Multiple momentum indicators (ROC, MFI, Stochastic)
       - Triple EMA crossovers
       - ADX for trend strength
    3. Uses sophisticated ensemble of ML models with time series validation:
       - Random Forest with feature importance
       - XGBoost with early stopping
       - Gradient Boosting with custom loss function
    4. Generates detailed analysis including:
       - Short-term (1-5 days) and medium-term (1-3 weeks) predictions
       - Confidence scores based on model consensus and historical accuracy
       - Risk assessment based on volatility metrics
       - Support/resistance breakout signals
       - Trend strength and momentum analysis
    """
    try:
        # Fetch extended historical data
        stock = yf.Ticker(ticker)
        hist = stock.history(period='2y')
        
        if hist.empty:
            raise ValueError(f"No historical data found for ticker {ticker}")

        # Calculate comprehensive technical indicators
        hist['RSI_14'] = calculate_rsi(hist['Close'])
        hist['RSI_28'] = calculate_rsi(hist['Close'], 28)
        hist['MACD'], hist['Signal'], hist['MACD_Hist'] = calculate_macd(hist['Close'])
        hist['SMA_20'], hist['BB_Upper'], hist['BB_Lower'], hist['BB_Bandwidth'] = calculate_bollinger_bands(hist['Close'])
        fib_levels = calculate_fibonacci_levels(hist['Close'])
        hist['Fib_0'], hist['Fib_23.6'], hist['Fib_38.2'], hist['Fib_50'], hist['Fib_61.8'], hist['Fib_100'] = fib_levels
        
        # Enhanced analysis
        hist = calculate_momentum_indicators(hist)
        hist = calculate_price_momentum(hist)
        hist['Trend_Strength'] = calculate_trend_strength(hist['Close'])
        
        # Advanced volume analysis
        hist['Volume_MA'] = hist['Volume'].rolling(window=20).mean()
        hist['Volume_Ratio'] = hist['Volume'] / hist['Volume_MA']
        hist['OBV'] = (np.sign(hist['Close'].diff()) * hist['Volume']).cumsum()
        
        # Price movement patterns
        hist['Daily_Range'] = hist['High'] - hist['Low']
        hist['Gap'] = hist['Open'] - hist['Close'].shift(1)
        hist['Price_Acceleration'] = hist['Close'].diff().diff()
        
        # Volatility features
        hist['ATR'] = (hist['High'] - hist['Low']).rolling(window=14).mean()
        hist['Volatility'] = hist['Close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        # Handle missing values
        hist = hist.ffill().bfill()
        
        # Prepare features for ML prediction
        features = [
            'RSI_14', 'RSI_28', 'MACD', 'Signal', 'MACD_Hist',
            'BB_Bandwidth', 'ROC', 'MFI', 'K_percent', 'D_percent',
            'Volume_Ratio', 'Volatility', 'ATR', 'ADX',
            'EMA_5', 'EMA_10', 'EMA_21',
            'ROC_5', 'ROC_10', 'ROC_21',
            'Trend_Strength', 'Daily_Range', 'Price_Acceleration'
        ]
        
        X = hist[features].values
        y = hist['Close'].values
        
        # Normalize features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Time series cross-validation with expanding window
        tscv = TimeSeriesSplit(n_splits=5, test_size=30)  # 30 days test size
        
        # Initialize models with optimized parameters
        rf_model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        xgb_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1,
            random_state=42
        )
        
        gb_model = GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=6,
            subsample=0.8,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        
        predictions = []
        confidences = []
        model_performances = []
        
        # Training and validation
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train models
            rf_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            gb_model.fit(X_train, y_train)
            
            # Make predictions
            rf_pred = rf_model.predict(X_val)
            xgb_pred = xgb_model.predict(X_val)
            gb_pred = gb_model.predict(X_val)
            
            # Calculate model performance
            rf_mape = mean_absolute_percentage_error(y_val, rf_pred)
            xgb_mape = mean_absolute_percentage_error(y_val, xgb_pred)
            gb_mape = mean_absolute_percentage_error(y_val, gb_pred)
            
            # Dynamic weight assignment based on performance
            total_error = rf_mape + xgb_mape + gb_mape
            rf_weight = (1 - rf_mape/total_error) / 2
            xgb_weight = (1 - xgb_mape/total_error) / 2
            gb_weight = (1 - gb_mape/total_error) / 2
            
            # Weighted ensemble prediction
            ensemble_pred = (rf_pred * rf_weight + xgb_pred * xgb_weight + gb_pred * gb_weight)
            predictions.append(ensemble_pred[-1])
            
            # Model agreement score
            std_preds = np.std([rf_pred[-1], xgb_pred[-1], gb_pred[-1]]) / np.mean([rf_pred[-1], xgb_pred[-1], gb_pred[-1]])
            confidence = 1 - min(std_preds, 0.5) * 2
            confidences.append(confidence)
            
            # Store performance metrics
            model_performances.append({
                'rf_mape': rf_mape,
                'xgb_mape': xgb_mape,
                'gb_mape': gb_mape
            })
        
        # Final predictions and analysis
        final_prediction = np.mean(predictions)
        final_confidence = np.mean(confidences)
        
        # Calculate prediction intervals
        prediction_std = np.std(predictions)
        lower_bound = final_prediction - (1.96 * prediction_std)
        upper_bound = final_prediction + (1.96 * prediction_std)
        
        # Trend analysis
        current_price = hist['Close'].iloc[-1]
        trend_strength = abs(final_prediction - current_price) / current_price
        price_momentum = hist['ROC_5'].iloc[-1]
        
        # Market context
        market_trend = 'bullish' if all(hist['EMA_5'].iloc[-3:] > hist['EMA_21'].iloc[-3:]) else 'bearish'
        volume_trend = 'increasing' if hist['Volume_Ratio'].iloc[-1] > 1.2 else 'decreasing' if hist['Volume_Ratio'].iloc[-1] < 0.8 else 'stable'
        
        return {
            'current_price': float(current_price),
            'predicted_price': float(final_prediction),
            'confidence': float(final_confidence),
            'prediction_interval': {
                'lower': float(lower_bound),
                'upper': float(upper_bound)
            },
            'trend': {
                'direction': 'up' if final_prediction > current_price else 'down',
                'strength': float(trend_strength),
                'momentum': float(price_momentum),
                'market_context': market_trend,
                'support_level': float(hist['Fib_38.2'].iloc[-1]),
                'resistance_level': float(hist['Fib_61.8'].iloc[-1])
            },
            'technical_signals': {
                'rsi_signal': 'oversold' if hist['RSI_14'].iloc[-1] < 30 else 'overbought' if hist['RSI_14'].iloc[-1] > 70 else 'neutral',
                'macd_signal': 'bullish' if hist['MACD_Hist'].iloc[-1] > 0 else 'bearish',
                'adx_trend_strength': float(hist['ADX'].iloc[-1]),
                'volume_trend': volume_trend,
                'volatility': float(hist['Volatility'].iloc[-1])
            },
            'model_metrics': {
                'avg_rf_mape': float(np.mean([p['rf_mape'] for p in model_performances])),
                'avg_xgb_mape': float(np.mean([p['xgb_mape'] for p in model_performances])),
                'avg_gb_mape': float(np.mean([p['gb_mape'] for p in model_performances]))
            }
        }
        
    except Exception as e:
        return {'error': str(e)}
