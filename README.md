# FINESSE

A comprehensive stock market data platform built with Flask and React, providing real-time financial data through the Yahoo Finance API.

## 🌟 Overview

FINESSE is a modern stock market data platform that delivers comprehensive financial information about stocks. Built with Python (Flask) backend and React frontend, it provides an intuitive interface to access market data and stock information.

## Getting Started

### Prerequisites

- [Uses yFinance API](https://github.com/ranaroussi/yfinance)
- Python 3.8 or higher
- pip package manager

### Backend Setup

1. Clone the repository:

```powershell
git clone https://github.com/flapjacck/finesse.git
cd finesse
```

2. Create and activate a virtual environment:

```powershell
cd backend
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:

```powershell
# Upgrade pip first
python -m pip install --upgrade pip
pip install -r requirements.txt
```

4. Run the server:

```powershell
py app.py
```

The server will start at `http://127.0.0.1:5000`

### Frontend Setup

1. Navigate to the frontend directory:

```bash
cd frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## API Documentation

The API provides the following endpoints:

### Health Check

Check if the API is up and running.

```http
GET /health
```

Example Response:

```json
{
    "status": "healthy",
    "timestamp": "2025-05-20T10:00:00.000Z",
    "version": "1.0.0"
}
```

### Get Stock Data

Get comprehensive stock data for a specific ticker symbol.

```http
GET http://127.0.0.1:5000/stock/data?ticker={symbol}
```

Parameters:

- `ticker` (required): Stock ticker symbol (e.g., AAPL, MSFT)

Example Response:

```json
{
    "status": "success",
    "data": {
        "symbol": "AAPL",
        "name": "Apple Inc.",
        "currentPrice": 175.50,
        "dayHigh": 176.30,
        "dayLow": 174.80,
        "marketCap": 2750000000000,
        "volume": 12345678,
        "fiftyTwoWeekHigh": 180.00,
        "fiftyTwoWeekLow": 140.00,
        "trailingPE": 28.5,
        "dividendYield": 0.65
    }
}
```

Error Response:

```json
{
    "error": "Error message here",
    "status": "error"
}
```

### Get Stock Prediction

Get advanced stock price predictions using machine learning and technical analysis.

```http
GET http://127.0.0.1:5000/stock/prediction?ticker={symbol}
```

This endpoint uses a sophisticated ensemble of machine learning models and technical indicators to predict stock price movements. The prediction system includes:

- 2 years of historical data analysis
- Multiple technical indicators (RSI, MACD, Bollinger Bands, etc.)
- Advanced momentum and trend analysis
- Ensemble ML models (Random Forest, XGBoost, Gradient Boosting)
- Time series cross-validation
- Dynamic model weighting based on performance

Parameters:

- `ticker` (required): Stock ticker symbol (e.g., AAPL, MSFT)

Example Response:

```json
{
  "data": {
    "confidence": 0.9490711014853327,
    "current_price": 163.99000549316406,
    "model_metrics": {
      "avg_gb_mape": 0.12209361041418505,
      "avg_rf_mape": 0.13017329335440375,
      "avg_xgb_mape": 0.1454700128365251
    },
    "predicted_price": 122.19888626062838,
    "prediction_interval": {
      "lower": 57.15579293348053,
      "upper": 187.24197958777626
    },
    "technical_signals": {
      "adx_trend_strength": 2597.9239502754817,
      "macd_signal": "bullish",
      "rsi_signal": "overbought",
      "volatility": 0.5711223344851456,
      "volume_trend": "decreasing"
    },
    "trend": {
      "direction": "down",
      "market_context": "bullish",
      "momentum": 5.364948686880422,
      "resistance_level": 146.34227668762207,
      "strength": 0.2548394282130672,
      "support_level": 135.37771690368652
    }
  },
  "status": "success"
}
```

Response Fields:

- `current_price`: Current stock price
- `predicted_price`: ML model ensemble's price prediction
- `confidence`: Prediction confidence score (0-1)
- `prediction_interval`: 95% confidence interval for the prediction
- `trend`:
  - `direction`: Price trend direction
  - `strength`: Trend strength indicator
  - `momentum`: Short-term price momentum
  - `market_context`: Overall market trend
  - `support_level`: Key support price level
  - `resistance_level`: Key resistance price level
- `technical_signals`:
  - `rsi_signal`: RSI-based market condition
  - `macd_signal`: MACD trend signal
  - `adx_trend_strength`: ADX trend strength (0-100)
  - `volume_trend`: Volume trend analysis
  - `volatility`: Annualized volatility
- `model_metrics`: Performance metrics for each model (MAPE)

Error Response:

```json
{
    "error": "Error message here",
    "status": "error"
}
```

## Rate Limits

- Global: 100 requests per day, 10 per minute
- Stock data endpoint: 5 requests per minute
- Stock prediction endpoint: 5 requests per minute
