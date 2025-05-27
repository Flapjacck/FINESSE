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

Get comprehensive stock data for a specific ticker symbol. The endpoint provides extensive financial information organized into several categories:

- Basic company information (name, description, sector, industry, etc.)
- Current and historical price data
- Market statistics and trading information
- Key financial metrics and ratios
- Technical indicators
- Analyst recommendations and price targets

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
        "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "website": "https://www.apple.com",
        "country": "United States",
        "fullTimeEmployees": 164000,
        
        "currentPrice": 175.50,
        "previousClose": 174.80,
        "open": 175.00,
        "dayHigh": 176.30,
        "dayLow": 174.80,
        "regularMarketPrice": 175.50,
        "preMarketPrice": 175.20,
        "postMarketPrice": 175.60,
        "dayChange": 0.70,
        "dayChangePercent": 0.40,
        "fiftyTwoWeekHigh": 180.00,
        "fiftyTwoWeekLow": 140.00,
        
        "marketCap": 2750000000000,
        "volume": 12345678,
        "avgVolume": 15000000,
        "sharesOutstanding": 15680000000,
        "floatShares": 15670000000,
        
        "trailingPE": 28.5,
        "forwardPE": 26.8,
        "priceToBook": 44.5,
        "profitMargins": 0.25,
        "operatingMargins": 0.30,
        "grossMargins": 0.43,
        "dividendYield": 0.65,
        "dividendRate": 0.96,
        "payoutRatio": 0.15,
        "beta": 1.28,
        "enterpriseValue": 2800000000000,
        "enterpriseToEbitda": 21.5,
        "forwardEps": 6.55,
        "trailingEps": 6.15,
        "bookValue": 3.94,
        "debtToEquity": 168.9,
        "currentRatio": 1.04,
        "quickRatio": 0.88,
        "returnOnEquity": 0.49,
        "returnOnAssets": 0.25,
        
        "fiftyDayAverage": 173.45,
        "twoHundredDayAverage": 170.80,
        "averageVolume10days": 14500000,
        
        "targetHighPrice": 190.00,
        "targetLowPrice": 160.00,
        "targetMeanPrice": 175.00,
        "recommendationMean": 2.0,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 35
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

Get advanced stock price predictions for multiple time frames using machine learning and technical analysis.

```http
GET http://127.0.0.1:5000/stock/prediction?ticker={symbol}
```

This endpoint uses a sophisticated ensemble of machine learning models to predict stock trends across different time periods. The prediction system includes:

- 5 years of historical data analysis for reliable long-term predictions
- Multiple technical indicators (RSI, MACD, Bollinger Bands)
- Time-based predictions for different horizons (1 day, 1 week, 1 month, 1 year)
- Ensemble ML models (Random Forest, XGBoost, Gradient Boosting)
- Dynamic model complexity adjustment based on data availability
- Adaptive feature scaling and robust error handling
- Confidence-based trend classification

Parameters:

- `ticker` (required): Stock ticker symbol (e.g., AAPL, MSFT)

Example Response:

```json
{
  "data": {
    "current_price": 163.99,
    "time_based_predictions": {
      "1d": {
        "trend": "bullish",
        "confidence": 0.85,
        "predicted_price": 165.75,
        "percent_change": 1.07
      },
      "1w": {
        "trend": "neutral",
        "confidence": 0.78,
        "predicted_price": 164.50,
        "percent_change": 0.31
      },
      "1m": {
        "trend": "bearish",
        "confidence": 0.72,
        "predicted_price": 155.30,
        "percent_change": -5.30
      },
      "1y": {
        "trend": "bullish",
        "confidence": 0.65,
        "predicted_price": 180.80,
        "percent_change": 10.25
      }
    },
    "technical_signals": {
      "rsi_signal": "neutral",
      "macd_signal": "bullish",
      "volatility": 0.15,
      "volume_trend": "increasing"
    },
    "timestamp": "2025-05-27T10:30:00.000Z"
  },
  "status": "success"
}
```

Response Fields:

- `current_price`: Current stock price
- `time_based_predictions`: Predictions for different time periods
  - `1d`: One day prediction
  - `1w`: One week prediction
  - `1m`: One month prediction
  - `1y`: One year prediction
  For each time period:
  - `trend`: Direction prediction (bullish/bearish/neutral)
  - `confidence`: Prediction confidence score (0-1)
  - `predicted_price`: Expected price at end of period
  - `percent_change`: Expected percentage change
- `technical_signals`: Current market indicators
  - `rsi_signal`: RSI-based condition (overbought/oversold/neutral)
  - `macd_signal`: MACD trend signal (bullish/bearish)
  - `volatility`: Current price volatility
  - `volume_trend`: Volume trend analysis (increasing/decreasing)
- `timestamp`: Time of prediction

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
