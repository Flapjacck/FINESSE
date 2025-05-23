# FINESSE

A comprehensive stock market data platform built with Flask and React, providing real-time financial data through the Yahoo Finance API.

## 🌟 Overview

FINESSE is a modern stock market data platform that delivers comprehensive financial information about stocks. Built with Python (Flask) backend and React frontend, it provides an intuitive interface to access market data and stock information.

## Getting Started

### Prerequisites

- [Uses yFinance API](https://github.com/ranaroussi/yfinance)
- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:

```powershell
git clone https://github.com/flapjacck/finesse.git
cd finesse
```

2. Create and activate a virtual environment:

```powershell
python -m venv venv
.\venv\Scripts\activate
```

3. Install required packages:

```powershell
cd backend
pip install flask flask-cors flask-limiter yfinance transformers torch numpy pandas
```

4. Run the server:

```powershell
python app.py
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
GET /stock/data?ticker={symbol}
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

Get stock movement prediction for a specific ticker symbol.

```http
GET /stock/prediction?ticker={symbol}
```

Parameters:

- `ticker` (required): Stock ticker symbol (e.g., AAPL, MSFT)

Example Response:

```json
{
    "status": "success",
    "data": {
        "ticker": "AAPL",
        "current_price": 175.50,
        "predicted_direction": "upward",
        "confidence_score": 85.42,
        "prediction_date": "2025-05-22",
        "metrics": {
            "average_price_30d": 172.30,
            "volatility_30d": 2.15,
            "recent_trend": 1.85
        }
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

## Rate Limits

- Global: 100 requests per day, 10 per minute
- Stock data endpoint: 5 requests per minute
- Stock prediction endpoint: 5 requests per minute
