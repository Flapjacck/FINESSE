# FINESSE
A simple yet powerful Flask-based API that delivers real-time stock news headlines using Yahoo Finance data.

## 🌟 Overview

FINESSE is an advanced stock analysis platform that combines news sentiment analysis, technical indicators, and fundamental metrics to provide comprehensive trading insights. Built with Python and Flask, it leverages machine learning and financial analysis to help investors make informed decisions.

### Key Features

🔍 **Comprehensive Analysis**
- Real-time news sentiment analysis
- Technical indicators and trading signals
- Fundamental metrics and health scoring
- Smart trading recommendations

📊 **Technical Analysis**
- RSI and MACD indicators
- Multiple timeframe moving averages
- Volume trend analysis
- Price momentum tracking

🤖 **Sentiment Analysis**
- Natural language processing of news
- Confidence-weighted scoring
- Trend detection
- Aggregated market sentiment

📈 **Smart Recommendations**
- AI-driven trading signals
- Multi-factor analysis
- Confidence scoring
- Human-readable summaries

🎯 **Decision Support**
- Company health scoring
- Market trend analysis
- Risk assessment
- Performance metrics

## Sentiment Analysis

The API uses DistilBERT (specifically `distilbert-base-uncased-finetuned-sst-2-english`) for sentiment analysis of news headlines. 

### How Sentiment Scoring Works

Each news headline receives two sentiment indicators:

1. **Label**: Either "POSITIVE" or "NEGATIVE"
2. **Score**: A confidence value between 0 and 1
   - Closer to 1 = Higher confidence
   - Example: 0.9987 means 99.87% confidence in the classification

### Understanding the Results

- High scores (>0.95) indicate strong confidence in the sentiment classification
- Same confidence scores can have different labels because:
  - Question headlines often receive negative sentiment due to implied uncertainty
  - Direct statements tend to receive positive sentiment
  - The model analyzes full sentence structure and context
  - Word choice and syntax affect classification

### Example Interpretations

```json
{
    "title": "Should Stock X Be in Your Portfolio?",
    "sentiment": {
        "label": "NEGATIVE",
        "score": 0.998
    }
}
```
↑ Question format suggests uncertainty = Negative sentiment

```json
{
    "title": "Stock X Among Best Performers",
    "sentiment": {
        "label": "POSITIVE",
        "score": 0.999
    }
}
```
↑ Direct positive statement = Positive sentiment

## Coming Soon

- Historical sentiment tracking
- Web interface
- Stock price correlation with news sentiment

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

## API Documentation

The API provides the following endpoints:

### Health Check

Check if the API is up and running.

```
GET /health
```

Example:
```
http://localhost:5000/health
```

Response:
```json
{
    "status": "healthy",
    "timestamp": "2025-05-20T10:00:00.000Z",
    "version": "1.0.0"
}
```

### Get Stock Analysis and News

Retrieve comprehensive stock analysis including news articles, technical indicators, sentiment analysis, and trading recommendations.

```
GET /news?ticker={symbol}&limit={count}
```

Parameters:

- `ticker` (required): Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- `limit` (optional): Number of news articles to return (default: 15, max: 50)

Example:
```
http://localhost:5000/analyze?ticker=AAPL&limit=10
```

Success Response:

```json
{
    "meta": {
        "ticker": "AAPL",
        "count": 10,
        "status": "success",
        "timestamp": "2025-05-20T10:00:00.000Z",
        "request_id": "AAPL-20250520-100000"
    },
    "data": {
        "news": [
            {
                "title": "Example News Title",
                "publisher": "Publisher Name",
                "link": "https://example.com/news",
                "published": "2025-05-20T09:30:00.000Z",
                "summary": "News article summary...",
                "source": "Source Name",
                "type": "article",
                "sentiment": {
                    "label": "positive",
                    "score": 0.9234,
                    "nuanced_score": 0.8567,
                    "confidence": "very high"
                }
            }
        ],
        "analysis": {
            "sentiment": {
                "summary": {
                    "positive_percentage": 75.5,
                    "negative_percentage": 15.5,
                    "neutral_percentage": 9.0,
                    "overall_sentiment": 0.654,
                    "average_confidence": 0.891
                },
                "trend": {
                    "direction": "positive",
                    "strength": 0.654,
                    "confidence": 0.891
                }
            },
            "technical": {
                "indicators": {
                    "rsi": 65.4,
                    "macd": {
                        "value": 2.45,
                        "signal": 1.89,
                        "histogram": 0.56
                    },
                    "moving_averages": {
                        "sma_20": 185.45,
                        "sma_50": 180.67,
                        "sma_200": 175.89
                    }
                },
                "signals": {
                    "rsi": "neutral",
                    "trend": "bullish",
                    "momentum": "bullish",
                    "volume": "neutral"
                },
                "momentum": {
                    "short_term": 2.67,
                    "long_term": 5.45
                }
            },
            "fundamental": {
                "metrics": {
                    "pe_ratio": 28.5,
                    "market_cap": 2950000000000,
                    "beta": 1.21,
                    "dividend_yield": 0.0065,
                    "revenue_growth": 0.15,
                    "profit_margins": 0.25
                },
                "health_score": 0.8756
            }
        },
        "recommendation": {
            "action": "STRONG_BUY",
            "confidence": 0.8912,
            "factors": {
                "sentiment": 0.654,
                "technical": 0.789,
                "combined": 0.721
            },
            "summary": "Strong buy recommendation based on bullish technical signals (momentum, trend) and positive market sentiment"
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

#### Analysis Components

1. **Sentiment Analysis**
   - News sentiment scoring (-1 to 1)
   - Confidence levels
   - Trend analysis
   - Aggregated statistics

2. **Technical Analysis**
   - RSI (Relative Strength Index)
   - Moving Averages (20, 50, 200 day)
   - Volume Analysis
   - Momentum Indicators
   - Trading Signals

3. **Fundamental Analysis**
   - Key Financial Metrics
   - Market Position
   - Growth Indicators
   - Company Health Score

4. **Recommendations**
   - Action (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
   - Confidence Score
   - Contributing Factors
   - Human-readable Summary

Rate Limits:

- Global: 100 requests per day, 10 per minute
- News endpoint: 5 requests per minute