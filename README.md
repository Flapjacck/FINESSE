# FINESSE
A simple yet powerful Flask-based API that delivers real-time stock news headlines using Yahoo Finance data.

## 🌟 Overview

FINESSE helps investors and traders stay informed by providing instant access to the latest news headlines for any stock ticker. Built with Python and Flask, it leverages the `yfinance` library to fetch accurate and up-to-date financial news.

## Features

- Real-time stock news retrieval with customizable limits
- Rate limiting for API protection
- CORS support for web applications
- Input validation and error handling
- Detailed metadata for each news article
- Health check endpoint for monitoring
- Live data from Yahoo Finance

## Coming Soon

- AI-powered sentiment analysis of news headlines
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
pip install flask yfinance
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
    "timestamp": "2025-05-18T10:00:00.000Z",
    "version": "1.0.0"
}
```

### Get Stock News

Retrieve news articles for a specific stock ticker.

```
GET /news?ticker={symbol}&limit={count}
```

Parameters:

- `ticker` (required): Stock ticker symbol (e.g., AAPL, MSFT, GOOGL)
- `limit` (optional): Number of news articles to return (default: 15, max: 50)

Example:
```
http://localhost:5000/news?ticker=AAPL&limit=10
```

Success Response:

```json
{
    "ticker": "AAPL",
    "count": 10,
    "status": "success",
    "timestamp": "2025-05-18T10:00:00.000Z",
    "news": [
        {
            "title": "Example News Title",
            "publisher": "Publisher Name",
            "link": "https://example.com/news",
            "published": "2025-05-18T09:00:00.000Z",
            "summary": "News article summary...",
            "source": "Source Name",
            "type": "article"
        }
        // ... more news items
    ]
}
```

Rate Limits:

- Global: 100 requests per day, 10 per minute
- News endpoint: 5 requests per minute