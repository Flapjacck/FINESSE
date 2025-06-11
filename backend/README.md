# Stock Screener Backend

A Flask-based REST API for screening stocks based on fundamental analysis criteria.

## Features

- **Stock Screening**: Filter stocks based on various financial metrics
- **Predefined Presets**: Ready-to-use screening criteria for different investment strategies
- **Stock Details**: Get detailed information about specific stocks
- **RESTful API**: Clean JSON API for easy integration

## Screening Criteria Supported

- Market Cap (min/max)
- P/E Ratio (min/max)
- Dividend Yield (minimum)
- Trading Volume (minimum)
- Debt-to-Equity Ratio (maximum)
- Price-to-Book Ratio (maximum)
- Return on Equity (minimum)
- Sector filtering
- Beta (maximum)

## Setup and Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application:**

   ```bash
   python app.py
   ```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check

```bash
GET /api/health
```

**Browser URL:** [http://localhost:5000/api/health](http://localhost:5000/api/health)

Returns the API status and timestamp.

## Data Source

This application uses Yahoo Finance API through the `yfinance` library to fetch real-time stock data and fundamental metrics.
