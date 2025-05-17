# FINESSE
A simple yet powerful Flask-based API that delivers real-time stock news headlines using Yahoo Finance data.

## 🌟 Overview

FINESSE helps investors and traders stay informed by providing instant access to the latest news headlines for any stock ticker. Built with Python and Flask, it leverages the `yfinance` library to fetch accurate and up-to-date financial news.

## Features

- Real-time stock news retrieval
- Top 5 most recent headlines per request
- Simple and intuitive API endpoint
- Fast response times
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

