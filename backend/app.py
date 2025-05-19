from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import yfinance as yf
from datetime import datetime
import re
from utils.sentiment_analyzer import sentiment_analyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize sentiment analyzer when app starts
sentiment_analyzer.initialize()

# Configure rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["100 per day", "10 per minute"]
)

def validate_ticker(ticker):
    """
    Validate the ticker symbol format.
    Args:
        ticker (str): Stock ticker symbol
    Returns:
        bool: True if valid, False otherwise
    
    Supports:
    - US tickers (e.g., AAPL, TSLA)
    - International tickers with extensions (e.g., CLS.TO)
    - Longer international tickers
    - Tickers with hyphens
    """
    if not ticker or not isinstance(ticker, str):
        return False
    # Extended ticker format validation
    # - Up to 8 characters for main symbol
    # - Optional .XX or -XX suffix for international markets
    # - No special characters except dots and hyphens
    return bool(re.match(r'^[A-Za-z0-9]{1,8}(?:[.-][A-Za-z0-9]{1,4})?$', ticker))

@app.route('/news', methods=['GET'])
@limiter.limit("5 per minute")  # Specific rate limit for this endpoint
def get_stock_news():
    """
    Fetch latest news articles for a given stock ticker.
    Query Parameters:
        ticker (str): Stock ticker symbol (required)
        limit (int): Number of news articles to return (optional, default=15, max=50)
    Returns:
        JSON object containing news articles or error message
    """
    # Get and validate ticker from query parameter
    ticker = request.args.get('ticker', '').strip().upper()
    if not validate_ticker(ticker):
        return jsonify({
            'error': 'Invalid ticker format. Please provide a valid stock symbol.',
            'status': 'error'
        }), 400
    
    # Get and validate limit parameter
    try:
        limit = min(int(request.args.get('limit', 15)), 50)
    except ValueError:
        limit = 15
    
    try:
        # Create ticker object and fetch news
        stock = yf.Ticker(ticker)
        print(f"[{datetime.now().isoformat()}] Fetching news for {ticker}...")
        news = stock.get_news(count=limit)
        
        # Extract headlines and relevant info
        top_news = []
        for item in news:
            if not isinstance(item, dict):
                continue
            
            try:
                content = item.get('content', {})
                if not content:
                    continue
                
                provider = content.get('provider', {})
                click_through = content.get('clickThroughUrl', {})
                
                # Get the title for sentiment analysis
                title = content.get('title', '')
                
                # Enhanced news item with metadata and sentiment analysis
                news_item = {
                    'title': title,
                    'publisher': provider.get('displayName'),
                    'link': click_through.get('url'),
                    'published': content.get('pubDate'),
                    'summary': content.get('description', ''),
                    'source': provider.get('name'),
                    'type': content.get('type', 'article'),
                    'sentiment': sentiment_analyzer.analyze_text(title)
                }
                
                # Only append if we have required fields
                if all(news_item[key] for key in ['title', 'link']):
                    top_news.append(news_item)
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] Error processing news item: {str(e)}")
                continue
        
        # Check if we were able to extract any news
        if not top_news:
            return jsonify({
                'error': f'No news articles found for {ticker}',
                'status': 'error'
            }), 404

        return jsonify({
            'ticker': ticker,
            'count': len(top_news),
            'status': 'success',
            'timestamp': datetime.now().isoformat(),
            'news': top_news
        })
    
    except Exception as e:
        print(f"[{datetime.now().isoformat()}] Error fetching news for {ticker}: {str(e)}")
        return jsonify({
            'error': f'Error fetching news for {ticker}: {str(e)}',
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """
    Simple health check endpoint to verify API status
    Returns:
        JSON object containing API status and version information
    """
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded errors"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'status': 'error',
        'retry_after': e.description
    }), 429

if __name__ == '__main__':
    app.run(debug=True, port=5000)
