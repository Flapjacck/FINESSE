from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import yfinance as yf
from datetime import datetime
import re

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

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

@app.route('/stock/prediction', methods=['GET'])
@limiter.limit("5 per minute")  # Specific rate limit for this endpoint
def get_stock_prediction():
    """
    Get stock prediction for a given ticker symbol.
    
    Query Parameters:
        ticker (str): Stock ticker symbol
        
    Returns:
        JSON response containing prediction data or error message
    """
    ticker = request.args.get('ticker')
    
    if not ticker:
        return jsonify({
            'error': 'Ticker symbol is required',
            'status': 'error'
        }), 400
        
    if not validate_ticker(ticker):
        return jsonify({
            'error': 'Invalid ticker format',
            'status': 'error'
        }), 400
        
    try:
        from utils.prediction_utils import get_stock_prediction as get_prediction
        prediction_data = get_prediction(ticker.upper())
        
        return jsonify({
            'data': prediction_data,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/stock/data', methods=['GET'])
@limiter.limit("5 per minute")  # Specific rate limit for this endpoint
def get_stock_data():
    """
    Get comprehensive stock data for a given ticker symbol.
    
    Query Parameters:
        ticker (str): Stock ticker symbol
        
    Returns:
        JSON response containing stock data or error message
    """
    ticker = request.args.get('ticker')
    
    if not ticker:
        return jsonify({
            'error': 'Ticker symbol is required',
            'status': 'error'
        }), 400
        
    if not validate_ticker(ticker):
        return jsonify({
            'error': 'Invalid ticker format',
            'status': 'error'
        }), 400
        
    try:
        from utils.stock_utils import get_stock_info
        stock_data = get_stock_info(ticker.upper())
        
        return jsonify({
            'data': stock_data,
            'status': 'success'
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
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
