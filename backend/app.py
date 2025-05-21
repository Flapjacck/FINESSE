from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import yfinance as yf
from datetime import datetime
import re
from utils.sentiment_analyzer import sentiment_analyzer

def calculate_health_score(fundamentals: dict) -> float:
    """
    Calculate a health score from fundamental metrics.
    
    Args:
        fundamentals (dict): Dictionary containing fundamental metrics
        
    Returns:
        float: Health score between 0 and 1
    """
    if not fundamentals:
        return 0.5
        
    # Define weights for different metrics
    weights = {
        'pe_ratio': 0.2,
        'profit_margins': 0.25,
        'revenue_growth': 0.25,
        'beta': 0.15,
        'dividend_yield': 0.15
    }
    
    # Calculate weighted score
    score = 0.5  # Default neutral score
    total_weight = 0
    
    if fundamentals.get('pe_ratio'):
        pe_score = min(1.0, 15 / fundamentals['pe_ratio']) if fundamentals['pe_ratio'] > 0 else 0
        score += pe_score * weights['pe_ratio']
        total_weight += weights['pe_ratio']
    
    if fundamentals.get('profit_margins'):
        profit_score = min(1.0, fundamentals['profit_margins'] * 5)
        score += profit_score * weights['profit_margins']
        total_weight += weights['profit_margins']
    
    if fundamentals.get('revenue_growth'):
        growth_score = min(1.0, fundamentals['revenue_growth'] * 2)
        score += growth_score * weights['revenue_growth']
        total_weight += weights['revenue_growth']
    
    if fundamentals.get('beta'):
        beta_score = 1 - min(1.0, abs(fundamentals['beta'] - 1))
        score += beta_score * weights['beta']
        total_weight += weights['beta']
    
    if fundamentals.get('dividend_yield'):
        div_score = min(1.0, fundamentals['dividend_yield'] / 0.05)
        score += div_score * weights['dividend_yield']
        total_weight += weights['dividend_yield']
    
    # Normalize score based on available metrics
    return round(score / total_weight if total_weight > 0 else 0.5, 4)

def generate_recommendation_summary(recommendation: dict, sentiment: dict, technical: dict) -> str:
    """
    Generate a human-readable summary of the stock recommendation.
    
    Args:
        recommendation (dict): Recommendation data
        sentiment (dict): Sentiment analysis data
        technical (dict): Technical analysis data
        
    Returns:
        str: A concise summary of the recommendation
    """
    action = recommendation['action']
    confidence = recommendation['confidence']
    
    # Get the primary factors driving the recommendation
    sentiment_signal = "positive" if sentiment['overall_sentiment'] > 0 else "negative"
    tech_signals = [signal for signal, value in technical.get('signals', {}).items() 
                   if value in ['bullish', 'oversold']]
    
    # Generate summary based on confidence and signals
    if confidence >= 0.8:
        strength = "Strong"
    elif confidence >= 0.6:
        strength = "Moderate"
    else:
        strength = "Weak"
        
    summary = f"{strength} {action} recommendation based on "
    
    factors = []
    if tech_signals:
        factors.append(f"bullish technical signals ({', '.join(tech_signals)})")
    if abs(sentiment['overall_sentiment']) > 0.3:
        factors.append(f"{sentiment_signal} market sentiment")
        
    if factors:
        summary += " and ".join(factors)
    else:
        summary += "mixed market signals"
        
    return summary

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

@app.route('/stock/analysis', methods=['GET'])
@limiter.limit("5 per minute")  # Specific rate limit for this endpoint
def get_stock_analysis():
    """
    Fetch latest news articles and comprehensive analysis for a given stock ticker.
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
        
        # Process news items and collect titles for batch sentiment analysis
        titles = []
        news_items = []
        
        for item in news:
            if not isinstance(item, dict):
                continue
            
            try:
                content = item.get('content', {})
                if not content:
                    continue
                
                provider = content.get('provider', {})
                click_through = content.get('clickThroughUrl', {})
                title = content.get('title', '')
                
                # Create news item without sentiment
                news_item = {
                    'title': title,
                    'publisher': provider.get('displayName'),
                    'link': click_through.get('url'),
                    'published': content.get('pubDate'),
                    'summary': content.get('description', ''),
                    'source': provider.get('name'),
                    'type': content.get('type', 'article')
                }
                  # Only include items with required fields
                if all(news_item[key] for key in ['title', 'link']) and title:
                    titles.append(title)
                    news_items.append(news_item)
                    
            except Exception as e:
                print(f"[{datetime.now().isoformat()}] Error processing news item: {str(e)}")
                continue
        
        if not titles:
            return jsonify({
                'error': f'No valid news articles found for {ticker}',
                'status': 'error'
            }), 404        # Perform comprehensive stock analysis with market sentiment
        analysis_results = sentiment_analyzer.analyze_stock(ticker, titles)
        
        # Enhance news items with sentiment data and format timestamps
        formatted_news = []
        for item, sentiment in zip(news_items, analysis_results['sentiment_analysis']['individual_sentiments']):
            # Convert timestamp to ISO format for frontend consistency
            try:
                published_date = datetime.fromisoformat(item['published'].replace('Z', '+00:00'))
                formatted_date = published_date.isoformat()
            except:
                formatted_date = datetime.now().isoformat()

            formatted_news.append({
                'title': item['title'],
                'publisher': item['publisher'],
                'link': item['link'],
                'published': formatted_date,
                'summary': item['summary'],
                'source': item['source'],
                'type': item['type'],
                'sentiment': {
                    'label': sentiment['label'],
                    'score': round(sentiment['score'], 4),
                    'nuanced_score': round(sentiment['nuanced_score'], 4),
                    'confidence': sentiment['confidence']
                }
            })

        # Format the response data for frontend consumption
        response_data = {
            'meta': {
                'ticker': ticker,
                'count': len(formatted_news),
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'request_id': f"{ticker}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            },
            'data': {
                'news': formatted_news,
                'analysis': {
                    'sentiment': {
                        'summary': analysis_results['sentiment_analysis']['statistics'],
                        'trend': {
                            'direction': 'positive' if analysis_results['sentiment_analysis']['statistics']['overall_sentiment'] > 0 else 'negative',
                            'strength': abs(analysis_results['sentiment_analysis']['statistics']['overall_sentiment']),
                            'confidence': analysis_results['sentiment_analysis']['statistics']['average_confidence']
                        }
                    },
                    'technical': {
                        'indicators': analysis_results['technical_analysis'],
                        'signals': analysis_results['technical_analysis'].get('signals', {}),
                        'momentum': {
                            'short_term': analysis_results['technical_analysis'].get('relative_strength', {}).get('short_term', 0),
                            'long_term': analysis_results['technical_analysis'].get('relative_strength', {}).get('long_term', 0)
                        }
                    },
                    'fundamental': {
                        'metrics': analysis_results['fundamentals'],
                        'health_score': calculate_health_score(analysis_results['fundamentals'])
                    }
                },
                'recommendation': {
                    'action': analysis_results['recommendation']['action'].upper(),
                    'confidence': round(analysis_results['recommendation']['confidence'], 4),
                    'factors': analysis_results['recommendation']['factors'],
                    'summary': generate_recommendation_summary(
                        analysis_results['recommendation'],
                        analysis_results['sentiment_analysis']['statistics'],
                        analysis_results['technical_analysis']
                    )
                }
            }
        }

        return jsonify(response_data)
    
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
