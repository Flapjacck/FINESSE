from flask import Flask, request, jsonify
import yfinance as yf

app = Flask(__name__)

@app.route('/news', methods=['GET'])
def get_stock_news():
    # Get ticker from query parameter
    ticker = request.args.get('ticker')
    
    # Return error if no ticker provided
    if not ticker:
        return jsonify({
            'error': 'Please provide a ticker parameter'
        }), 400
    
    try:        # Create ticker object
        stock = yf.Ticker(ticker)
        
        # Debug: Print the news data structure
        print(f"Fetching news for {ticker}...")
          # Get news data (using get_news method with a limit of 5)
        news = stock.get_news(count=5)
        
        # Extract headlines and relevant info
        top_news = []
        for item in news:
            # Basic validation
            if not isinstance(item, dict):
                continue
            
            try:
                # Get the content object which contains the actual news data
                content = item.get('content', {})
                if not content:
                    continue
                
                # Get provider info
                provider = content.get('provider', {})
                click_through = content.get('clickThroughUrl', {})
                
                news_item = {
                    'title': content.get('title'),
                    'publisher': provider.get('displayName'),
                    'link': click_through.get('url'),
                    'published': content.get('pubDate')
                }
                
                # Only append if we have required fields
                if news_item['title']:
                    top_news.append(news_item)
            except Exception as e:
                print(f"Error processing news item: {str(e)}")
                continue
        
        # Check if we were able to extract any news
        if not top_news:
            return jsonify({
                'error': f'No news articles found for {ticker}'
            }), 404

        return jsonify({
            'ticker': ticker.upper(),
            'news': top_news
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Error fetching news for {ticker}: {str(e)}'
        }), 500

if __name__ == '__main__':
    # Run the Flask app in debug mode
    app.run(debug=True, port=5000)
