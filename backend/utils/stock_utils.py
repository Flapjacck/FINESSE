import yfinance as yf
from typing import Dict, Any, Optional

def get_stock_info(ticker: str) -> Dict[str, Any]:
    """
    Get comprehensive stock information using yfinance.
    
    Args:
        ticker (str): The stock ticker symbol
        
    Returns:
        Dict[str, Any]: Dictionary containing stock information
        
    Raises:
        Exception: If there's an error fetching the stock data
    """
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)
        
        # Get basic info
        info = stock.info
        
        # Get recent market data
        hist = stock.history(period="1d")
        
        # Compile relevant information
        stock_data = {
            # Basic information
            "symbol": ticker,
            "name": info.get("longName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            
            # Price information
            "currentPrice": info.get("currentPrice"),
            "previousClose": info.get("previousClose"),
            "open": info.get("open"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            
            # Market data
            "marketCap": info.get("marketCap"),
            "volume": info.get("volume"),
            "avgVolume": info.get("averageVolume"),
            
            # Financial metrics
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "dividendYield": info.get("dividendYield"),
            "beta": info.get("beta"),
            
            # Additional metrics
            "fiftyDayAverage": info.get("fiftyDayAverage"),
            "twoHundredDayAverage": info.get("twoHundredDayAverage"),
            
            # Latest price from history
            "latestPrice": hist["Close"].iloc[-1] if not hist.empty else None
        }
        
        return {k: v for k, v in stock_data.items() if v is not None}
        
    except Exception as e:
        raise Exception(f"Error fetching data for ticker {ticker}: {str(e)}")
