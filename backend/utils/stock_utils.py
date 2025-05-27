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
            "description": info.get("longBusinessSummary"),
            "website": info.get("website"),
            "country": info.get("country"),
            "fullTimeEmployees": info.get("fullTimeEmployees"),
            
            # Price information
            "currentPrice": info.get("currentPrice"),
            "previousClose": info.get("previousClose"),
            "open": info.get("open"),
            "dayHigh": info.get("dayHigh"),
            "dayLow": info.get("dayLow"),
            "fiftyTwoWeekHigh": info.get("fiftyTwoWeekHigh"),
            "fiftyTwoWeekLow": info.get("fiftyTwoWeekLow"),
            "regularMarketPrice": info.get("regularMarketPrice"),
            "preMarketPrice": info.get("preMarketPrice"),
            "postMarketPrice": info.get("postMarketPrice"),
            
            # Market data
            "marketCap": info.get("marketCap"),
            "volume": info.get("volume"),
            "avgVolume": info.get("averageVolume"),
            "sharesOutstanding": info.get("sharesOutstanding"),
            "floatShares": info.get("floatShares"),
            
            # Financial metrics
            "trailingPE": info.get("trailingPE"),
            "forwardPE": info.get("forwardPE"),
            "priceToBook": info.get("priceToBook"),
            "profitMargins": info.get("profitMargins"),
            "operatingMargins": info.get("operatingMargins"),
            "grossMargins": info.get("grossMargins"),
            "dividendYield": info.get("dividendYield"),
            "dividendRate": info.get("dividendRate"),
            "payoutRatio": info.get("payoutRatio"),
            "beta": info.get("beta"),
            "enterpriseValue": info.get("enterpriseValue"),
            "enterpriseToEbitda": info.get("enterpriseToEbitda"),
            "forwardEps": info.get("forwardEps"),
            "trailingEps": info.get("trailingEps"),
            "bookValue": info.get("bookValue"),
            "debtToEquity": info.get("debtToEquity"),
            "currentRatio": info.get("currentRatio"),
            "quickRatio": info.get("quickRatio"),
            "returnOnEquity": info.get("returnOnEquity"),
            "returnOnAssets": info.get("returnOnAssets"),
            
            # Technical indicators
            "fiftyDayAverage": info.get("fiftyDayAverage"),
            "twoHundredDayAverage": info.get("twoHundredDayAverage"),
            "averageVolume10days": info.get("averageVolume10days"),
            "relativeStrengthIndex": info.get("relativeStrengthIndex"),
            
            # Analyst recommendations
            "targetHighPrice": info.get("targetHighPrice"),
            "targetLowPrice": info.get("targetLowPrice"),
            "targetMeanPrice": info.get("targetMeanPrice"),
            "recommendationMean": info.get("recommendationMean"),
            "recommendationKey": info.get("recommendationKey"),
            "numberOfAnalystOpinions": info.get("numberOfAnalystOpinions"),
            
            # Latest price from history
            "latestPrice": hist["Close"].iloc[-1] if not hist.empty else None,
            
            # Additional price metrics
            "dayChange": (info.get("currentPrice", 0) - info.get("previousClose", 0)) if info.get("currentPrice") and info.get("previousClose") else None,
            "dayChangePercent": ((info.get("currentPrice", 0) - info.get("previousClose", 0)) / info.get("previousClose", 1) * 100) if info.get("currentPrice") and info.get("previousClose") else None
        }
        
        return {k: v for k, v in stock_data.items() if v is not None}
        
    except Exception as e:
        raise Exception(f"Error fetching data for ticker {ticker}: {str(e)}")
