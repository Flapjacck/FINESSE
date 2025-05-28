export interface StockData {
    // Basic information
    symbol: string;
    name: string;
    sector: string | null;
    industry: string | null;
    description: string | null;
    website: string | null;
    country: string | null;
    fullTimeEmployees: number | null;
    
    // Price information
    currentPrice: number | null;
    previousClose: number | null;
    open: number | null;
    dayHigh: number | null;
    dayLow: number | null;
    fiftyTwoWeekHigh: number | null;
    fiftyTwoWeekLow: number | null;
    regularMarketPrice: number | null;
    preMarketPrice: number | null;
    postMarketPrice: number | null;
    
    // Market data
    marketCap: number | null;
    volume: number | null;
    avgVolume: number | null;
    sharesOutstanding: number | null;
    floatShares: number | null;
    
    // Financial metrics
    trailingPE: number | null;
    forwardPE: number | null;
    priceToBook: number | null;
    profitMargins: number | null;
    operatingMargins: number | null;
    grossMargins: number | null;
    dividendYield: number | null;
    dividendRate: number | null;
    payoutRatio: number | null;
    beta: number | null;
    enterpriseValue: number | null;
    enterpriseToEbitda: number | null;
    forwardEps: number | null;
    trailingEps: number | null;
    bookValue: number | null;
    debtToEquity: number | null;
    currentRatio: number | null;
    quickRatio: number | null;
    returnOnEquity: number | null;
    returnOnAssets: number | null;
    
    // Technical indicators
    fiftyDayAverage: number | null;
    twoHundredDayAverage: number | null;
    averageVolume10days: number | null;
    relativeStrengthIndex: number | null;
    
    // Analyst recommendations
    targetHighPrice: number | null;
    targetLowPrice: number | null;
    targetMeanPrice: number | null;
    recommendationMean: number | null;
    recommendationKey: string | null;
    numberOfAnalystOpinions: number | null;
    
    // Latest price info
    latestPrice: number | null;
    dayChange: number | null;
    dayChangePercent: number | null;
}

export interface StockPrediction {
    '1d': TimePeriodPrediction;
    '1w': TimePeriodPrediction;
    '1m': TimePeriodPrediction;
    '1y': TimePeriodPrediction;
    technicalSignals: TechnicalSignals;
}

export interface TimePeriodPrediction {
    trend: 'bullish' | 'bearish' | 'neutral';
    confidence: number;
    predictedPrice: number;
    percentChange: number;
}

export interface TechnicalSignals {
    rsi: {
        value: number;
        signal: 'overbought' | 'oversold' | 'neutral';
    };
    macd: {
        value: number;
        signal: 'buy' | 'sell' | 'neutral';
    };
    volatility: {
        value: number;
        trend: 'increasing' | 'decreasing' | 'stable';
    };
    volume: {
        trend: 'increasing' | 'decreasing' | 'stable';
        deviation: number;
    };
}

export interface APIResponse<T> {
    data: T;
    status: 'success' | 'error';
    error?: string;
}