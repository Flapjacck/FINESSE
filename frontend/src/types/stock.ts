export interface StockData {
    symbol: string;
    price: number;
    change: number;
    changePercent: number;
    marketCap: number;
    volume: number;
    high: number;
    low: number;
    open: number;
    previousClose: number;
    timestamp: string;
    fundamentals: {
        pe_ratio?: number;
        profit_margins?: number;
        revenue_growth?: number;
        beta?: number;
        dividend_yield?: number;
    };
    healthScore: number;
    sentiment: {
        score: number;
        label: 'positive' | 'negative' | 'neutral';
    };
}

export interface StockError {
    message: string;
    code: string;
}

export interface StockSearchResult {
    symbol: string;
    name: string;
    exchange: string;
}

export type StockInterval = '1d' | '5d' | '1mo' | '3mo' | '6mo' | '1y' | '2y' | '5y' | 'max';

export interface HistoricalData {
    date: string;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}