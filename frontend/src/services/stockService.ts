import type { StockData, StockError, StockSearchResult, StockInterval, HistoricalData } from '../types/stock';

const API_BASE_URL = 'http://localhost:5000';

export class StockService {
    private static async fetchWithError(url: string, options?: RequestInit): Promise<any> {
        const response = await fetch(url, options);
        const data = await response.json();
        
        if (!response.ok) {
            throw {
                message: data.message || 'An error occurred',
                code: data.code || 'UNKNOWN_ERROR'
            } as StockError;
        }
        
        return data;
    }

    static async getStockData(symbol: string): Promise<StockData> {
        return this.fetchWithError(`${API_BASE_URL}/stock/${symbol}`);
    }

    static async searchStocks(query: string): Promise<StockSearchResult[]> {
        return this.fetchWithError(`${API_BASE_URL}/search?q=${encodeURIComponent(query)}`);
    }

    static async getHistoricalData(symbol: string, interval: StockInterval): Promise<HistoricalData[]> {
        return this.fetchWithError(`${API_BASE_URL}/stock/${symbol}/history/${interval}`);
    }

    static async getStockSentiment(symbol: string): Promise<StockData['sentiment']> {
        return this.fetchWithError(`${API_BASE_URL}/stock/${symbol}/sentiment`);
    }
}
