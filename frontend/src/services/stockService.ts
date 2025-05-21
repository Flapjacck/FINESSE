import type { StockAnalysisResponse, StockError } from '../types/stock';

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

    static async getStockAnalysis(symbol: string, limit?: number): Promise<StockAnalysisResponse> {
        const url = new URL(`${API_BASE_URL}/stock/analysis`);
        url.searchParams.append('ticker', symbol);
        if (limit) {
            url.searchParams.append('limit', limit.toString());
        }
        return this.fetchWithError(url.toString());
    }
}
