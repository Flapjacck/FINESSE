import type { StockData, StockPrediction, APIResponse } from '../types/stock';

const API_BASE_URL = 'http://localhost:5000'; // Update this with your actual backend URL

/**
 * Validate a stock ticker symbol
 */
const validateTicker = (ticker: string): boolean => {
    if (!ticker || typeof ticker !== 'string') return false;
    // Matches the backend validation:
    // - Up to 8 characters for main symbol
    // - Optional .XX or -XX suffix for international markets
    // - No special characters except dots and hyphens
    return /^[A-Za-z0-9]{1,8}(?:[.-][A-Za-z0-9]{1,4})?$/.test(ticker);
};

/**
 * Get comprehensive stock data for a given ticker
 */
export const getStockData = async (ticker: string): Promise<StockData> => {
    if (!validateTicker(ticker)) {
        throw new Error('Invalid ticker format');
    }

    try {
        const response = await fetch(`${API_BASE_URL}/stock/data?ticker=${ticker}`);
        const result: APIResponse<StockData> = await response.json();
        
        if (result.status === 'error' || !result.data) {
            throw new Error(result.error || 'Failed to fetch stock data');
        }
        
        return result.data;
    } catch (error) {
        if (error instanceof Error) {
            throw error;
        }
        throw new Error('An unexpected error occurred while fetching stock data');
    }
};

/**
 * Get stock predictions for a given ticker
 */
export const getStockPrediction = async (ticker: string): Promise<StockPrediction> => {
    if (!validateTicker(ticker)) {
        throw new Error('Invalid ticker format');
    }

    try {
        const response = await fetch(`${API_BASE_URL}/stock/prediction?ticker=${ticker}`);
        const result: APIResponse<StockPrediction> = await response.json();
        
        if (result.status === 'error' || !result.data) {
            throw new Error(result.error || 'Failed to fetch stock predictions');
        }
        
        return result.data;
    } catch (error) {
        if (error instanceof Error) {
            throw error;
        }
        throw new Error('An unexpected error occurred while fetching stock predictions');
    }
};

/**
 * Check if the backend API is healthy
 */
export const checkAPIHealth = async (): Promise<boolean> => {
    try {
        const response = await fetch(`${API_BASE_URL}/health`);
        return response.ok;
    } catch {
        return false;
    }
};