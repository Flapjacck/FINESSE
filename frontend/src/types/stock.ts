export interface StockNews {
    title: string;
    publisher: string;
    link: string;
    published: string;
    summary: string;
    source: string | null;
    type: string;
    sentiment: {
        label: string;
        score: number;
        nuanced_score: number;
        confidence: string;
    };
}

export interface StockAnalysisResponse {
    meta: {
        ticker: string;
        count: number;
        status: string;
        timestamp: string;
        request_id: string;
    };
    data: {
        news: StockNews[];
        analysis: {
            sentiment: {
                summary: {
                    positive_percentage: number;
                    negative_percentage: number;
                    neutral_percentage: number;
                    overall_sentiment: number;
                    average_confidence: number;
                    sentiment_counts: {
                        positive: number;
                        negative: number;
                        neutral: number;
                    };
                };
                trend: {
                    direction: 'positive' | 'negative';
                    strength: number;
                    confidence: number;
                };
            };
            technical: {
                indicators: {
                    combined_score: number;
                    confidence: number;
                    metrics: {
                        momentum: number;
                        price: number;
                        rsi: number;
                        volatility: number;
                        volume_trend: number;
                    };
                    moving_averages: {
                        sma_20: number;
                        sma_200: number;
                        sma_50: number;
                    };
                    prediction: string;
                    relative_strength: {
                        long_term: number;
                        short_term: number;
                    };
                    signals: {
                        long_term: string;
                        momentum: string;
                        rsi: string;
                        trend: string;
                        volume: string;
                    };
                    technical_score: number;
                };
                momentum: {
                    long_term: number;
                    short_term: number;
                };
                signals: {
                    long_term: string;
                    momentum: string;
                    rsi: string;
                    trend: string;
                    volume: string;
                };
            };
            fundamental: {
                metrics: {
                    pe_ratio?: number;
                    market_cap?: number;
                    beta?: number;
                    dividend_yield?: number;
                    revenue_growth?: number;
                    profit_margins?: number;
                };
                health_score: number;
            };
        };
        recommendation: {
            action: string;
            confidence: number;
            factors: {
                combined: number;
                sentiment: number;
                technical: number;
            };
            summary: string;
        };
    };
}

export interface StockError {
    message: string;
    code: string;
}