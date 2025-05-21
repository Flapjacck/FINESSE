import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import type { StockAnalysisResponse } from "../../types/stock";
import { StockService } from "../../services/stockService";
import SearchBar from "../SearchBar";

const StockPage = () => {
  const { symbol } = useParams<{ symbol: string }>();
  const [data, setData] = useState<StockAnalysisResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchData = async () => {
      if (!symbol) return;

      try {
        setIsLoading(true);
        setError(null);
        const response = await StockService.getStockAnalysis(symbol);
        setData(response);
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to fetch stock data");
      } finally {
        setIsLoading(false);
      }
    };

    fetchData();
  }, [symbol]);

  if (!symbol) {
    return <div>No symbol provided</div>;
  }

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-green-500"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen p-4">
        <SearchBar darkMode={false} className="max-w-2xl mx-auto mb-8" />
        <div className="text-red-500 text-center">{error}</div>
      </div>
    );
  }

  if (!data) {
    return null;
  }

  const { meta, data: stockData } = data;

  const sentimentColor =
    stockData.analysis.sentiment.summary.overall_sentiment >= 0
      ? "text-green-500"
      : "text-red-500";

  const recommendationColor = stockData.recommendation.action.includes("BUY")
    ? "text-green-500"
    : stockData.recommendation.action.includes("SELL")
    ? "text-red-500"
    : "text-yellow-500";

  return (
    <div className="min-h-screen p-4 bg-gray-50">
      <SearchBar darkMode={false} className="max-w-2xl mx-auto mb-8" />

      <div className="max-w-6xl mx-auto space-y-6">
        {/* Header */}
        <div className="bg-white rounded-xl p-6 shadow-sm">
          <div className="flex justify-between items-center">
            <h1 className="text-3xl font-bold">${meta.ticker}</h1>
            <span className="text-sm text-gray-500">
              Last updated: {new Date(meta.timestamp).toLocaleString()}
            </span>
          </div>
        </div>

        {/* Recommendation */}
        <div className="bg-white rounded-xl p-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Recommendation</h2>
          <div className="space-y-2">
            <p className={`text-2xl font-bold ${recommendationColor}`}>
              {stockData.recommendation.action}
            </p>
            <p className="text-gray-600">{stockData.recommendation.summary}</p>
            <p className="text-sm text-gray-500">
              Confidence:{" "}
              {(stockData.recommendation.confidence * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        {/* Analysis Summary */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Sentiment Analysis */}
          <div className="bg-white rounded-xl p-6 shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Sentiment Analysis</h2>
            <div className="space-y-4">
              <div>
                <p className="text-gray-600">Overall Sentiment</p>
                <p className={`text-2xl font-bold ${sentimentColor}`}>
                  {(
                    stockData.analysis.sentiment.summary.overall_sentiment * 100
                  ).toFixed(1)}
                  %
                </p>
              </div>
              <div className="grid grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-600">Positive</p>
                  <p className="text-lg font-semibold text-green-500">
                    {stockData.analysis.sentiment.summary.positive_percentage}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Negative</p>
                  <p className="text-lg font-semibold text-red-500">
                    {stockData.analysis.sentiment.summary.negative_percentage}%
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-600">Neutral</p>
                  <p className="text-lg font-semibold text-yellow-500">
                    {stockData.analysis.sentiment.summary.neutral_percentage}%
                  </p>
                </div>
              </div>
            </div>
          </div>

          {/* Technical Analysis */}
          <div className="bg-white rounded-xl p-6 shadow-sm">
            <h2 className="text-xl font-semibold mb-4">Technical Analysis</h2>
            <div className="space-y-4">
              {/* Signals */}
              <div className="grid grid-cols-2 gap-4">
                {Object.entries(stockData.analysis.technical.signals).map(
                  ([key, value]) => (
                    <div key={key}>
                      <p className="text-sm text-gray-600 capitalize">
                        {key.replace("_", " ")}
                      </p>
                      <p
                        className={`text-lg font-semibold ${
                          value === "bullish"
                            ? "text-green-500"
                            : value === "bearish"
                            ? "text-red-500"
                            : "text-yellow-500"
                        }`}
                      >
                        {value.charAt(0).toUpperCase() + value.slice(1)}
                      </p>
                    </div>
                  )
                )}
              </div>
            </div>
          </div>
        </div>

        {/* News Feed */}
        <div className="bg-white rounded-xl p-6 shadow-sm">
          <h2 className="text-xl font-semibold mb-4">Latest News</h2>
          <div className="space-y-4">
            {stockData.news.map((item, index) => (
              <div
                key={index}
                className="border-b last:border-0 pb-4 last:pb-0"
              >
                <a
                  href={item.link}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="block hover:bg-gray-50 rounded-lg transition-colors duration-200 p-2"
                >
                  <div className="flex justify-between items-start">
                    <h3 className="text-lg font-medium text-gray-900">
                      {item.title}
                    </h3>
                    <span
                      className={`px-2 py-1 rounded text-sm ${
                        item.sentiment.label === "positive"
                          ? "bg-green-100 text-green-800"
                          : item.sentiment.label === "negative"
                          ? "bg-red-100 text-red-800"
                          : "bg-yellow-100 text-yellow-800"
                      }`}
                    >
                      {item.sentiment.label.charAt(0).toUpperCase() +
                        item.sentiment.label.slice(1)}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{item.summary}</p>
                  <div className="mt-2 flex items-center text-sm text-gray-500">
                    <span>{item.publisher}</span>
                    <span className="mx-2">•</span>
                    <span>{new Date(item.published).toLocaleDateString()}</span>
                  </div>
                </a>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default StockPage;
