#pragma once

#include "../connectors/WebSocketBlockchainConnector.h"
#include "../../MarketDataService.h"
#include <zmq.hpp>
#include <queue>
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <unordered_map>

namespace minah {
namespace blockchain {
namespace pipeline {

/**
 * @brief WebSocket data pipeline for real-time blockchain market data
 *
 * This pipeline processes WebSocket data streams from blockchain nodes,
 * transforms the data into Minah's market data format, and publishes
 * it through ZeroMQ for consumption by trading strategies.
 */
class WebSocketDataPipeline {
public:
    /**
     * @brief Configuration for the data pipeline
     */
    struct Config {
        // ZeroMQ configuration
        std::string zmq_publisher_endpoint = "tcp://*:5556";
        std::string zmq_control_endpoint = "tcp://*:5557";
        int zmq_io_threads = 4;
        int zmq_sndhwm = 1000000; // Send high water mark

        // Data processing configuration
        size_t max_queue_size = 100000;
        size_t batch_size = 100;
        int batch_timeout_ms = 10;

        // Performance tuning
        size_t num_processing_threads = std::thread::hardware_concurrency();
        bool enable_compression = false;
        bool enable_deduplication = true;

        // Monitoring
        bool enable_metrics = true;
        int metrics_interval_ms = 1000;
    };

    /**
     * @brief Market data types from blockchain
     */
    enum class MarketDataType {
        ORDER_BOOK_UPDATE,
        TRADE,
        KLINE,
        LIQUIDITY_UPDATE,
        FUNDING_RATE,
        BLOCK_EVENT,
        TRANSACTION_EVENT
    };

    /**
     * @brief Market data message structure
     */
    struct MarketDataMessage {
        MarketDataType type;
        std::string symbol;
        uint64_t timestamp;
        std::string blockchain_tx_hash;
        nlohmann::json data;
        uint64_t sequence_number;
    };

    /**
     * @brief Pipeline statistics
     */
    struct PipelineStats {
        uint64_t messages_processed = 0;
        uint64_t messages_dropped = 0;
        uint64_t bytes_processed = 0;
        double avg_processing_time_us = 0.0;
        double avg_queue_depth = 0.0;
        uint64_t last_sequence_number = 0;
        std::chrono::steady_clock::time_point last_update_time;
    };

public:
    explicit WebSocketDataPipeline(const Config& config);
    ~WebSocketDataPipeline();

    // Lifecycle management
    bool start();
    void stop();
    bool isRunning() const { return is_running_; }

    // Connector management
    bool addBlockchainConnector(const std::string& id,
                               std::shared_ptr<WebSocketBlockchainConnector> connector,
                               const std::vector<std::string>& symbols);

    bool removeBlockchainConnector(const std::string& id);

    // Subscription management
    bool subscribeToOrderBook(const std::string& connector_id,
                            const std::string& symbol,
                            int depth = 100);

    bool subscribeToTrades(const std::string& connector_id,
                          const std::string& symbol);

    bool subscribeToKlines(const std::string& connector_id,
                          const std::string& symbol,
                          const std::string& interval);

    bool subscribeToFundingRates(const std::string& connector_id,
                                const std::vector<std::string>& symbols);

    // Pipeline control
    void pause();
    void resume();
    void flush();

    // Statistics and monitoring
    PipelineStats getStatistics() const;
    void resetStatistics();

    // Health monitoring
    bool isHealthy() const;
    std::vector<std::string> getConnectorStatus() const;

private:
    // Core processing methods
    void dataProcessingThread();
    void publishingThread();
    void metricsThread();

    // Data transformation methods
    MarketDataMessage transformOrderBookData(const nlohmann::json& raw_data,
                                            const std::string& symbol,
                                            const std::string& connector_id);

    MarketDataMessage transformTradeData(const nlohmann::json& raw_data,
                                        const std::string& symbol,
                                        const std::string& connector_id);

    MarketDataMessage transformKlineData(const nlohmann::json& raw_data,
                                        const std::string& symbol,
                                        const std::string& connector_id);

    MarketDataMessage transformFundingRateData(const nlohmann::json& raw_data,
                                              const std::vector<std::string>& symbols,
                                              const std::string& connector_id);

    // Utility methods
    std::string createTopic(const MarketDataType& type,
                           const std::string& symbol,
                           const std::string& connector_id);

    bool shouldPublishMessage(const MarketDataMessage& message);
    bool isDuplicateMessage(const MarketDataMessage& message);

    // ZeroMQ methods
    void setupZeroMQ();
    void cleanupZeroMQ();
    void publishMessage(const MarketDataMessage& message);

    // Configuration and state
    Config config_;
    std::atomic<bool> is_running_;
    std::atomic<bool> is_paused_;

    // Connectors and subscriptions
    std::mutex connectors_mutex_;
    std::unordered_map<std::string, std::shared_ptr<WebSocketBlockchainConnector>> connectors_;
    std::unordered_map<std::string, std::vector<std::string>> connector_symbols_;

    // Data queues
    std::queue<MarketDataMessage> data_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_condition_;

    // Processing threads
    std::vector<std::thread> processing_threads_;
    std::thread publishing_thread_;
    std::thread metrics_thread_;

    // ZeroMQ infrastructure
    std::unique_ptr<zmq::context_t> zmq_context_;
    std::unique_ptr<zmq::socket_t> publisher_socket_;
    std::unique_ptr<zmq::socket_t> control_socket_;

    // Statistics
    mutable std::mutex stats_mutex_;
    PipelineStats stats_;
    std::unordered_map<std::string, uint64_t> last_sequence_numbers_;

    // Deduplication
    std::mutex dedup_mutex_;
    std::unordered_set<std::string> recent_message_hashes_;
    std::chrono::steady_clock::time_point last_cleanup_time_;

    // Metrics
    std::queue<std::pair<std::chrono::steady_clock::time_point, double>> processing_times_;
    std::queue<std::chrono::steady_clock::time_point> queue_depth_snapshots_;
};

/**
 * @brief Specialized pipeline for Hyperliquid market data
 *
 * This pipeline handles Hyperliquid-specific data formats and provides
 * optimized processing for high-frequency trading operations.
 */
class HyperliquidDataPipeline : public WebSocketDataPipeline {
public:
    explicit HyperliquidDataPipeline(const Config& config);

    // Hyperliquid-specific subscriptions
    bool subscribeToAllTrades();
    bool subscribeToOrderBook(const std::string& coin, int depth = 100);
    bool subscribeToFundingHistory(const std::vector<std::string>& coins);

private:
    // Hyperliquid-specific data transformation
    MarketDataMessage transformHyperliquidTrade(const nlohmann::json& trade_data);
    MarketDataMessage transformHyperliquidOrderBook(const nlohmann::json& orderbook_data,
                                                    const std::string& coin);
    MarketDataMessage transformHyperliquidFunding(const nlohmann::json& funding_data);

    // Hyperliquid utilities
    std::string normalizeSymbol(const std::string& coin);
    double normalizePrice(const std::string& price_str);
    double normalizeSize(const std::string& size_str);
};

/**
 * @brief Factory for creating blockchain data pipelines
 */
class WebSocketDataPipelineFactory {
public:
    static std::unique_ptr<WebSocketDataPipeline> create(const WebSocketDataPipeline::Config& config,
                                                         NetworkType network_type);
};

} // namespace pipeline
} // namespace blockchain
} // namespace minah