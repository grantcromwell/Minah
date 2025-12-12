#pragma once

#include "BlockchainNodeConnector.h"
#include <websocketpp/config/asio_client.hpp>
#include <websocketpp/client.hpp>
#include <boost/asio.hpp>
#include <boost/asio/ssl.hpp>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>

namespace minah {
namespace blockchain {

/**
 * @brief WebSocket-based blockchain node connector
 *
 * This implementation uses WebSocket connections for real-time blockchain data streaming
 * with automatic reconnection, subscription management, and failover support.
 */
class WebSocketBlockchainConnector : public BlockchainNodeConnector {
public:
    using websocket_client = websocketpp::client<websocketpp::config::asio_tls_client>;
    using message_ptr = websocket_client::message_ptr;
    using connection_hdl = websocketpp::connection_hdl;

    /**
     * @brief Configuration for WebSocket connection
     */
    struct Config {
        std::string ws_url;
        std::string http_url;
        uint64_t connection_timeout_ms = 10000;
        uint64_t ping_interval_ms = 30000;
        uint64_t max_reconnect_attempts = 10;
        uint64_t reconnect_delay_ms = 5000;
        bool verify_ssl_certificates = true;
        std::map<std::string, std::string> headers;
    };

    explicit WebSocketBlockchainConnector(NetworkType network, const Config& config);
    ~WebSocketBlockchainConnector() override;

    // Connection management
    bool connect() override;
    void disconnect() override;

    // Basic blockchain queries
    std::future<uint64_t> getBlockNumber() override;
    std::future<Block> getBlock(uint64_t blockNumber) override;
    std::future<Block> getBlock(const std::string& blockHash) override;
    std::future<std::vector<Transaction>> getBlockTransactions(uint64_t blockNumber) override;

    // Transaction operations
    std::future<std::string> sendRawTransaction(const std::string& rawTx) override;
    std::future<Transaction> getTransaction(const std::string& txHash) override;
    std::future<Transaction> getTransactionReceipt(const std::string& txHash) override;
    std::future<uint64_t> getTransactionCount(const std::string& address) override;

    // Contract interactions
    std::future<std::string> call(const std::string& to, const std::string& data) override;
    std::future<uint64_t> estimateGas(const nlohmann::json& txParams) override;
    std::future<uint64_t> getGasPrice() override;
    std::future<uint64_t> getBalance(const std::string& address) override;

    // Event subscriptions
    bool subscribeToNewBlocks(NewBlockCallback callback) override;
    bool subscribeToNewTransactions(NewTransactionCallback callback) override;
    bool subscribeToLogs(const std::vector<std::string>& addresses,
                        const std::vector<std::string>& topics,
                        LogEventCallback callback) override;

    // Network-specific methods
    uint64_t getChainId() const override;

    // Health and performance
    bool isHealthy() const override;
    uint64_t getLatestBlockTimestamp() const override;
    double getAverageResponseTime() const override;

private:
    // Internal WebSocket handling
    void setupWebSocketClient();
    void startIOService();
    void stopIOService();
    void performConnection();
    void handleReconnection();

    // WebSocket event handlers
    void onOpen(connection_hdl hdl);
    void onFail(connection_hdl hdl);
    void onClose(connection_hdl hdl);
    void onMessage(connection_hdl hdl, message_ptr msg);

    // HTTP RPC calls
    std::string sendRPCRequest(const nlohmann::json& request);
    std::future<std::string> asyncRPCRequest(const nlohmann::json& request);

    // Subscription management
    void setupSubscriptions();
    void sendSubscription(const std::string& method, const nlohmann::json& params, const std::string& id);
    void handleSubscriptionMessage(const nlohmann::json& message);

    // Utility methods
    nlohmann::json createRPCRequest(const std::string& method, const nlohmann::json& params);
    Block parseBlock(const nlohmann::json& blockJson);
    Transaction parseTransaction(const nlohmann::json& txJson);

    // Member variables
    Config config_;
    std::unique_ptr<websocket_client> ws_client_;
    connection_hdl connection_;
    std::unique_ptr<boost::asio::io_service::work> work_;
    std::unique_ptr<std::thread> io_thread_;

    // State management
    std::atomic<bool> is_running_;
    std::atomic<uint64_t> reconnect_attempts_;
    std::atomic<uint64_t> last_ping_time_;
    mutable std::mutex connection_mutex_;

    // Subscription tracking
    std::map<std::string, uint64_t> subscription_ids_;
    std::atomic<uint64_t> next_subscription_id_;

    // Performance tracking
    mutable std::mutex performance_mutex_;
    std::queue<std::pair<std::chrono::steady_clock::time_point, std::chrono::steady_clock::time_point>> response_times_;
    std::atomic<uint64_t> latest_block_timestamp_;

    // HTTP client for RPC calls
    std::unique_ptr<boost::asio::ssl::context> ssl_context_;
};

/**
 * @brief Hyperliquid-specific WebSocket connector
 *
 * Specialized connector for Hyperliquid with custom event formats and
 * optimized for high-frequency trading operations.
 */
class HyperliquidWebSocketConnector : public WebSocketBlockchainConnector {
public:
    explicit HyperliquidWebSocketConnector(const std::string& ws_url);

    // Hyperliquid-specific methods
    std::future<nlohmann::json> getMetadatas();
    std::future<nlohmann::json> getSpotOrderBook(const std::string& coin, int depth = 100);
    std::future<nlohmann::json> getPerpOrderBook(const std::string& coin, int depth = 100);
    std::future<std::vector<nlohmann::json>> getRecentTrades(const std::string& coin, int limit = 100);
    std::future<nlohmann::json> getFundingRates(const std::vector<std::string>& coins);

    // Hyperliquid subscriptions
    bool subscribeToAllTrades(std::function<void(const nlohmann::json&)> callback);
    bool subscribeToOrderBook(const std::string& coin, std::function<void(const nlohmann::json&)> callback);
    bool subscribeToFundingRates(std::function<void(const nlohmann::json&)> callback);
    bool subscribeToTradeUpdates(std::function<void(const nlohmann::json&)> callback);

private:
    // Hyperliquid-specific message handling
    void handleHyperliquidMessage(const nlohmann::json& message);
    nlohmann::json createHyperliquidRequest(const std::string& method, const nlohmann::json& params);

    // Hyperliquid-specific callbacks
    std::vector<std::function<void(const nlohmann::json&)>> trade_callbacks_;
    std::map<std::string, std::function<void(const nlohmann::json&)>> orderbook_callbacks_;
    std::function<void(const nlohmann::json&)> funding_callbacks_;
    std::function<void(const nlohmann::json&)> trade_update_callbacks_;
};

} // namespace blockchain
} // namespace minah