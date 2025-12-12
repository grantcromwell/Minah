#pragma once

#include "../connectors/WebSocketBlockchainConnector.h"
#include "../../OrderManagerService.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace minah {
namespace blockchain {
namespace adapters {

/**
 * @brief DEX adapter interface for Minah integration
 *
 * This interface defines the contract for integrating decentralized exchanges
 * with Minah's order management system.
 */
class DEXAdapter {
public:
    virtual ~DEXAdapter() = default;

    /**
     * @brief Execute a swap/trade on the DEX
     */
    virtual std::string executeSwap(const SwapParams& params) = 0;

    /**
     * @brief Estimate gas for a transaction
     */
    virtual uint64_t estimateGas(const Transaction& tx) = 0;

    /**
     * @brief Check if transaction is finalized
     */
    virtual bool isTransactionFinalized(const std::string& txHash) = 0;

    /**
     * @brief Get available liquidity pools
     */
    virtual std::vector<LiquidityPool> getLiquidityPools() = 0;
};

/**
 * @brief Swap parameters structure
 */
struct SwapParams {
    std::string from_token;          // Token to sell (e.g., "ETH", "USDC")
    std::string to_token;            // Token to buy
    double amount;                   // Amount to sell in from_token
    uint64_t min_amount_out;         // Minimum amount to receive (slippage protection)
    uint64_t max_gas_price;          // Maximum gas price willing to pay
    uint64_t deadline;               // Transaction deadline timestamp
    std::string user_address;        // User's wallet address
    bool use_flashbots;              // Whether to use flashbots for MEV protection
};

/**
 * @brief Transaction structure
 */
struct Transaction {
    std::string to;                  // Recipient address
    uint64_t value;                  // ETH value in wei
    std::string data;                // Transaction calldata
    uint64_t gas_limit;              // Gas limit
    uint64_t gas_price;              // Gas price in wei
    uint64_t nonce;                  // Transaction nonce
};

/**
 * @brief Liquidity pool information
 */
struct LiquidityPool {
    std::string token_a;             // First token in pair
    std::string token_b;             // Second token in pair
    double reserve_a;                // Reserve of token_a
    double reserve_b;                // Reserve of token_b
    double apr;                      // Annual percentage rate (for LPs)
    uint64_t fee_rate;               // Fee rate in basis points
    std::string pool_address;        // Pool contract address
    double tvl;                      // Total value locked in USD
};

/**
 * @brief Hyperliquid DEX adapter
 *
 * This adapter provides integration with Hyperliquid, a high-performance
 * decentralized exchange built on its own L1 blockchain. It supports both
 * spot and perpetual trading with sub-second settlement times.
 */
class HyperliquidAdapter : public DEXAdapter {
public:
    /**
     * @brief Configuration for Hyperliquid adapter
     */
    struct Config {
        std::string api_endpoint = "https://api.hyperliquid.xyz/info";
        std::string ws_endpoint = "wss://api.hyperliquid.xyz/ws";
        std::string exchange_address = "0x5CC3C495C3525B65e76E18d1541AB22E77E0a62e";
        std::string clearinghouse_address = "0x107e19A4585A4718441C55D8a792575A6F13B7c6";
        uint64_t chain_id = 998; // Hyperliquid testnet
        uint64_t default_slippage_bps = 50; // 0.5%
        uint64_t gas_price_gwei = 2;
        uint64_t gas_limit = 1000000;
        uint64_t transaction_timeout_ms = 30000;
    };

    /**
     * @brief Order types supported by Hyperliquid
     */
    enum class OrderType {
        MARKET,
        LIMIT,
        STOP_MARKET,
        STOP_LIMIT,
        TAKE_PROFIT_MARKET,
        TAKE_PROFIT_LIMIT,
        TRAILING_STOP
    };

    /**
     * @brief Order side
     */
    enum class OrderSide {
        BUY,
        SELL
    };

    /**
     * @brief Order structure
     */
    struct Order {
        std::string coin;             // Symbol (e.g., "ETH", "BTC")
        OrderSide side;               // Buy or sell
        OrderType type;               // Order type
        double size;                  // Order size in base currency
        double price;                 // Price for limit orders
        double trigger_price;         // Trigger price for stop orders
        uint64_t reduce_only;         // Whether to reduce position only
        uint64_t time_in_force;       // Time in force (GTC, IOC, FOK)
        uint64_t client_order_id;     // Client-defined order ID
    };

    /**
     * @brief Position information
     */
    struct Position {
        std::string coin;             // Symbol
        double size;                  // Position size (positive = long, negative = short)
        double entry_price;           // Average entry price
        double unrealized_pnl;        // Unrealized PNL
        double realized_pnl;          // Realized PNL
        double margin_used;           // Margin used
        double leverage;              // Current leverage
        uint64_t last_update_timestamp; // Last update time
    };

    /**
     * @brief Trade execution result
     */
    struct TradeResult {
        std::string tx_hash;          // Transaction hash
        std::string order_id;         // Order ID
        double executed_size;         // Actually executed size
        double executed_price;        // Average execution price
        double fee_paid;              // Fees paid
        uint64_t timestamp;           // Execution timestamp
        std::string status;           // Status: "filled", "partial", "failed"
        std::string error_message;    // Error message if failed
    };

public:
    explicit HyperliquidAdapter(const Config& config,
                               std::shared_ptr<HyperliquidWebSocketConnector> connector);
    ~HyperliquidAdapter() override;

    // DEXAdapter interface
    std::string executeSwap(const SwapParams& params) override;
    uint64_t estimateGas(const Transaction& tx) override;
    bool isTransactionFinalized(const std::string& txHash) override;
    std::vector<LiquidityPool> getLiquidityPools() override;

    // Hyperliquid-specific trading methods
    std::future<TradeResult> placeOrder(const Order& order);
    std::future<bool> cancelOrder(const std::string& order_id, const std::string& coin);
    std::future<bool> modifyOrder(const std::string& order_id, const Order& new_order);
    std::future<std::vector<Order>> getOpenOrders(const std::string& coin = "");
    std::future<std::vector<Position>> getPositions();
    std::future<std::vector<TradeResult>> getOrderHistory(const std::string& coin = "",
                                                         int limit = 100);

    // Market data methods
    std::future<nlohmann::json> getMeta();
    std::future<nlohmann::json> getAllMids();
    std::future<nlohmann::json> getSpotOrderBook(const std::string& coin, int depth = 100);
    std::future<nlohmann::json> getPerpOrderBook(const std::string& coin, int depth = 100);
    std::future<std::vector<nlohmann::json>> getRecentTrades(const std::string& coin, int limit = 100);
    std::future<nlohmann::json> getFundingHistory(const std::vector<std::string>& coins);
    std::future<nlohmann::json> getHistoricalOpenInterest(const std::string& coin);

    // Portfolio and balance methods
    std::future<nlohmann::json> getSpotUserState(const std::string& user_address);
    std::future<nlohmann::json> getClearinghouseState(const std::string& user_address);
    std::future<nlohmann::json> getFundingHistory(const std::string& user_address);

    // Advanced trading features
    std::future<TradeResult> placeBulkOrders(const std::vector<Order>& orders);
    std::future<TradeResult> placeOcoOrder(const Order& order, const Order& stop_loss);
    std::future<TradeResult> placeTpSlOrder(const Order& order, double take_profit, double stop_loss);

    // Subscription methods for real-time updates
    bool subscribeToTrades(const std::string& coin,
                          std::function<void(const nlohmann::json&)> callback);
    bool subscribeToOrders(std::function<void(const nlohmann::json&)> callback);
    bool subscribeToPositions(std::function<void(const nlohmann::json&)> callback);
    bool subscribeToPnl(std::function<void(const nlohmann::json&)> callback);

private:
    // Internal methods
    nlohmann::json createOrderRequest(const Order& order);
    nlohmann::json createCancelRequest(const std::string& order_id, const std::string& coin);
    nlohmann::json createBulkOrderRequest(const std::vector<Order>& orders);

    std::string signTransaction(const nlohmann::json& request, const std::string& private_key);
    std::string executeRequest(const nlohmann::json& request);
    std::future<std::string> executeAsyncRequest(const nlohmann::json& request);

    // Order management
    void handleOrderUpdate(const nlohmann::json& update);
    void handlePositionUpdate(const nlohmann::json& update);
    void handleTradeUpdate(const nlohmann::json& update);
    void handlePnlUpdate(const nlohmann::json& update);

    // Utility methods
    std::string formatPrice(double price, const std::string& coin);
    std::string formatSize(double size, const std::string& coin);
    double parsePrice(const std::string& price_str);
    double parseSize(const std::string& size_str);
    uint64_t getCurrentNonce(const std::string& address);
    double calculateSlippage(double price, uint64_t slippage_bps, OrderSide side);

    // Configuration and connections
    Config config_;
    std::shared_ptr<HyperliquidWebSocketConnector> ws_connector_;

    // State management
    std::unordered_map<std::string, Order> open_orders_;
    std::unordered_map<std::string, Position> positions_;
    mutable std::mutex state_mutex_;

    // Callbacks
    std::vector<std::function<void(const nlohmann::json&)>> trade_callbacks_;
    std::vector<std::function<void(const nlohmann::json&)>> order_callbacks_;
    std::vector<std::function<void(const nlohmann::json&)>> position_callbacks_;
    std::vector<std::function<void(const nlohmann::json&)>> pnl_callbacks_;

    // Performance tracking
    std::chrono::steady_clock::time_point last_order_time_;
    uint64_t total_orders_placed_;
    uint64_t total_orders_filled_;
    double total_volume_usd_;
    mutable std::mutex performance_mutex_;
};

/**
 * @brief Factory for creating DEX adapters
 */
class DEXAdapterFactory {
public:
    /**
     * @brief Create a Hyperliquid adapter
     */
    static std::unique_ptr<HyperliquidAdapter> createHyperliquidAdapter(
        const HyperliquidAdapter::Config& config,
        std::shared_ptr<HyperliquidWebSocketConnector> connector);

    /**
     * @brief Create adapter for specified network
     */
    static std::unique_ptr<DEXAdapter> createAdapter(
        BlockchainNodeConnector::NetworkType network,
        const nlohmann::json& config,
        std::shared_ptr<WebSocketBlockchainConnector> connector);
};

} // namespace adapters
} // namespace blockchain
} // namespace minah