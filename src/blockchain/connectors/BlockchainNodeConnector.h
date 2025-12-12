#pragma once

#include <string>
#include <vector>
#include <memory>
#include <functional>
#include <future>
#include <nlohmann/json.hpp>

namespace minah {
namespace blockchain {

/**
 * @brief Base interface for blockchain node connections
 *
 * This abstract class defines the interface for connecting to blockchain nodes,
 * handling transactions, and retrieving blockchain data. It supports both REST
 * and WebSocket connections for real-time data streaming.
 */
class BlockchainNodeConnector {
public:
    /**
     * @brief Connection status enum
     */
    enum class ConnectionStatus {
        DISCONNECTED,
        CONNECTING,
        CONNECTED,
        ERROR
    };

    /**
     * @brief Blockchain network types
     */
    enum class NetworkType {
        ETHEREUM,
        POLYGON,
        ARBITRUM,
        OPTIMISM,
        BASE,
        HYPERLIQUID
    };

    /**
     * @brief Transaction data structure
     */
    struct Transaction {
        std::string hash;
        std::string from;
        std::string to;
        uint64_t value;
        uint64_t gasLimit;
        uint64_t gasPrice;
        uint64_t gasUsed;
        uint64_t blockNumber;
        uint64_t transactionIndex;
        std::string input;
        bool status;
        nlohmann::json logs;
    };

    /**
     * @brief Block data structure
     */
    struct Block {
        uint64_t number;
        std::string hash;
        std::string parentHash;
        std::string miner;
        uint64_t timestamp;
        std::vector<Transaction> transactions;
        uint64_t gasLimit;
        uint64_t gasUsed;
        std::string extraData;
    };

    /**
     * @brief Event callback types
     */
    using NewBlockCallback = std::function<void(const Block&)>;
    using NewTransactionCallback = std::function<void(const Transaction&)>;
    using LogEventCallback = std::function<void(const nlohmann::json&)>;

public:
    explicit BlockchainNodeConnector(NetworkType network, const std::string& endpoint)
        : network_(network), endpoint_(endpoint), status_(ConnectionStatus::DISCONNECTED) {}

    virtual ~BlockchainNodeConnector() = default;

    // Connection management
    virtual bool connect() = 0;
    virtual void disconnect() = 0;
    virtual ConnectionStatus getConnectionStatus() const { return status_; }
    virtual bool isConnected() const { return status_ == ConnectionStatus::CONNECTED; }

    // Basic blockchain queries
    virtual std::future<uint64_t> getBlockNumber() = 0;
    virtual std::future<Block> getBlock(uint64_t blockNumber) = 0;
    virtual std::future<Block> getBlock(const std::string& blockHash) = 0;
    virtual std::future<std::vector<Transaction>> getBlockTransactions(uint64_t blockNumber) = 0;

    // Transaction operations
    virtual std::future<std::string> sendRawTransaction(const std::string& rawTx) = 0;
    virtual std::future<Transaction> getTransaction(const std::string& txHash) = 0;
    virtual std::future<Transaction> getTransactionReceipt(const std::string& txHash) = 0;
    virtual std::future<uint64_t> getTransactionCount(const std::string& address) = 0;

    // Contract interactions
    virtual std::future<std::string> call(const std::string& to, const std::string& data) = 0;
    virtual std::future<uint64_t> estimateGas(const nlohmann::json& txParams) = 0;
    virtual std::future<uint64_t> getGasPrice() = 0;
    virtual std::future<uint64_t> getBalance(const std::string& address) = 0;

    // Event subscriptions
    virtual bool subscribeToNewBlocks(NewBlockCallback callback) = 0;
    virtual bool subscribeToNewTransactions(NewTransactionCallback callback) = 0;
    virtual bool subscribeToLogs(const std::vector<std::string>& addresses,
                                const std::vector<std::string>& topics,
                                LogEventCallback callback) = 0;

    // Network-specific methods
    virtual NetworkType getNetworkType() const { return network_; }
    virtual uint64_t getChainId() const = 0;
    virtual std::vector<std::string> getSupportedEndpoints() const { return {endpoint_}; }

    // Health and performance
    virtual bool isHealthy() const = 0;
    virtual uint64_t getLatestBlockTimestamp() const = 0;
    virtual double getAverageResponseTime() const = 0;

protected:
    NetworkType network_;
    std::string endpoint_;
    ConnectionStatus status_;
    std::vector<NewBlockCallback> newBlockCallbacks_;
    std::vector<NewTransactionCallback> newTxCallbacks_;
    std::vector<LogEventCallback> logCallbacks_;
};

/**
 * @brief Factory for creating blockchain node connectors
 */
class BlockchainNodeConnectorFactory {
public:
    /**
     * @brief Create a connector for the specified network
     *
     * @param network The blockchain network type
     * @param endpoint The node endpoint URL
     * @return std::unique_ptr<BlockchainNodeConnector> The connector instance
     */
    static std::unique_ptr<BlockchainNodeConnector> create(
        BlockchainNodeConnector::NetworkType network,
        const std::string& endpoint);

    /**
     * @brief Create a connector with multiple endpoint failover
     *
     * @param network The blockchain network type
     * @param endpoints List of node endpoints for failover
     * @return std::unique_ptr<BlockchainNodeConnector> The connector instance
     */
    static std::unique_ptr<BlockchainNodeConnector> createWithFailover(
        BlockchainNodeConnector::NetworkType network,
        const std::vector<std::string>& endpoints);
};

} // namespace blockchain
} // namespace minah