#include "WebSocketDataPipeline.h"
#include <chrono>
#include <algorithm>
#include <sstream>
#include <openssl/sha.h>

namespace minah {
namespace blockchain {
namespace pipeline {

WebSocketDataPipeline::WebSocketDataPipeline(const Config& config)
    : config_(config)
    , is_running_(false)
    , is_paused_(false)
    , last_cleanup_time_(std::chrono::steady_clock::now()) {

    setupZeroMQ();
}

WebSocketDataPipeline::~WebSocketDataPipeline() {
    stop();
    cleanupZeroMQ();
}

bool WebSocketDataPipeline::start() {
    if (is_running_) {
        return true;
    }

    is_running_ = true;
    is_paused_ = false;

    // Start processing threads
    for (size_t i = 0; i < config_.num_processing_threads; ++i) {
        processing_threads_.emplace_back(&WebSocketDataPipeline::dataProcessingThread, this);
    }

    // Start publishing thread
    publishing_thread_ = std::thread(&WebSocketDataPipeline::publishingThread, this);

    // Start metrics thread if enabled
    if (config_.enable_metrics) {
        metrics_thread_ = std::thread(&WebSocketDataPipeline::metricsThread, this);
    }

    // Update statistics
    {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        stats_.last_update_time = std::chrono::steady_clock::now();
    }

    return true;
}

void WebSocketDataPipeline::stop() {
    is_running_ = false;
    is_paused_ = false;

    // Wake up all threads
    queue_condition_.notify_all();

    // Wait for threads to finish
    for (auto& thread : processing_threads_) {
        if (thread.joinable()) {
            thread.join();
        }
    }
    processing_threads_.clear();

    if (publishing_thread_.joinable()) {
        publishing_thread_.join();
    }

    if (metrics_thread_.joinable()) {
        metrics_thread_.join();
    }
}

bool WebSocketDataPipeline::addBlockchainConnector(
    const std::string& id,
    std::shared_ptr<WebSocketBlockchainConnector> connector,
    const std::vector<std::string>& symbols) {

    std::lock_guard<std::mutex> lock(connectors_mutex_);

    if (connectors_.find(id) != connectors_.end()) {
        return false; // Already exists
    }

    // Store connector
    connectors_[id] = connector;
    connector_symbols_[id] = symbols;

    // Set up subscriptions based on connector type
    if (auto hyperliquid_connector = std::dynamic_pointer_cast<HyperliquidWebSocketConnector>(connector)) {
        // Hyperliquid-specific subscriptions
        for (const auto& symbol : symbols) {
            hyperliquid_connector->subscribeToOrderBook(symbol, [this, id, symbol](const nlohmann::json& data) {
                auto message = transformOrderBookData(data, symbol, id);
                std::lock_guard<std::mutex> queue_lock(queue_mutex_);
                if (data_queue_.size() < config_.max_queue_size) {
                    data_queue_.push(message);
                    queue_condition_.notify_one();
                }
            });

            hyperliquid_connector->subscribeToAllTrades([this, id, symbol](const nlohmann::json& data) {
                auto message = transformTradeData(data, symbol, id);
                std::lock_guard<std::mutex> queue_lock(queue_mutex_);
                if (data_queue_.size() < config_.max_queue_size) {
                    data_queue_.push(message);
                    queue_condition_.notify_one();
                }
            });
        }

        hyperliquid_connector->subscribeToFundingRates([this, id](const nlohmann::json& data) {
            auto message = transformFundingRateData(data, connector_symbols_[id], id);
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            if (data_queue_.size() < config_.max_queue_size) {
                data_queue_.push(message);
                queue_condition_.notify_one();
            }
        });
    } else {
        // Generic blockchain subscriptions
        connector->subscribeToNewBlocks([this, id](const BlockchainNodeConnector::Block& block) {
            nlohmann::json block_data;
            block_data["number"] = block.number;
            block_data["hash"] = block.hash;
            block_data["timestamp"] = block.timestamp;

            MarketDataMessage message;
            message.type = MarketDataType::BLOCK_EVENT;
            message.symbol = "BLOCK";
            message.timestamp = block.timestamp;
            message.blockchain_tx_hash = block.hash;
            message.data = block_data;
            message.sequence_number = stats_.last_sequence_number++;

            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            if (data_queue_.size() < config_.max_queue_size) {
                data_queue_.push(message);
                queue_condition_.notify_one();
            }
        });
    }

    return true;
}

void WebSocketDataPipeline::dataProcessingThread() {
    std::vector<MarketDataMessage> batch;
    batch.reserve(config_.batch_size);

    while (is_running_) {
        // Wait for data or timeout
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_condition_.wait_for(lock, std::chrono::milliseconds(config_.batch_timeout_ms),
                                 [this]() { return !data_queue_.empty() || !is_running_; });

        // Collect batch of messages
        while (!data_queue_.empty() && batch.size() < config_.batch_size && !is_paused_) {
            batch.push_back(std::move(data_queue_.front()));
            data_queue_.pop();
        }
        lock.unlock();

        // Process batch
        if (!batch.empty()) {
            for (auto& message : batch) {
                if (shouldPublishMessage(message)) {
                    publishMessage(message);
                }
            }
            batch.clear();
        }
    }
}

void WebSocketDataPipeline::publishingThread() {
    while (is_running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));

        // Publishing is handled in dataProcessingThread
        // This thread can be used for optimization or additional publishing logic
    }
}

void WebSocketDataPipeline::metricsThread() {
    while (is_running_) {
        std::this_thread::sleep_for(std::chrono::milliseconds(config_.metrics_interval_ms));

        // Update queue depth snapshots
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            queue_depth_snapshots_.push(std::chrono::steady_clock::now());

            // Keep only last 1000 snapshots
            while (queue_depth_snapshots_.size() > 1000) {
                queue_depth_snapshots_.pop();
            }
        }

        // Cleanup old message hashes for deduplication
        if (config_.enable_deduplication) {
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::minutes>(now - last_cleanup_time_);

            if (elapsed.count() >= 5) { // Cleanup every 5 minutes
                std::lock_guard<std::mutex> dedup_lock(dedup_mutex_);
                // Keep only hashes from last 5 minutes
                last_cleanup_time_ = now;
            }
        }
    }
}

MarketDataMessage WebSocketDataPipeline::transformOrderBookData(
    const nlohmann::json& raw_data,
    const std::string& symbol,
    const std::string& connector_id) {

    MarketDataMessage message;
    message.type = MarketDataType::ORDER_BOOK_UPDATE;
    message.symbol = symbol;
    message.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    message.data = raw_data;
    message.sequence_number = stats_.last_sequence_number++;

    return message;
}

MarketDataMessage WebSocketDataPipeline::transformTradeData(
    const nlohmann::json& raw_data,
    const std::string& symbol,
    const std::string& connector_id) {

    MarketDataMessage message;
    message.type = MarketDataType::TRADE;
    message.symbol = symbol;
    message.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    message.data = raw_data;
    message.sequence_number = stats_.last_sequence_number++;

    return message;
}

MarketDataMessage WebSocketDataPipeline::transformKlineData(
    const nlohmann::json& raw_data,
    const std::string& symbol,
    const std::string& connector_id) {

    MarketDataMessage message;
    message.type = MarketDataType::KLINE;
    message.symbol = symbol;
    message.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    message.data = raw_data;
    message.sequence_number = stats_.last_sequence_number++;

    return message;
}

MarketDataMessage WebSocketDataPipeline::transformFundingRateData(
    const nlohmann::json& raw_data,
    const std::vector<std::string>& symbols,
    const std::string& connector_id) {

    MarketDataMessage message;
    message.type = MarketDataType::FUNDING_RATE;
    message.symbol = "FUNDING_RATES";
    message.timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    message.data = raw_data;
    message.data["symbols"] = symbols;
    message.sequence_number = stats_.last_sequence_number++;

    return message;
}

std::string WebSocketDataPipeline::createTopic(
    const MarketDataType& type,
    const std::string& symbol,
    const std::string& connector_id) {

    std::ostringstream oss;
    oss << connector_id << ".";

    switch (type) {
        case MarketDataType::ORDER_BOOK_UPDATE:
            oss << "orderbook." << symbol;
            break;
        case MarketDataType::TRADE:
            oss << "trade." << symbol;
            break;
        case MarketDataType::KLINE:
            oss << "kline." << symbol;
            break;
        case MarketDataType::LIQUIDITY_UPDATE:
            oss << "liquidity." << symbol;
            break;
        case MarketDataType::FUNDING_RATE:
            oss << "funding";
            break;
        case MarketDataType::BLOCK_EVENT:
            oss << "block";
            break;
        case MarketDataType::TRANSACTION_EVENT:
            oss << "transaction";
            break;
    }

    return oss.str();
}

bool WebSocketDataPipeline::shouldPublishMessage(const MarketDataMessage& message) {
    // Check for duplicates if enabled
    if (config_.enable_deduplication && isDuplicateMessage(message)) {
        return false;
    }

    return true;
}

bool WebSocketDataPipeline::isDuplicateMessage(const MarketDataMessage& message) {
    std::string message_hash = std::to_string(static_cast<uint64_t>(message.type)) +
                              message.symbol +
                              std::to_string(message.timestamp) +
                              message.data.dump();

    // Create SHA256 hash
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256;
    SHA256_Init(&sha256);
    SHA256_Update(&sha256, message_hash.c_str(), message_hash.length());
    SHA256_Final(hash, &sha256);

    std::string hash_str(reinterpret_cast<char*>(hash), SHA256_DIGEST_LENGTH);

    std::lock_guard<std::mutex> lock(dedup_mutex_);
    if (recent_message_hashes_.find(hash_str) != recent_message_hashes_.end()) {
        return true;
    }

    recent_message_hashes_.insert(hash_str);
    return false;
}

void WebSocketDataPipeline::setupZeroMQ() {
    zmq_context_ = std::make_unique<zmq::context_t>(config_.zmq_io_threads);

    // Publisher socket
    publisher_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, ZMQ_PUB);
    publisher_socket_->set(zmq::sockopt::sndhwm, config_.zmq_sndhwm);
    publisher_socket_->bind(config_.zmq_publisher_endpoint);

    // Control socket
    control_socket_ = std::make_unique<zmq::socket_t>(*zmq_context_, ZMQ_PAIR);
    control_socket_->bind(config_.zmq_control_endpoint);
}

void WebSocketDataPipeline::cleanupZeroMQ() {
    if (publisher_socket_) {
        publisher_socket_->close();
    }

    if (control_socket_) {
        control_socket_->close();
    }

    if (zmq_context_) {
        zmq_context_->close();
    }
}

void WebSocketDataPipeline::publishMessage(const MarketDataMessage& message) {
    try {
        // Serialize message
        nlohmann::json serialized_message = {
            {"type", static_cast<int>(message.type)},
            {"symbol", message.symbol},
            {"timestamp", message.timestamp},
            {"tx_hash", message.blockchain_tx_hash},
            {"sequence", message.sequence_number},
            {"data", message.data}
        };

        std::string topic = createTopic(message.type, message.symbol, "");
        std::string serialized = serialized_message.dump();

        // Publish to ZeroMQ
        zmq::message_t topic_msg(topic);
        zmq::message_t data_msg(serialized);

        publisher_socket_->send(topic_msg, zmq::send_flags::sndmore);
        publisher_socket_->send(data_msg, zmq::send_flags::dontwait);

        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.messages_processed++;
            stats_.bytes_processed += topic.size() + serialized.size();
            stats_.last_sequence_number = message.sequence_number;
            stats_.last_update_time = std::chrono::steady_clock::now();
        }

    } catch (const zmq::error_t& e) {
        if (e.num() != EAGAIN) {
            std::cerr << "ZeroMQ publish error: " << e.what() << std::endl;

            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.messages_dropped++;
        }
    }
}

PipelineStats WebSocketDataPipeline::getStatistics() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);

    PipelineStats stats = stats_;

    // Calculate average queue depth
    if (!queue_depth_snapshots_.empty()) {
        double total_depth = 0.0;
        {
            std::lock_guard<std::mutex> queue_lock(queue_mutex_);
            total_depth = static_cast<double>(data_queue_.size());
        }
        stats.avg_queue_depth = total_depth;
    }

    return stats;
}

bool WebSocketDataPipeline::isHealthy() const {
    if (!is_running_) {
        return false;
    }

    // Check connectors
    {
        std::lock_guard<std::mutex> lock(connectors_mutex_);
        for (const auto& [id, connector] : connectors_) {
            if (!connector->isConnected()) {
                return false;
            }
        }
    }

    // Check queue size
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (data_queue_.size() > config_.max_queue_size * 0.9) {
            return false; // Queue almost full
        }
    }

    return true;
}

// HyperliquidDataPipeline implementation
HyperliquidDataPipeline::HyperliquidDataPipeline(const Config& config)
    : WebSocketDataPipeline(config) {
    // Hyperliquid-specific initialization
}

std::unique_ptr<WebSocketDataPipeline> WebSocketDataPipelineFactory::create(
    const WebSocketDataPipeline::Config& config,
    NetworkType network_type) {

    if (network_type == NetworkType::HYPERLIQUID) {
        return std::make_unique<HyperliquidDataPipeline>(config);
    }

    return std::make_unique<WebSocketDataPipeline>(config);
}

} // namespace pipeline
} // namespace blockchain
} // namespace minah