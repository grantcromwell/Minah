#include "WebSocketBlockchainConnector.h"
#include <websocketpp/config/asio_client.hpp>
#include <boost/beast/websocket.hpp>
#include <boost/beast/http.hpp>
#include <boost/asio/ssl.hpp>
#include <chrono>
#include <iostream>

namespace minah {
namespace blockchain {

WebSocketBlockchainConnector::WebSocketBlockchainConnector(NetworkType network, const Config& config)
    : BlockchainNodeConnector(network, config.http_url)
    , config_(config)
    , is_running_(false)
    , reconnect_attempts_(0)
    , last_ping_time_(0)
    , next_subscription_id_(1)
    , latest_block_timestamp_(0) {

    setupWebSocketClient();
}

WebSocketBlockchainConnector::~WebSocketBlockchainConnector() {
    disconnect();
}

bool WebSocketBlockchainConnector::connect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);

    if (status_ == ConnectionStatus::CONNECTED) {
        return true;
    }

    status_ = ConnectionStatus::CONNECTING;

    try {
        // Parse WebSocket URL
        websocketpp::lib::error_code ec;
        ws_client_->get_io_service().dispatch([this]() {
            performConnection();
        });

        startIOService();

        // Wait for connection or timeout
        auto start = std::chrono::steady_clock::now();
        while (status_ == ConnectionStatus::CONNECTING) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start);

            if (elapsed.count() > config_.connection_timeout_ms) {
                status_ = ConnectionStatus::ERROR;
                std::cerr << "Connection timeout" << std::endl;
                return false;
            }
        }

        if (status_ == ConnectionStatus::CONNECTED) {
            setupSubscriptions();
            reconnect_attempts_ = 0;
            return true;
        }

    } catch (const std::exception& e) {
        std::cerr << "Connection error: " << e.what() << std::endl;
        status_ = ConnectionStatus::ERROR;
    }

    return false;
}

void WebSocketBlockchainConnector::disconnect() {
    std::lock_guard<std::mutex> lock(connection_mutex_);

    is_running_ = false;

    if (ws_client_ && connection_.lock()) {
        try {
            ws_client_->close(connection_, websocketpp::close::status::going_away, "");
        } catch (const std::exception& e) {
            std::cerr << "Error closing WebSocket: " << e.what() << std::endl;
        }
    }

    stopIOService();
    status_ = ConnectionStatus::DISCONNECTED;
}

std::future<uint64_t> WebSocketBlockchainConnector::getBlockNumber() {
    return asyncRPCRequest(createRPCRequest("eth_blockNumber", nlohmann::json::array()))
        .then([](std::future<std::string> future) {
            std::string response = future.get();
            auto json = nlohmann::json::parse(response);
            std::string blockNumberHex = json["result"];
            return std::stoull(blockNumberHex, nullptr, 16);
        });
}

std::future<Block> WebSocketBlockchainConnector::getBlock(uint64_t blockNumber) {
    nlohmann::json params = {
        "0x" + minah::utils::to_hex(blockNumber),
        true  // Include full transactions
    };

    return asyncRPCRequest(createRPCRequest("eth_getBlockByNumber", params))
        .then([this](std::future<std::string> future) {
            std::string response = future.get();
            auto json = nlohmann::json::parse(response);
            return parseBlock(json["result"]);
        });
}

void WebSocketBlockchainConnector::setupWebSocketClient() {
    ws_client_ = std::make_unique<websocket_client>();

    // Set up access channels
    ws_client_->set_access_channels(websocketpp::log::alevel::connect);
    ws_client_->set_access_channels(websocketpp::log::alevel::disconnect);
    ws_client_->set_access_channels(websocketpp::log::alevel::app);

    // Initialize ASIO
    ws_client_->init_asio();

    // Set up handlers
    ws_client_->set_open_handler([this](connection_hdl hdl) { onOpen(hdl); });
    ws_client_->set_fail_handler([this](connection_hdl hdl) { onFail(hdl); });
    ws_client_->set_close_handler([this](connection_hdl hdl) { onClose(hdl); });
    ws_client_->set_message_handler([this](connection_hdl hdl, message_ptr msg) {
        onMessage(hdl, msg);
    });
}

void WebSocketBlockchainConnector::startIOService() {
    if (io_thread_) {
        return; // Already running
    }

    work_ = std::make_unique<boost::asio::io_service::work>(ws_client_->get_io_service());
    is_running_ = true;

    io_thread_ = std::make_unique<std::thread>([this]() {
        while (is_running_) {
            try {
                ws_client_->run();
                break; // Exit when run() completes
            } catch (const std::exception& e) {
                std::cerr << "IO service error: " << e.what() << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(1));
            }
        }
    });
}

void WebSocketBlockchainConnector::stopIOService() {
    is_running_ = false;

    if (work_) {
        work_.reset();
    }

    if (io_thread_ && io_thread_->joinable()) {
        io_thread_->join();
        io_thread_.reset();
    }
}

void WebSocketBlockchainConnector::performConnection() {
    websocketpp::lib::error_code ec;
    websocket_client::connection_ptr con = ws_client_->get_connection(config_.ws_url, ec);

    if (ec) {
        std::cerr << "Connection initialization error: " << ec.message() << std::endl;
        status_ = ConnectionStatus::ERROR;
        return;
    }

    // Add custom headers
    for (const auto& header : config_.headers) {
        con->append_header(header.first, header.second);
    }

    // Configure SSL for wss:// URLs
    if (config_.ws_url.substr(0, 6) == "wss://") {
        con->get_socket().set_verify_mode(websocketpp::lib::asio::ssl::verify_peer);
        con->get_socket().set_verify_callback(websocketpp::lib::bind(&WebSocketBlockchainConnector::verify_certificate, this));
    }

    ws_client_->connect(con);
}

bool WebSocketBlockchainConnector::verify_certificate(bool preverified, boost::asio::ssl::verify_context& ctx) {
    if (!config_.verify_ssl_certificates) {
        return true;
    }

    // Add custom certificate verification logic here
    return preverified;
}

void WebSocketBlockchainConnector::onOpen(connection_hdl hdl) {
    std::cout << "WebSocket connection opened" << std::endl;
    connection_ = hdl;
    status_ = ConnectionStatus::CONNECTED;
}

void WebSocketBlockchainConnector::onFail(connection_hdl hdl) {
    std::cout << "WebSocket connection failed" << std::endl;
    status_ = ConnectionStatus::ERROR;
    handleReconnection();
}

void WebSocketBlockchainConnector::onClose(connection_hdl hdl) {
    std::cout << "WebSocket connection closed" << std::endl;
    status_ = ConnectionStatus::DISCONNECTED;

    if (is_running_) {
        handleReconnection();
    }
}

void WebSocketBlockchainConnector::onMessage(connection_hdl hdl, message_ptr msg) {
    try {
        auto message = nlohmann::json::parse(msg->get_payload());

        if (message.contains("method")) {
            // This is a subscription message
            handleSubscriptionMessage(message);
        } else if (message.contains("id")) {
            // This is an RPC response
            // Handle it through the promise mechanism
        }
    } catch (const std::exception& e) {
        std::cerr << "Error parsing WebSocket message: " << e.what() << std::endl;
    }
}

void WebSocketBlockchainConnector::handleReconnection() {
    if (reconnect_attempts_ >= config_.max_reconnect_attempts) {
        std::cerr << "Max reconnection attempts reached" << std::endl;
        is_running_ = false;
        return;
    }

    reconnect_attempts_++;

    std::this_thread::sleep_for(std::chrono::milliseconds(config_.reconnect_delay_ms));

    if (is_running_) {
        performConnection();
    }
}

std::string WebSocketBlockchainConnector::sendRPCRequest(const nlohmann::json& request) {
    // Implementation for synchronous HTTP RPC calls
    // This would use Boost.Beast HTTP client
    return "";
}

std::future<std::string> WebSocketBlockchainConnector::asyncRPCRequest(const nlohmann::json& request) {
    // Implementation for async RPC calls with promise/future
    auto promise = std::make_shared<std::promise<std::string>>();
    auto future = promise->get_future();

    // Send request and store promise
    // Implementation details would go here

    return future;
}

nlohmann::json WebSocketBlockchainConnector::createRPCRequest(const std::string& method, const nlohmann::json& params) {
    return {
        {"jsonrpc", "2.0"},
        {"method", method},
        {"params", params},
        {"id", next_subscription_id_++}
    };
}

void WebSocketBlockchainConnector::setupSubscriptions() {
    // Set up new block subscription
    subscribeToNewBlocks([this](const Block& block) {
        latest_block_timestamp_ = block.timestamp;
        for (auto& callback : newBlockCallbacks_) {
            callback(block);
        }
    });
}

bool WebSocketBlockchainConnector::subscribeToNewBlocks(NewBlockCallback callback) {
    newBlockCallbacks_.push_back(callback);

    auto request = createRPCRequest("eth_subscribe", nlohmann::json::array({"newHeads"}));
    sendSubscription("eth_subscribe", nlohmann::json::array({"newHeads"}), "newHeads");

    return true;
}

void WebSocketBlockchainConnector::sendSubscription(const std::string& method,
                                                  const nlohmann::json& params,
                                                  const std::string& id) {
    auto request = createRPCRequest(method, params);
    std::string message = request.dump();

    ws_client_->get_alog().write(websocketpp::log::alevel::app, "Sending subscription: " + message);

    if (auto con = connection_.lock()) {
        ws_client_->send(con, message, websocketpp::frame::opcode::text);
    }
}

void WebSocketBlockchainConnector::handleSubscriptionMessage(const nlohmann::json& message) {
    std::string subscription = message["params"]["subscription"];

    if (subscription.find("newHeads") != std::string::npos) {
        // Handle new block
        Block block = parseBlock(message["params"]["result"]);
        for (auto& callback : newBlockCallbacks_) {
            callback(block);
        }
    }
}

Block WebSocketBlockchainConnector::parseBlock(const nlohmann::json& blockJson) {
    Block block;
    block.number = std::stoull(blockJson["number"].get<std::string>(), nullptr, 16);
    block.hash = blockJson["hash"];
    block.parentHash = blockJson["parentHash"];
    block.miner = blockJson.value("miner", "");
    block.timestamp = std::stoull(blockJson["timestamp"].get<std::string>(), nullptr, 16);
    block.gasLimit = std::stoull(blockJson["gasLimit"].get<std::string>(), nullptr, 16);
    block.gasUsed = std::stoull(blockJson["gasUsed"].get<std::string>(), nullptr, 16);
    block.extraData = blockJson.value("extraData", "");

    // Parse transactions if present
    for (const auto& txJson : blockJson["transactions"]) {
        block.transactions.push_back(parseTransaction(txJson));
    }

    return block;
}

Transaction WebSocketBlockchainConnector::parseTransaction(const nlohmann::json& txJson) {
    Transaction tx;
    tx.hash = txJson["hash"];
    tx.from = txJson["from"];
    tx.to = txJson.value("to", "");
    tx.value = std::stoull(txJson["value"].get<std::string>(), nullptr, 16);
    tx.gasLimit = std::stoull(txJson["gas"].get<std::string>(), nullptr, 16);
    tx.gasPrice = std::stoull(txJson["gasPrice"].get<std::string>(), nullptr, 16);
    tx.blockNumber = std::stoull(txJson["blockNumber"].get<std::string>(), nullptr, 16);
    tx.transactionIndex = std::stoull(txJson["transactionIndex"].get<std::string>(), nullptr, 16);
    tx.input = txJson["input"];
    tx.status = std::stoull(txJson.value("status", "0x1"), nullptr, 16) == 1;

    return tx;
}

uint64_t WebSocketBlockchainConnector::getChainId() const {
    switch (network_) {
        case NetworkType::ETHEREUM: return 1;
        case NetworkType::POLYGON: return 137;
        case NetworkType::ARBITRUM: return 42161;
        case NetworkType::OPTIMISM: return 10;
        case NetworkType::BASE: return 8453;
        case NetworkType::HYPERLIQUID: return 998; // Hyperliquid testnet
        default: return 0;
    }
}

bool WebSocketBlockchainConnector::isHealthy() const {
    return status_ == ConnectionStatus::CONNECTED &&
           (std::time(nullptr) - latest_block_timestamp_) < 60; // Latest block within last minute
}

uint64_t WebSocketBlockchainConnector::getLatestBlockTimestamp() const {
    return latest_block_timestamp_;
}

double WebSocketBlockchainConnector::getAverageResponseTime() const {
    std::lock_guard<std::mutex> lock(performance_mutex_);

    if (response_times_.empty()) {
        return 0.0;
    }

    double total = 0.0;
    while (!response_times_.empty()) {
        auto& [start, end] = response_times_.front();
        total += std::chrono::duration<double, std::milli>(end - start).count();
        response_times_.pop();
    }

    return total / 100.0; // Average of last 100 responses
}

// Implementation of other methods would continue here...

// HyperliquidWebSocketConnector implementation
HyperliquidWebSocketConnector::HyperliquidWebSocketConnector(const std::string& ws_url)
    : WebSocketBlockchainConnector(NetworkType::HYPERLIQUID, {
        ws_url,
        "https://api.hyperliquid.xyz/info",
        10000,
        30000,
        10,
        5000,
        true,
        {{"Content-Type", "application/json"}}
    }) {
    // Hyperliquid-specific initialization
}

} // namespace blockchain
} // namespace minah