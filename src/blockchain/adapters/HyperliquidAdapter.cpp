#include "HyperliquidAdapter.h"
#include <boost/asio.hpp>
#include <boost/beast.hpp>
#include <openssl/evp.h>
#include <openssl/hmac.h>
#include <chrono>
#include <iostream>

namespace minah {
namespace blockchain {
namespace adapters {

using tcp = boost::asio::ip::tcp;
namespace http = boost::beast::http;
namespace websocket = boost::beast::websocket;

HyperliquidAdapter::HyperliquidAdapter(
    const Config& config,
    std::shared_ptr<HyperliquidWebSocketConnector> connector)
    : config_(config)
    , ws_connector_(connector)
    , total_orders_placed_(0)
    , total_orders_filled_(0)
    , total_volume_usd_(0.0)
    , last_order_time_(std::chrono::steady_clock::now()) {

    // Set up subscriptions for real-time updates
    if (ws_connector_) {
        ws_connector_->subscribeToTradeUpdates([this](const nlohmann::json& data) {
            handleTradeUpdate(data);
        });
    }
}

HyperliquidAdapter::~HyperliquidAdapter() = default;

std::string HyperliquidAdapter::executeSwap(const SwapParams& params) {
    // Create Hyperliquid swap order
    Order order;
    order.coin = params.from_token;
    order.side = (params.amount > 0) ? OrderSide::SELL : OrderSide::BUY;
    order.type = OrderType::MARKET;
    order.size = std::abs(params.amount);
    order.reduce_only = 0;
    order.time_in_force = 1001; // IOC (Immediate Or Cancel)
    order.client_order_id = getCurrentTimestamp();

    // Execute the order
    auto result_future = placeOrder(order);
    result_future.wait(); // Wait for execution

    auto result = result_future.get();
    return result.tx_hash;
}

uint64_t HyperliquidAdapter::estimateGas(const Transaction& tx) {
    // Hyperliquid has fixed gas structure since it's L1
    uint64_t base_gas = config_.gas_limit;
    uint64_t variable_gas = tx.data.length() * 100; // Rough estimate for calldata

    // Adjust for network congestion
    uint64_t adjusted_gas_price = std::max(
        config_.gas_price_gwei,
        static_cast<uint64_t>(config_.gas_price_gwei * 1.5) // 50% buffer
    );

    return base_gas + variable_gas;
}

bool HyperliquidAdapter::isTransactionFinalized(const std::string& txHash) {
    // For Hyperliquid, finality is nearly instant (< 1 second)
    // Check transaction status via connector
    try {
        auto tx_future = ws_connector_->getTransaction(txHash);
        tx_future.wait();
        auto tx = tx_future.get();
        return tx.status && tx.blockNumber > 0;
    } catch (...) {
        return false;
    }
}

std::vector<LiquidityPool> HyperliquidAdapter::getLiquidityPools() {
    // Hyperliquid uses automated market makers
    // This would fetch pool information from the API
    std::vector<LiquidityPool> pools;

    try {
        auto meta_future = getMeta();
        meta_future.wait();
        auto meta = meta_future.get();

        // Parse metadata to extract pool information
        for (const auto& coin_info : meta["universe"]) {
            LiquidityPool pool;
            pool.token_a = coin_info["name"];
            pool.token_b = "USD"; // Hyperliquid quotes everything in USD
            pool.reserve_a = coin_info["maxLeverage"];
            pool.reserve_b = coin_info["maxLeverage"]; // Simplified
            pool.fee_rate = coin_info["makerFee"] * 10000; // Convert to basis points
            pool.apr = 0.0; // Would need to calculate from funding rates
            pool.pool_address = coin_info["indexPrice"];
            pool.tvl = coin_info["openInterest"];
            pools.push_back(pool);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error fetching liquidity pools: " << e.what() << std::endl;
    }

    return pools;
}

std::future<HyperliquidAdapter::TradeResult> HyperliquidAdapter::placeOrder(const Order& order) {
    return std::async(std::launch::async, [this, order]() {
        TradeResult result;
        result.order_id = std::to_string(order.client_order_id);
        result.executed_size = 0.0;
        result.executed_price = 0.0;
        result.fee_paid = 0.0;
        result.timestamp = getCurrentTimestamp();
        result.status = "placed";

        try {
            // Create order request
            auto request = createOrderRequest(order);

            // Execute request
            auto response = executeAsyncRequest(request).get();
            auto json_response = nlohmann::json::parse(response);

            if (json_response.contains("status") && json_response["status"] == "ok") {
                if (json_response.contains("response") &&
                    json_response["response"].contains("statuses") &&
                    !json_response["response"]["statuses"].empty()) {

                    auto status = json_response["response"]["statuses"][0];

                    if (status["status"] == "filled") {
                        result.status = "filled";
                        result.executed_size = status["filled"];
                        result.executed_price = status["averagePx"];
                        result.fee_paid = std::abs(order.size * result.executed_price * 0.0005); // 0.05% fee

                        // Update statistics
                        {
                            std::lock_guard<std::mutex> lock(performance_mutex_);
                            total_orders_filled_++;
                            total_volume_usd_ += order.size * result.executed_price;
                        }
                    }
                }

                // Update state
                {
                    std::lock_guard<std::mutex> lock(state_mutex_);
                    open_orders_[result.order_id] = order;
                    last_order_time_ = std::chrono::steady_clock::now();
                }
            } else {
                result.status = "failed";
                result.error_message = json_response.value("response", "Unknown error");
            }

        } catch (const std::exception& e) {
            result.status = "failed";
            result.error_message = e.what();
        }

        return result;
    });
}

std::future<nlohmann::json> HyperliquidAdapter::getMeta() {
    return executeAsyncRequest({
        {"method", "meta"},
        {"id", 1}
    });
}

nlohmann::json HyperliquidAdapter::createOrderRequest(const Order& order) {
    nlohmann::json request = {
        {"method", "exchange"},
        {"id", getCurrentTimestamp()},
        {"request", {
            {"type", "order"},
            {"orders", {createSingleOrderJson(order)}}
        }}
    };

    return request;
}

nlohmann::json HyperliquidAdapter::createSingleOrderJson(const Order& order) {
    nlohmann::json order_json = {
        {"a", order.coin},
        {"b", order.side == OrderSide::BUY ? "b" : "s"}, // b=buy, s=sell
        {"p", formatPrice(order.price, order.coin)},
        {"s", formatSize(order.size, order.coin)},
        {"r", order.reduce_only},
        {"t", {
            {"limit", 1001}, // GTC
            {"ioc", 1002},
            {"fok", 1003}
        }.at("ioc")}, // Default to IOC
        {"c", std::to_string(order.client_order_id)}
    };

    if (order.type != OrderType::MARKET) {
        order_json["p"] = formatPrice(order.price, order.coin);
    }

    return order_json;
}

std::string HyperliquidAdapter::executeRequest(const nlohmann::json& request) {
    try {
        // Create HTTP request
        boost::asio::io_context io_context;
        tcp::resolver resolver(io_context);
        auto const results = resolver.resolve("api.hyperliquid.xyz", "443");

        boost::asio::ssl::context ctx(boost::asio::ssl::context::tlsv12_client);
        boost::beast::ssl_stream<boost::beast::tcp_stream> stream(io_context, ctx);

        boost::beast::tcp_stream(stream.next_layer()).connect(results);
        stream.handshake(boost::asio::ssl::stream_base::client);

        // Create HTTP POST request
        http::request<http::string_body> req{http::verb::post, "/exchange", 11};
        req.set(http::field::host, "api.hyperliquid.xyz");
        req.set(http::field::user_agent, "Minah/1.0");
        req.set(http::field::content_type, "application/json");
        req.body() = request.dump();
        req.prepare_payload();

        // Send the HTTP request
        http::write(stream, req);

        // Receive the HTTP response
        boost::beast::flat_buffer buffer;
        http::response<http::string_body> res;
        http::read(stream, buffer, res);

        // Extract response body
        std::string response_body = res.body();

        // Gracefully close the stream
        boost::system::error_code ec;
        stream.shutdown(ec);

        return response_body;

    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to execute Hyperliquid request: " + std::string(e.what()));
    }
}

std::future<std::string> HyperliquidAdapter::executeAsyncRequest(const nlohmann::json& request) {
    return std::async(std::launch::async, [this, request]() {
        return executeRequest(request);
    });
}

std::string HyperliquidAdapter::formatPrice(double price, const std::string& coin) {
    // Format price with appropriate precision based on coin
    if (coin == "BTC") {
        return std::to_string(price).substr(0, std::to_string(price).find('.') + 2);
    } else if (coin == "ETH") {
        return std::to_string(price).substr(0, std::to_string(price).find('.') + 3);
    } else {
        return std::to_string(price);
    }
}

std::string HyperliquidAdapter::formatSize(double size, const std::string& coin) {
    // Format size with appropriate precision
    return std::to_string(size);
}

void HyperliquidAdapter::handleOrderUpdate(const nlohmann::json& update) {
    std::lock_guard<std::mutex> lock(state_mutex_);

    std::string order_id = update.value("oid", "");
    if (order_id.empty()) return;

    if (update.contains("status")) {
        std::string status = update["status"];

        if (status == "filled" || status == "cancelled") {
            // Remove from open orders
            auto it = open_orders_.find(order_id);
            if (it != open_orders_.end()) {
                open_orders_.erase(it);
            }
        }
    }

    // Notify callbacks
    for (auto& callback : order_callbacks_) {
        callback(update);
    }
}

void HyperliquidAdapter::handleTradeUpdate(const nlohmann::json& update) {
    // Update statistics
    {
        std::lock_guard<std::mutex> lock(performance_mutex_);
        total_orders_placed_++;
    }

    // Notify callbacks
    for (auto& callback : trade_callbacks_) {
        callback(update);
    }
}

uint64_t HyperliquidAdapter::getCurrentTimestamp() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
}

uint64_t HyperliquidAdapter::getCurrentNonce(const std::string& address) {
    // This would get the current nonce from the blockchain
    // For now, return a timestamp-based nonce
    return getCurrentTimestamp();
}

// Factory implementation
std::unique_ptr<HyperliquidAdapter> DEXAdapterFactory::createHyperliquidAdapter(
    const HyperliquidAdapter::Config& config,
    std::shared_ptr<HyperliquidWebSocketConnector> connector) {

    return std::make_unique<HyperliquidAdapter>(config, connector);
}

} // namespace adapters
} // namespace blockchain
} // namespace minah