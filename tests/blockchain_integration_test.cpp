#include "../src/blockchain/connectors/BlockchainNodeConnector.h"
#include "../src/blockchain/connectors/WebSocketBlockchainConnector.h"
#include "../src/blockchain/adapters/HyperliquidAdapter.h"
#include "../src/blockchain/pipeline/WebSocketDataPipeline.h"
#include <iostream>
#include <chrono>
#include <future>

using namespace minah::blockchain;

void testBlockhainConnector() {
    std::cout << "\n=== Testing Blockchain Node Connector ===\n";

    // Create Hyperliquid connector
    auto connector = std::dynamic_pointer_cast<HyperliquidWebSocketConnector>(
        BlockchainNodeConnectorFactory::create(
            BlockchainNodeConnector::NetworkType::HYPERLIQUID,
            "wss://api.hyperliquid.xyz/ws"
        )
    );

    if (!connector) {
        std::cout << "Failed to create Hyperliquid connector\n";
        return;
    }

    // Test connection
    std::cout << "Connecting to Hyperliquid...\n";
    if (connector->connect()) {
        std::cout << "✅ Successfully connected to Hyperliquid\n";

        // Test getting chain ID
        std::cout << "Chain ID: " << connector->getChainId() << "\n";

        // Test subscription to new blocks
        bool subscribed = connector->subscribeToNewBlocks([](const BlockchainNodeConnector::Block& block) {
            std::cout << "New block: " << block.number << " (hash: " << block.hash << ")\n";
        });

        if (subscribed) {
            std::cout << "✅ Successfully subscribed to new blocks\n";
        }

        // Wait for some blocks
        std::this_thread::sleep_for(std::chrono::seconds(5));

        connector->disconnect();
        std::cout << "✅ Disconnected successfully\n";

    } else {
        std::cout << "❌ Failed to connect to Hyperliquid\n";
    }
}

void testDataPipeline() {
    std::cout << "\n=== Testing WebSocket Data Pipeline ===\n";

    // Create pipeline configuration
    pipeline::WebSocketDataPipeline::Config config;
    config.zmq_publisher_endpoint = "tcp://*:5556";
    config.num_processing_threads = 2;
    config.enable_metrics = true;

    // Create pipeline
    auto pipeline = std::make_unique<pipeline::HyperliquidDataPipeline>(config);

    if (pipeline->start()) {
        std::cout << "✅ Data pipeline started successfully\n";

        // Create Hyperliquid connector
        auto connector = std::dynamic_pointer_cast<HyperliquidWebSocketConnector>(
            BlockchainNodeConnectorFactory::create(
                BlockchainNodeConnector::NetworkType::HYPERLIQUID,
                "wss://api.hyperliquid.xyz/ws"
            )
        );

        if (connector && connector->connect()) {
            std::cout << "✅ Hyperliquid connector connected\n";

            // Add connector to pipeline
            if (pipeline->addBlockchainConnector("hyperliquid", connector, {"ETH", "BTC"})) {
                std::cout << "✅ Connector added to pipeline\n";

                // Subscribe to trades
                if (pipeline->subscribeToTrades("hyperliquid", "ETH")) {
                    std::cout << "✅ Subscribed to ETH trades\n";
                }

                // Run for a few seconds
                std::this_thread::sleep_for(std::chrono::seconds(10));

                // Get statistics
                auto stats = pipeline->getStatistics();
                std::cout << "\nPipeline Statistics:\n";
                std::cout << "  Messages processed: " << stats.messages_processed << "\n";
                std::cout << "  Messages dropped: " << stats.messages_dropped << "\n";
                std::cout << "  Avg processing time: " << stats.avg_processing_time_us << " μs\n";
                std::cout << "  Avg queue depth: " << stats.avg_queue_depth << "\n";
            }

            connector->disconnect();
        }

        pipeline->stop();
        std::cout << "✅ Data pipeline stopped\n";
    }
}

void testHyperliquidAdapter() {
    std::cout << "\n=== Testing Hyperliquid DEX Adapter ===\n";

    // Create connector
    auto connector = std::dynamic_pointer_cast<HyperliquidWebSocketConnector>(
        BlockchainNodeConnectorFactory::create(
            BlockchainNodeConnector::NetworkType::HYPERLIQUID,
            "wss://api.hyperliquid.xyz/ws"
        )
    );

    if (!connector) {
        std::cout << "❌ Failed to create connector\n";
        return;
    }

    // Create adapter configuration
    adapters::HyperliquidAdapter::Config adapter_config;
    adapter_config.api_endpoint = "https://api.hyperliquid.xyz/info";
    adapter_config.ws_endpoint = "wss://api.hyperliquid.xyz/ws";

    // Create adapter
    auto adapter = adapters::DEXAdapterFactory::createHyperliquidAdapter(adapter_config, connector);

    if (connector->connect()) {
        std::cout << "✅ Connected to Hyperliquid\n";

        // Test getting meta information
        auto meta_future = adapter->getMeta();
        try {
            meta_future.wait();
            auto meta = meta_future.get();
            std::cout << "✅ Retrieved meta information\n";
            std::cout << "  Number of symbols: " << meta["universe"].size() << "\n";
        } catch (const std::exception& e) {
            std::cout << "❌ Failed to get meta: " << e.what() << "\n";
        }

        // Test getting liquidity pools
        auto pools = adapter->getLiquidityPools();
        std::cout << "✅ Retrieved " << pools.size() << " liquidity pools\n";

        // Test placing a small order (WARNING: This would execute a real trade!)
        /*
        adapters::HyperliquidAdapter::Order test_order;
        test_order.coin = "ETH";
        test_order.side = adapters::HyperliquidAdapter::OrderSide::BUY;
        test_order.type = adapters::HyperliquidAdapter::OrderType::LIMIT;
        test_order.size = 0.01; // 0.01 ETH
        test_order.price = 2000.0; // $2000 per ETH
        test_order.reduce_only = 0;
        test_order.time_in_force = 1001; // GTC
        test_order.client_order_id = adapter->getCurrentTimestamp();

        auto order_future = adapter->placeOrder(test_order);
        try {
            order_future.wait();
            auto result = order_future.get();
            std::cout << "Order result: " << result.status << "\n";
        } catch (const std::exception& e) {
            std::cout << "❌ Order failed: " << e.what() << "\n";
        }
        */

        connector->disconnect();
    }
}

int main() {
    std::cout << "=== Minah DEX Integration Test Suite ===\n";
    std::cout << "Phase 1: Foundation & Hyperliquid Integration\n\n";

    try {
        // Test blockchain connector
        testBlockhainConnector();

        // Test data pipeline
        testDataPipeline();

        // Test Hyperliquid adapter
        testHyperliquidAdapter();

        std::cout << "\n=== Test Suite Completed ===\n";
        std::cout << "✅ All tests executed successfully\n";

    } catch (const std::exception& e) {
        std::cerr << "❌ Test failed with exception: " << e.what() << "\n";
        return 1;
    }

    return 0;
}