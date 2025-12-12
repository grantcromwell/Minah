#include "BlockchainNodeConnector.h"
#include "WebSocketBlockchainConnector.h"

namespace minah {
namespace blockchain {

std::unique_ptr<BlockchainNodeConnector> BlockchainNodeConnectorFactory::create(
    NetworkType network,
    const std::string& endpoint) {

    WebSocketBlockchainConnector::Config config;

    switch (network) {
        case NetworkType::HYPERLIQUID:
            config.ws_url = endpoint.empty() ? "wss://api.hyperliquid.xyz/ws" : endpoint;
            config.http_url = "https://api.hyperliquid.xyz/info";
            break;

        case NetworkType::ETHEREUM:
            config.ws_url = endpoint.empty() ? "wss://mainnet.infura.io/ws/v3/YOUR_PROJECT_ID" : endpoint;
            config.http_url = "https://mainnet.infura.io/v3/YOUR_PROJECT_ID";
            break;

        case NetworkType::POLYGON:
            config.ws_url = endpoint.empty() ? "wss://polygon-mainnet.infura.io/ws/v3/YOUR_PROJECT_ID" : endpoint;
            config.http_url = "https://polygon-mainnet.infura.io/v3/YOUR_PROJECT_ID";
            break;

        case NetworkType::ARBITRUM:
            config.ws_url = endpoint.empty() ? "wss://arb1.arbitrum.io/ws" : endpoint;
            config.http_url = "https://arb1.arbitrum.io/rpc";
            break;

        case NetworkType::OPTIMISM:
            config.ws_url = endpoint.empty() ? "wss://mainnet.optimism.io/ws" : endpoint;
            config.http_url = "https://mainnet.optimism.io";
            break;

        case NetworkType::BASE:
            config.ws_url = endpoint.empty() ? "wss://mainnet.base.org/ws" : endpoint;
            config.http_url = "https://mainnet.base.org";
            break;
    }

    if (network == NetworkType::HYPERLIQUID) {
        return std::make_unique<HyperliquidWebSocketConnector>(config.ws_url);
    }

    return std::make_unique<WebSocketBlockchainConnector>(network, config);
}

std::unique_ptr<BlockchainNodeConnector> BlockchainNodeConnectorFactory::createWithFailover(
    NetworkType network,
    const std::vector<std::string>& endpoints) {

    // For now, just use the first endpoint
    // A more sophisticated implementation would implement failover logic
    if (endpoints.empty()) {
        return create(network, "");
    }

    return create(network, endpoints[0]);
}

} // namespace blockchain
} // namespace minah