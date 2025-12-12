# Minah Blockchain Integration

This directory contains the blockchain integration components for Minah's DEX capabilities. The implementation provides institutional-grade connectivity to decentralized exchanges, with initial focus on Hyperliquid and planned expansion to other major DEX protocols.

## Architecture

### Core Components

1. **Connectors** (`connectors/`)
   - `BlockchainNodeConnector.h` - Base interface for blockchain node connections
   - `WebSocketBlockchainConnector.h` - WebSocket-based implementation for real-time data
   - `HyperliquidWebSocketConnector.h` - Hyperliquid-specific connector

2. **Adapters** (`adapters/`)
   - `HyperliquidAdapter.h` - DEX trading adapter for Hyperliquid
   - Supports order placement, cancellation, position management
   - Implements MEV protection and gas optimization

3. **Data Pipeline** (`pipeline/`)
   - `WebSocketDataPipeline.h` - Real-time market data processing pipeline
   - ZeroMQ-based publishing for strategy consumption
   - Multi-threaded processing with batch optimization

4. **Utilities** (`utils/`)
   - `Utils.h` - Hex conversion, Ethereum address utilities

## Key Features

### Performance
- Sub-500ms execution latency on compatible chains
- WebSocket streaming for real-time market data
- Lock-free data structures for high-throughput processing
- Batch processing for optimal gas usage

### Reliability
- Automatic reconnection with exponential backoff
- Multiple endpoint failover support
- Message deduplication to prevent duplicate processing
- Comprehensive health monitoring and metrics

### Security
- TLS/WSS encryption for all communications
- Hardware security module (HSM) integration ready
- Transaction signing with EIP-712 structured data
- MEV protection through private mempools

## Getting Started

### Building

```bash
# Clone web3cpp dependency (handled by CMake)
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Running Tests

```bash
# Run the blockchain integration test suite
./blockchain_integration_test
```

### Basic Usage

```cpp
#include "blockchain/connectors/BlockchainNodeConnector.h"
#include "blockchain/adapters/HyperliquidAdapter.h"

// Create connector
auto connector = minah::blockchain::BlockchainNodeConnectorFactory::create(
    minah::blockchain::NetworkType::HYPERLIQUID,
    "wss://api.hyperliquid.xyz/ws"
);

// Connect
if (connector->connect()) {
    std::cout << "Connected to Hyperliquid" << std::endl;

    // Create adapter
    minah::blockchain::adapters::HyperliquidAdapter::Config config;
    auto adapter = minah::blockchain::adapters::DEXAdapterFactory::createHyperliquidAdapter(
        config, connector);

    // Place order
    minah::blockchain::adapters::HyperliquidAdapter::Order order;
    order.coin = "ETH";
    order.side = minah::blockchain::adapters::HyperliquidAdapter::OrderSide::BUY;
    order.type = minah::blockchain::adapters::HyperliquidAdapter::OrderType::MARKET;
    order.size = 0.1;

    auto result = adapter->placeOrder(order).get();
    std::cout << "Order result: " << result.status << std::endl;
}
```

## Configuration

### Environment Variables

- `HYPERLIQUID_API_URL` - Hyperliquid API endpoint
- `HYPERLIQUID_WS_URL` - Hyperliquid WebSocket endpoint
- `PRIVATE_KEY` - Private key for transaction signing (hex string)

### Configuration Files

Create `config/blockchain.yaml`:

```yaml
blockchain:
  connectors:
    hyperliquid:
      network: "hyperliquid"
      ws_url: "wss://api.hyperliquid.xyz/ws"
      http_url: "https://api.hyperliquid.xyz/info"
      chain_id: 998

  data_pipeline:
    zmq_publisher_endpoint: "tcp://*:5556"
    num_processing_threads: 4
    batch_size: 100
    enable_metrics: true

  security:
    use_private_mempool: true
    max_gas_price_gwei: 10
    slippage_tolerance_bps: 50
```

## Roadmap

### Phase 1 (Current - Q1 2025)
- [x] Hyperliquid integration
- [x] WebSocket data streaming
- [x] Basic order execution
- [ ] Testnet deployment
- [ ] Security audit

### Phase 2 (Q2 2025)
- [ ] Uniswap V3 integration
- [ ] Arbitrum support
- [ ] Gas optimization engine
- [ ] Advanced order types (TWAP, VWAP)

### Phase 3 (Q3 2025)
- [ ] Cross-chain routing
- [ ] MEV protection strategies
- [ ] Layer 2 optimizations
- [ ] Institutional compliance features

## Monitoring

### Metrics

The pipeline exposes the following metrics through ZeroMQ:

- `messages_processed` - Total messages processed
- `messages_dropped` - Messages dropped due to queue overflow
- `avg_processing_time_us` - Average processing time in microseconds
- `avg_queue_depth` - Average queue depth
- `last_sequence_number` - Last processed sequence number

### Health Checks

```bash
# Check connector status
curl http://localhost:8080/health/blockchain

# Check pipeline statistics
curl http://localhost:8080/metrics/blockchain
```

## Security Considerations

1. **Private Key Management**
   - Store private keys securely (HSM or encrypted vault)
   - Never log or expose private keys
   - Use hardware wallets for production deployments

2. **Network Security**
   - Use WSS/TLS for all communications
   - Validate SSL certificates
   - Implement rate limiting to prevent DoS

3. **Transaction Security**
   - Implement transaction replay protection
   - Use deterministic nonce management
   - Validate all contract interactions

4. **MEV Protection**
   - Use private mempools when available
   - Implement commit-reveal schemes for large orders
   - Monitor for front-running attempts

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check WebSocket endpoint URL
   - Verify network connectivity
   - Check SSL certificate validity

2. **Order Rejections**
   - Verify account has sufficient margin
   - Check gas price is adequate
   - Ensure nonce is correct

3. **Data Pipeline Lag**
   - Increase processing threads
   - Check ZeroMQ queue depth
   - Monitor network latency

### Logging

Enable debug logging:

```cpp
// In your initialization code
minah::Logger::setLevel(minah::Logger::DEBUG);
```

Logs will be written to `logs/blockchain.log`.

## Contributing

When contributing to the blockchain integration:

1. Follow the existing code style
2. Add comprehensive unit tests
3. Update documentation
4. Ensure security review for all changes

## License

This component is part of Minah and licensed under the Apache License 2.0.