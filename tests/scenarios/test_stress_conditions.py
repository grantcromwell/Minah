""
Scenario tests for stress conditions and edge cases
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytest

class TestStressScenarios:
    """Tests for handling extreme market conditions"""
    
    def test_flash_crash_scenario(self):
        """Test system behavior during a flash crash"""
        from src.abm.order_book import OrderBook
        from src.abm.risk import RiskManager
        from src.abm.execution import ExecutionEngine
        
        # Initialize components
        ob = OrderBook(symbol="TEST")
        risk_manager = RiskManager(
            max_position_size=10.0,
            max_daily_loss=0.1,
            max_order_size=2.0,
            circuit_breaker_pct=0.05  # 5% price move triggers circuit breaker
        )
        engine = ExecutionEngine(risk_manager=risk_manager)
        
        # Add initial liquidity
        for i in range(10):
            ob.add_order(price=100.0 - (i * 0.1), size=1.0, side='bid')
            ob.add_order(price=100.0 + (i * 0.1), size=1.0, side='ask')
        
        # Simulate flash crash: large sell order
        crash_order = {
            'type': 'market',
            'side': 'sell',
            'size': 100.0,  # Large sell order
            'symbol': 'TEST',
            'trader_id': 'PANIC_SELLER',
            'timestamp': datetime.now()
        }
        
        # Execute the order
        with pytest.raises(Exception) as excinfo:
            engine.execute_order(crash_order, ob)
        
        # Verify circuit breaker was triggered
        assert "circuit breaker" in str(excinfo.value).lower()
        
        # Verify order book state
        assert ob.best_bid is not None
        assert ob.best_ask is not None
        assert (ob.best_ask - ob.best_bid) / ob.best_bid < 0.05  # Spread remains reasonable
    
    def test_low_liquidity_scenario(self):
        """Test system behavior with very low liquidity"""
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        
        # Initialize with very thin order book
        ob = OrderBook(symbol="TEST")
        ob.add_order(price=99.9, size=0.01, side='bid')
        ob.add_order(price=100.1, size=0.01, side='ask')
        
        engine = ExecutionEngine()
        
        # Try to execute a market order
        order = {
            'type': 'market',
            'side': 'buy',
            'size': 1.0,  # Much larger than available liquidity
            'symbol': 'TEST',
            'trader_id': 'LIQUIDITY_TAKER',
            'timestamp': datetime.now()
        }
        
        # Should raise due to insufficient liquidity
        with pytest.raises(Exception) as excinfo:
            engine.execute_order(order, ob)
        
        assert "insufficient liquidity" in str(excinfo.value).lower()
    
    def test_high_frequency_order_flow(self):
        """Test handling of high-frequency order flow"""
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        import time
        
        ob = OrderBook(symbol="TEST")
        engine = ExecutionEngine()
        
        num_orders = 1000
        start_time = time.time()
        
        for i in range(num_orders):
            # Alternate between buy and sell orders
            side = 'buy' if i % 2 == 0 else 'sell'
            price = 100.0 + (0.01 * (i % 10 - 5))  # Oscillate around 100.0
            
            order = {
                'type': 'limit',
                'side': side,
                'price': price,
                'size': 0.1,
                'symbol': 'TEST',
                'trader_id': f'HFT_{i}',
                'timestamp': datetime.now()
            }
            
            try:
                engine.execute_order(order, ob)
            except Exception as e:
                pytest.fail(f"Order {i} failed: {str(e)}")
        
        end_time = time.time()
        orders_per_second = num_orders / (end_time - start_time)
        
        print(f"\nHigh-Frequency Order Processing:")
        print(f"  Processed {num_orders:,} orders in {end_time - start_time:.2f} seconds")
        print(f"  Throughput: {orders_per_second:,.0f} orders/second")
        
        assert orders_per_second > 1000, "Throughput below 1K orders/second"

class TestEdgeCases:
    """Tests for edge cases and error conditions"""
    
    def test_self_trade_prevention(self):
        """Verify self-trading is prevented"""
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        
        ob = OrderBook(symbol="TEST")
        engine = ExecutionEngine()
        
        # Add a resting order
        resting_order = {
            'type': 'limit',
            'side': 'buy',
            'price': 100.0,
            'size': 1.0,
            'symbol': 'TEST',
            'trader_id': 'TRADER_1',
            'timestamp': datetime.now()
        }
        engine.execute_order(resting_order, ob)
        
        # Try to execute an opposing order from the same trader
        aggressive_order = {
            'type': 'limit',
            'side': 'sell',
            'price': 100.0,
            'size': 1.0,
            'symbol': 'TEST',
            'trader_id': 'TRADER_1',  # Same trader ID
            'timestamp': datetime.now()
        }
        
        with pytest.raises(Exception) as excinfo:
            engine.execute_order(aggressive_order, ob)
        
        assert "self-trade" in str(excinfo.value).lower()
    
    def test_fractional_orders(self):
        """Test handling of very small fractional orders"""
        from decimal import Decimal
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        
        ob = OrderBook(symbol="TEST")
        engine = ExecutionEngine()
        
        # Add a resting order
        resting_order = {
            'type': 'limit',
            'side': 'buy',
            'price': 100.0,
            'size': 1.0,
            'symbol': 'TEST',
            'trader_id': 'TRADER_1',
            'timestamp': datetime.now()
        }
        engine.execute_order(resting_order, ob)
        
        # Try to execute a very small order
        small_order = {
            'type': 'limit',
            'side': 'sell',
            'price': 100.0,
            'size': Decimal('0.00000001'),  # Very small size
            'symbol': 'TEST',
            'trader_id': 'TRADER_2',
            'timestamp': datetime.now()
        }
        
        # Should be rejected due to minimum size
        with pytest.raises(ValueError) as excinfo:
            engine.execute_order(small_order, ob)
        
        assert "below minimum" in str(excinfo.value).lower()
    
    def test_extreme_price_orders(self):
        """Test handling of orders with extreme prices"""
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        
        ob = OrderBook(symbol="TEST")
        engine = ExecutionEngine()
        
        # Test very high price
        high_price_order = {
            'type': 'limit',
            'side': 'buy',
            'price': 1e12,  # Extremely high price
            'size': 1.0,
            'symbol': 'TEST',
            'trader_id': 'TRADER_1',
            'timestamp': datetime.now()
        }
        
        with pytest.raises(ValueError) as excinfo:
            engine.execute_order(high_price_order, ob)
        
        assert "price" in str(excinfo.value).lower()
        
        # Test zero/negative price
        zero_price_order = {
            'type': 'limit',
            'side': 'buy',
            'price': 0.0,
            'size': 1.0,
            'symbol': 'TEST',
            'trader_id': 'TRADER_1',
            'timestamp': datetime.now()
        }
        
        with pytest.raises(ValueError) as excinfo:
            engine.execute_order(zero_price_order, ob)
        
        assert "price" in str(excinfo.value).lower()

class TestRecoveryScenarios:
    """Tests for system recovery after failures"""
    
    def test_order_book_recovery(self, tmp_path):
        """Test order book recovery from a snapshot"""
        import os
        import json
        from src.abm.order_book import OrderBook
        
        # Create a test order book
        ob1 = OrderBook(symbol="TEST")
        
        # Add some orders
        orders = []
        for i in range(10):
            order_id = ob1.add_order(
                price=100.0 + (i * 0.1),
                size=1.0,
                side='bid' if i % 2 == 0 else 'ask'
            )
            orders.append((order_id, ob1.get_order(order_id)))
        
        # Save snapshot
        snapshot_path = tmp_path / "order_book_snapshot.json"
        ob1.save_snapshot(snapshot_path)
        
        # Create a new order book and load snapshot
        ob2 = OrderBook(symbol="TEST")
        ob2.load_snapshot(snapshot_path)
        
        # Verify orders were restored
        for order_id, original_order in orders:
            restored_order = ob2.get_order(order_id)
            assert restored_order is not None
            assert restored_order['price'] == original_order['price']
            assert restored_order['size'] == original_order['size']
            assert restored_order['side'] == original_order['side']
    
    def test_market_data_recovery(self):
        """Test recovery of market data feed"""
        from src.abm.market_data import MarketDataFeed
        
        # Create a feed with a simulated disconnection
        feed = MarketDataFeed(reconnect_attempts=3, reconnect_delay=0.1)
        
        # Simulate a disconnection
        feed.connected = False
        
        # Try to get data (should trigger reconnection)
        try:
            feed.get_latest("TEST")
            pytest.fail("Expected ConnectionError")
        except ConnectionError:
            # Expected - we're testing the reconnection logic
            pass
        
        # Verify reconnection was attempted
        assert feed.reconnect_attempts > 0
