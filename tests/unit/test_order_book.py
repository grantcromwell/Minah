""
Unit tests for order book functionality
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path

class TestOrderBook:
    """Test cases for order book implementation"""
    
    def test_order_book_initialization(self, sample_order_book):
        """Test order book initialization and basic properties"""
        from src.abm.order_book import OrderBook
        
        ob = OrderBook(symbol="TEST")
        assert ob.symbol == "TEST"
        assert ob.best_bid is None
        assert ob.best_ask is None
        assert ob.spread is None
        
        # Test with initial orders
        orders = [
            {'price': 99.5, 'size': 1.0, 'side': 'bid'},
            {'price': 100.5, 'size': 1.0, 'side': 'ask'}
        ]
        ob = OrderBook(symbol="TEST", initial_orders=orders)
        assert ob.best_bid == 99.5
        assert ob.best_ask == 100.5
        assert ob.spread == 1.0
    
    def test_add_cancel_orders(self):
        """Test adding and canceling orders"""
        from src.abm.order_book import OrderBook
        
        ob = OrderBook(symbol="TEST")
        
        # Add a bid order
        order_id = ob.add_order(price=99.5, size=1.0, side='bid')
        assert order_id is not None
        assert ob.best_bid == 99.5
        
        # Add an ask order
        order_id = ob.add_order(price=100.5, size=1.0, side='ask')
        assert ob.best_ask == 100.5
        
        # Cancel the bid order
        assert ob.cancel_order(order_id)
        assert ob.best_ask == 100.5
        assert ob.best_bid is None
    
    def test_market_data_generation(self, sample_market_data):
        """Test market data generation and validation"""
        from src.abm.market_data import validate_market_data
        
        # Test valid data
        assert validate_market_data(sample_market_data)
        
        # Test missing columns
        invalid_data = sample_market_data.drop(columns=['open'])
        with pytest.raises(ValueError):
            validate_market_data(invalid_data)
            
        # Test negative prices
        invalid_data = sample_market_data.copy()
        invalid_data.iloc[0, 0] = -100  # Negative open price
        with pytest.raises(ValueError):
            validate_market_data(invalid_data)

class TestRiskManagement:
    """Test cases for risk management components"""
    
    def test_value_at_risk(self):
        """Test Value at Risk calculation"""
        from src.abm.risk import calculate_var
        
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)
        
        # Calculate VaR at 95% confidence
        var_95 = calculate_var(returns, confidence_level=0.95)
        
        # Check that VaR is negative (represents loss)
        assert var_95 < 0
        
        # Check that approximately 5% of returns are below VaR
        pct_below_var = (returns < var_95).mean()
        assert abs(pct_below_var - 0.05) < 0.01  # Within 1% of expected 5%
    
    def test_expected_shortfall(self):
        # Test Expected Shortfall calculation
        from src.abm.risk import calculate_es

        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.001, 0.02, 1000)

        # Calculate ES at 95% confidence
        es_95 = calculate_es(returns, confidence_level=0.95)

        # ES should be more negative than VaR at same confidence level
        var_95 = np.percentile(returns, 5)
        assert es_95 < var_95

        # ES should be the average of the worst 5% of returns
        worst_returns = np.sort(returns)[:50]  # 5% of 1000
        expected_es = worst_returns.mean()
        assert abs(es_95 - expected_es) < 1e-10
