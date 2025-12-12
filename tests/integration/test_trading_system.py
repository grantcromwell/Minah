""
Integration tests for the trading system
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class TestTradingSystemIntegration:
    """Integration tests for the trading system"""
    
    def test_market_order_execution(self, sample_order_book):
        """Test market order execution flow"""
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        from src.abm.agents import MarketMaker, Arbitrageur
        
        # Initialize components
        order_book = OrderBook(symbol="TEST")
        execution_engine = ExecutionEngine()
        market_maker = MarketMaker("MM1", order_book)
        arbitrageur = Arbitrageur("ARB1", order_book)
        
        # Add initial liquidity
        market_maker.place_orders()
        
        # Place a market buy order
        initial_balance = 100000
        order = {
            'type': 'market',
            'side': 'buy',
            'size': 1.0,
            'symbol': 'TEST',
            'trader_id': 'TEST_TRADER',
            'timestamp': datetime.now()
        }
        
        # Execute the order
        execution = execution_engine.execute_order(order, order_book)
        
        # Verify execution
        assert execution['status'] == 'filled'
        assert execution['filled_size'] == 1.0
        assert execution['avg_price'] > 0
        
        # Verify balance update
        new_balance = initial_balance - execution['avg_price'] * execution['filled_size']
        assert 0 < new_balance < initial_balance
    
    def test_limit_order_matching(self):
        """Test limit order matching logic"""
        from src.abm.order_book import OrderBook
        
        ob = OrderBook(symbol="TEST")
        
        # Add a limit buy order
        buy_order_id = ob.add_order(price=100.0, size=1.0, side='bid')
        
        # Add a matching limit sell order
        sell_order_id = ob.add_order(price=100.0, size=1.0, side='ask')
        
        # Verify orders are matched
        assert ob.best_bid is None
        assert ob.best_ask is None
        
        # Verify order status
        assert ob.get_order(buy_order_id)['status'] == 'filled'
        assert ob.get_order(sell_order_id)['status'] == 'filled'
    
    def test_risk_limits(self):
        """Test risk limit enforcement"""
        from src.abm.risk import RiskManager
        from src.abm.order_book import OrderBook
        
        ob = OrderBook(symbol="TEST")
        risk_manager = RiskManager(
            max_position_size=10.0,
            max_daily_loss=0.1,  # 10% max daily loss
            max_order_size=2.0
        )
        
        # Test order size limit
        assert not risk_manager.check_order_risk(ob, 'buy', 3.0)  # Exceeds max order size
        assert risk_manager.check_order_risk(ob, 'buy', 1.5)      # Within limits
        
        # Test position limit
        ob.positions['TEST'] = {'size': 9.5, 'avg_price': 100.0}
        assert not risk_manager.check_order_risk(ob, 'buy', 1.0)  # Would exceed position limit
        
        # Test daily loss limit
        ob.pnl = -0.11 * 100000  # 11% loss on 100k portfolio
        assert not risk_manager.check_order_risk(ob, 'buy', 0.1)  # Exceeds daily loss limit

class TestBacktestingIntegration:
    """Integration tests for backtesting functionality"""
    
    def test_backtest_end_to_end(self, sample_market_data):
        """Test complete backtesting workflow"""
        from src.abm.backtest import BacktestEngine
        from src.abm.strategies import MeanReversionStrategy
        
        # Initialize backtest
        strategy = MeanReversionStrategy(
            lookback=20,
            entry_z=1.0,
            exit_z=0.5
        )
        
        backtest = BacktestEngine(
            data=sample_market_data,
            strategy=strategy,
            initial_capital=100000.0,
            commission=0.001  # 0.1% commission
        )
        
        # Run backtest
        results = backtest.run()
        
        # Verify results structure
        assert 'returns' in results
        assert 'positions' in results
        assert 'trades' in results
        assert 'sharpe_ratio' in results
        
        # Basic sanity checks
        assert len(results['returns']) == len(sample_market_data)
        assert results['sharpe_ratio'] is not None
        
        # Verify that trades were generated
        assert len(results['trades']) > 0
        
        # Verify position sizing
        max_position = max(abs(p) for p in results['positions'])
        assert 0 < max_position <= 1.0  # Assuming position sizing is normalized

class TestMLIntegration:
    """Integration tests for ML components"""
    
    def test_feature_engineering_pipeline(self, sample_market_data):
        """Test feature engineering pipeline"""
        from src.abm.ml.features import FeatureEngineer
        
        # Initialize feature engineer
        feature_engineer = FeatureEngineer(
            features=['returns', 'volatility', 'momentum'],
            lookback_windows=[5, 10, 20]
        )
        
        # Generate features
        features = feature_engineer.transform(sample_market_data)
        
        # Verify output
        assert not features.isnull().any().any()
        assert len(features) == len(sample_market_data)
        
        # Check that expected features were created
        expected_columns = [
            'returns_5', 'returns_10', 'returns_20',
            'volatility_5', 'volatility_10', 'volatility_20',
            'momentum_5', 'momentum_10', 'momentum_20'
        ]
        
        for col in expected_columns:
            assert col in features.columns
    
    def test_ml_model_integration(self, sample_market_data):
        """Test ML model training and prediction"""
        from src.abm.ml.models import MLPredictor
        
        # Create train/test split
        train_size = int(len(sample_market_data) * 0.8)
        train_data = sample_market_data.iloc[:train_size]
        test_data = sample_market_data.iloc[train_size:]
        
        # Initialize and train model
        model = MLPredictor(
            model_type='random_forest',
            target_horizon=5,
            features=['returns_5', 'volatility_5', 'momentum_5']
        )
        
        # Train model
        model.train(train_data)
        
        # Make predictions
        predictions = model.predict(test_data)
        
        # Verify predictions
        assert len(predictions) == len(test_data)
        assert all(0 <= p <= 1 for p in predictions)  # Assuming binary classification
