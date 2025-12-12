"""
Comprehensive tests for the backtesting engine
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from abm.backtest.engine import BacktestEngine
from abm.backtest.data import HistoricalDataHandler
from abm.backtest.strategies import MovingAverageCrossover
from abm.backtest.results import BacktestResult, PerformanceMetrics


@pytest.fixture
def sample_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='1h')
    np.random.seed(42)

    # Generate realistic price data
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = 100 * (1 + returns).cumprod()

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.01, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.01, len(dates))),
        'close': prices * (1 + np.random.normal(0, 0.005, len(dates))),
        'volume': np.random.lognormal(mean=10, sigma=1, size=len(dates))
    }, index=dates)

    return data


@pytest.fixture
def temp_data_dir(sample_data, tmp_path):
    """Create temporary data directory with sample data"""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()

    # Save sample data
    sample_data.to_csv(data_dir / "aapl_1h.csv")

    return str(data_dir)


class TestHistoricalDataHandler:
    """Test cases for HistoricalDataHandler"""

    def test_load_local_data(self, temp_data_dir):
        """Test loading data from local files"""
        handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        assert len(handler.data) == 1
        assert 'aapl' in handler.data
        assert len(handler.data['aapl']) > 0

        # Check data structure
        df = handler.get_data('aapl')
        assert isinstance(df.index, pd.DatetimeIndex)
        assert all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume'])

    def test_data_iterator(self, temp_data_dir):
        """Test data iteration functionality"""
        handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        # Test iteration
        timestamps = []
        for timestamp, data_slice in handler:
            timestamps.append(timestamp)
            assert isinstance(data_slice, dict)
            assert 'aapl' in data_slice

            # Break after testing a few iterations
            if len(timestamps) >= 5:
                break

        assert len(timestamps) == 5
        assert all(isinstance(ts, datetime) for ts in timestamps)

    def test_date_range_filtering(self, temp_data_dir):
        """Test date range filtering"""
        handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        # Set date range
        start_date = '2023-01-02'
        end_date = '2023-01-05'
        handler.set_date_range(start_date, end_date)

        # Check that data is filtered correctly
        filtered_data = []
        for timestamp, data_slice in handler:
            filtered_data.append(timestamp)

        assert len(filtered_data) > 0
        assert all(start_date <= ts.strftime('%Y-%m-%d') <= end_date for ts in filtered_data)


class TestBacktestEngine:
    """Test cases for BacktestEngine"""

    def test_engine_initialization(self, temp_data_dir):
        """Test engine initialization"""
        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            initial_cash=100000,
            start_date='2023-01-01',
            end_date='2023-01-05'
        )

        assert engine.initial_cash == 100000
        assert engine.cash == 100000
        assert engine.data_handler == data_handler
        assert len(engine.positions) == 0
        assert len(engine.trades) == 0
        assert len(engine.equity_curve) == 0

    def test_add_strategy(self, temp_data_dir):
        """Test adding a strategy to the engine"""
        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            initial_cash=100000
        )

        # Add strategy
        engine.add_strategy(MovingAverageCrossover, fast_window=5, slow_window=10)

        assert engine.strategy is not None
        assert isinstance(engine.strategy, MovingAverageCrossover)
        assert engine.strategy.parameters['fast_window'] == 5
        assert engine.strategy.parameters['slow_window'] == 10

    def test_run_backtest(self, temp_data_dir):
        """Test running a complete backtest"""
        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            initial_cash=100000,
            start_date='2023-01-02',
            end_date='2023-01-10'
        )

        # Add strategy
        engine.add_strategy(MovingAverageCrossover, fast_window=5, slow_window=10)

        # Run backtest
        result = engine.run()

        # Check result structure
        assert isinstance(result, BacktestResult)
        assert isinstance(result.metrics, PerformanceMetrics)
        assert isinstance(result.trades, pd.DataFrame)
        assert isinstance(result.equity_curve, pd.DataFrame)
        assert isinstance(result.positions, dict)

        # Check that we have some data
        assert len(result.equity_curve) > 0
        assert 'total' in result.equity_curve.columns
        assert 'cash' in result.equity_curve.columns
        assert 'return' in result.equity_curve.columns

        # Check final equity
        final_equity = result.equity_curve['total'].iloc[-1]
        assert isinstance(final_equity, (int, float))
        assert final_equity >= 0  # Should not have negative equity

    def test_performance_metrics(self, temp_data_dir):
        """Test performance metrics calculation"""
        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            initial_cash=100000,
            start_date='2023-01-02',
            end_date='2023-01-10'
        )

        engine.add_strategy(MovingAverageCrossover, fast_window=3, slow_window=8)
        result = engine.run()

        metrics = result.metrics

        # Check metric types
        assert isinstance(metrics.total_return, (int, float))
        assert isinstance(metrics.annual_return, (int, float))
        assert isinstance(metrics.sharpe_ratio, (int, float))
        assert isinstance(metrics.max_drawdown, (int, float))
        assert isinstance(metrics.win_rate, (int, float))
        assert isinstance(metrics.profit_factor, (int, float))
        assert isinstance(metrics.num_trades, int)

        # Check reasonable values
        assert metrics.num_trades >= 0
        assert 0 <= metrics.win_rate <= 1
        assert metrics.profit_factor >= 0

    def test_no_strategy_error(self, temp_data_dir):
        """Test that running without a strategy raises an error"""
        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            initial_cash=100000
        )

        # Should raise error when running without strategy
        with pytest.raises(ValueError, match="No strategy added"):
            engine.run()


class TestMovingAverageStrategy:
    """Test cases for MovingAverageCrossover strategy"""

    def test_strategy_initialization(self, temp_data_dir):
        """Test strategy initialization"""
        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        strategy = MovingAverageCrossover(
            data_handler=data_handler,
            execution_engine=None,  # Mock execution engine
            initial_capital=100000,
            fast_window=10,
            slow_window=20
        )

        assert strategy.parameters['fast_window'] == 10
        assert strategy.parameters['slow_window'] == 20
        assert strategy.initial_capital == 100000
        assert strategy.cash == 100000
        assert strategy.equity == 100000

    def test_moving_average_calculation(self, temp_data_dir):
        """Test moving average calculations"""
        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        strategy = MovingAverageCrossover(
            data_handler=data_handler,
            execution_engine=None,
            fast_window=5,
            slow_window=10
        )

        # Initialize strategy
        strategy.initialize()

        # Check that moving average lists are initialized
        assert 'aapl' in strategy.fast_ma
        assert 'aapl' in strategy.slow_ma
        assert len(strategy.fast_ma['aapl']) == 0
        assert len(strategy.slow_ma['aapl']) == 0


class TestPerformanceAnalysis:
    """Test performance analysis and optimization"""

    def test_execution_speed(self, temp_data_dir):
        """Test execution speed of backtest"""
        import time

        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            initial_cash=100000,
            start_date='2023-01-02',
            end_date='2023-01-10'
        )

        engine.add_strategy(MovingAverageCrossover, fast_window=5, slow_window=10)

        # Measure execution time
        start_time = time.time()
        result = engine.run()
        end_time = time.time()

        execution_time = end_time - start_time

        # Check that execution is reasonable (should be fast for this small dataset)
        assert execution_time < 5.0  # Should complete in under 5 seconds

        # Check that we got valid results
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 0

    def test_memory_usage(self, temp_data_dir):
        """Test memory usage during backtest"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        data_handler = HistoricalDataHandler(
            data_source='local',
            data_dir=temp_data_dir,
            symbols=['aapl'],
            data_frequency='1h'
        )

        engine = BacktestEngine(
            data_handler=data_handler,
            initial_cash=100000,
            start_date='2023-01-01',
            end_date='2023-01-31'  # Use more data for memory test
        )

        engine.add_strategy(MovingAverageCrossover, fast_window=5, slow_window=10)
        result = engine.run()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 100MB for this test)
        assert memory_increase < 100 * 1024 * 1024  # 100MB in bytes


if __name__ == "__main__":
    pytest.main([__file__, "-v"])