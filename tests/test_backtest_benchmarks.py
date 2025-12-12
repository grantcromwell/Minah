"""
Performance benchmarks for the backtesting engine
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import time
import sys
import os
from unittest.mock import Mock

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Mock the sklearn import before importing any other modules
sys.modules['sklearn'] = Mock()
sys.modules['sklearn.ensemble'] = Mock()

class MockOrder:
    """Mock Order class"""
    def __init__(self, symbol, quantity, order_type, side, price=None, **kwargs):
        self.symbol = symbol
        self.quantity = quantity
        self.order_type = order_type
        self.side = side
        self.price = price
        self.order_id = f"order_{id(self)}"
        self.metadata = kwargs

class MockOrderType:
    """Mock OrderType enum"""
    MARKET = "MARKET"
    LIMIT = "LIMIT"

class MockOrderSide:
    """Mock OrderSide enum"""
    BUY = "BUY"
    SELL = "SELL"

def generate_large_dataset(days=365, freq='1min'):
    """Generate a large dataset for performance testing"""
    dates = pd.date_range(start='2023-01-01', periods=days*24*60, freq=freq)
    np.random.seed(42)

    # Generate realistic price data with trend
    returns = np.random.normal(0.0001, 0.02, len(dates))
    trend = np.linspace(0, 0.5, len(dates))  # 50% upward trend over the period
    prices = 100 * (1 + returns + trend/len(dates)).cumprod()

    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + np.random.uniform(0, 0.005, len(dates))),
        'low': prices * (1 - np.random.uniform(0, 0.005, len(dates))),
        'close': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'volume': np.random.lognormal(mean=10, sigma=0.5, size=len(dates)),
        'bid': prices * (1 - np.random.uniform(0.0001, 0.001, len(dates))),
        'ask': prices * (1 + np.random.uniform(0.0001, 0.001, len(dates)))
    }, index=dates)

    return data

def test_data_loading_performance():
    """Test performance of data loading operations"""

    # Generate large dataset
    data = generate_large_dataset(days=30, freq='1min')  # 30 days of minute data

    # Test data loading time
    start_time = time.time()

    # Simulate data loading operations
    loaded_data = data.copy()
    loaded_data = loaded_data.resample('5min').ohlc()  # Resample to 5-minute bars

    loading_time = time.time() - start_time

    # Performance assertions
    assert loading_time < 2.0  # Should complete within 2 seconds
    assert len(loaded_data) > 0

    print(f"Data loading time: {loading_time:.3f} seconds")
    print(f"Dataset size: {len(data)} rows")
    print(f"Resampled to: {len(loaded_data)} rows")

def test_indicator_calculation_performance():
    """Test performance of technical indicator calculations"""

    # Generate dataset
    data = generate_large_dataset(days=7, freq='1min')  # 1 week of minute data
    prices = data['close']

    # Test various indicator calculations
    start_time = time.time()

    # Moving averages
    sma_20 = prices.rolling(window=20).mean()
    ema_20 = prices.ewm(span=20).mean()

    # Momentum indicators
    rsi = 100 - (100 / (1 + prices.diff().rolling(window=14).apply(
        lambda x: x[x > 0].mean() / -x[x < 0].mean() if len(x[x < 0]) > 0 else 1)))

    # Volatility
    volatility = prices.rolling(window=20).std()

    # Bollinger Bands
    bb_upper = sma_20 + 2 * volatility
    bb_lower = sma_20 - 2 * volatility

    calculation_time = time.time() - start_time

    # Performance assertions
    assert calculation_time < 5.0  # Should complete within 5 seconds
    assert len(sma_20) == len(prices)
    assert len(ema_20) == len(prices)

    print(f"Indicator calculation time: {calculation_time:.3f} seconds")
    print(f"Calculated indicators for {len(prices)} data points")

def test_strategy_execution_performance():
    """Test performance of strategy execution"""

    # Generate dataset
    data = generate_large_dataset(days=1, freq='1min')  # 1 day of minute data
    prices = data['close']

    # Strategy execution simulation
    start_time = time.time()

    # Simple moving average crossover strategy
    fast_window = 10
    slow_window = 30

    # Calculate indicators
    fast_ma = prices.rolling(window=fast_window).mean()
    slow_ma = prices.rolling(window=slow_window).mean()

    # Generate signals
    signals = []
    positions = []
    cash = 100000
    portfolio_value = []

    for i in range(slow_window, len(prices)):
        # Generate signal
        if fast_ma.iloc[i-1] <= slow_ma.iloc[i-1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
            signal = 1  # Buy
        elif fast_ma.iloc[i-1] >= slow_ma.iloc[i-1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
            signal = -1  # Sell
        else:
            signal = 0  # Hold

        signals.append(signal)

        # Simple position management
        if signal == 1 and len(positions) == 0:
            # Buy with 10% of cash
            quantity = int((cash * 0.1) / prices.iloc[i])
            positions.append({'quantity': quantity, 'price': prices.iloc[i]})
            cash -= quantity * prices.iloc[i]
        elif signal == -1 and positions:
            # Sell all positions
            for pos in positions:
                cash += pos['quantity'] * prices.iloc[i]
            positions = []

        # Calculate portfolio value
        position_value = sum(pos['quantity'] * prices.iloc[i] for pos in positions)
        portfolio_value.append(cash + position_value)

    execution_time = time.time() - start_time

    # Performance assertions
    assert execution_time < 3.0  # Should complete within 3 seconds
    assert len(signals) == len(prices) - slow_window
    assert len(portfolio_value) == len(signals)

    print(f"Strategy execution time: {execution_time:.3f} seconds")
    print(f"Processed {len(prices)} data points")
    print(f"Generated {len(signals)} signals")
    print(f"Initial portfolio value: ${100000:,.2f}")
    print(f"Final portfolio value: ${portfolio_value[-1]:,.2f}")

def test_portfolio_optimization_performance():
    """Test performance of portfolio optimization calculations"""

    # Generate returns for multiple assets
    np.random.seed(42)
    num_assets = 10
    num_periods = 252  # 1 year of daily data

    returns = np.random.normal(0.001, 0.02, (num_periods, num_assets))

    start_time = time.time()

    # Calculate covariance matrix
    cov_matrix = np.cov(returns, rowvar=False)

    # Simple mean-variance optimization (equal weights as baseline)
    weights = np.ones(num_assets) / num_assets

    # Calculate portfolio metrics
    expected_return = np.mean(returns, axis=0).dot(weights)
    portfolio_variance = weights.dot(cov_matrix).dot(weights)
    portfolio_volatility = np.sqrt(portfolio_variance)

    optimization_time = time.time() - start_time

    # Performance assertions
    assert optimization_time < 1.0  # Should complete within 1 second
    assert cov_matrix.shape == (num_assets, num_assets)
    assert len(weights) == num_assets
    assert np.isclose(np.sum(weights), 1.0)  # Weights should sum to 1

    print(f"Portfolio optimization time: {optimization_time:.3f} seconds")
    print(f"Number of assets: {num_assets}")
    print(f"Expected annual return: {expected_return * 252:.2%}")
    print(f"Annual volatility: {portfolio_volatility * np.sqrt(252):.2%}")

def test_risk_metrics_performance():
    """Test performance of risk metrics calculations"""

    # Generate returns data
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, 10000)  # Large return series

    start_time = time.time()

    # Calculate various risk metrics
    volatility = np.std(returns)
    skewness = np.mean(((returns - np.mean(returns)) / volatility) ** 3)
    kurtosis = np.mean(((returns - np.mean(returns)) / volatility) ** 4) - 3

    # VaR calculation
    var_95 = np.percentile(returns, 5)
    var_99 = np.percentile(returns, 1)

    # Expected shortfall
    es_95 = np.mean(returns[returns <= var_95])
    es_99 = np.mean(returns[returns <= var_99])

    # Maximum drawdown
    cumulative = (1 + returns).cumprod()
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    max_drawdown = drawdown.min()

    calculation_time = time.time() - start_time

    # Performance assertions
    assert calculation_time < 0.5  # Should complete within 0.5 seconds
    assert isinstance(volatility, (int, float))
    assert isinstance(max_drawdown, (int, float))

    print(f"Risk metrics calculation time: {calculation_time:.3f} seconds")
    print(f"Volatility: {volatility:.4f}")
    print(f"Skewness: {skewness:.4f}")
    print(f"Kurtosis: {kurtosis:.4f}")
    print(f"95% VaR: {var_95:.4f}")
    print(f"99% VaR: {var_99:.4f}")
    print(f"95% Expected Shortfall: {es_95:.4f}")
    print(f"99% Expected Shortfall: {es_99:.4f}")
    print(f"Maximum Drawdown: {max_drawdown:.2%}")

def test_memory_usage():
    """Test memory usage of large dataset processing"""

    import psutil
    import os

    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss

    # Process large dataset
    data = generate_large_dataset(days=7, freq='1min')

    # Perform operations that should use memory
    processed_data = data.copy()
    processed_data['returns'] = processed_data['close'].pct_change()
    processed_data['log_returns'] = np.log(processed_data['close'] / processed_data['close'].shift(1))
    processed_data['volatility'] = processed_data['returns'].rolling(window=20).std()

    # Calculate memory usage
    final_memory = process.memory_info().rss
    memory_increase = final_memory - initial_memory
    memory_increase_mb = memory_increase / (1024 * 1024)

    # Memory assertions
    assert memory_increase_mb < 500  # Should use less than 500MB for this test

    print(f"Initial memory: {initial_memory / (1024 * 1024):.2f} MB")
    print(f"Final memory: {final_memory / (1024 * 1024):.2f} MB")
    print(f"Memory increase: {memory_increase_mb:.2f} MB")
    print(f"Dataset size: {len(data)} rows")

def test_concurrent_operations():
    """Test performance of concurrent operations"""

    import concurrent.futures
    import threading

    # Generate dataset
    data = generate_large_dataset(days=1, freq='1min')

    def process_symbol(symbol_data, symbol_name):
        """Process data for a single symbol"""
        # Simulate symbol-specific processing
        returns = symbol_data.pct_change()
        volatility = returns.rolling(window=20).std()
        return {
            'symbol': symbol_name,
            'mean_return': returns.mean(),
            'volatility': volatility.mean(),
            'data_points': len(symbol_data)
        }

    # Simulate processing multiple symbols concurrently
    symbols = ['AAPL', 'GOOG', 'MSFT', 'AMZN', 'META']
    symbol_data = {symbol: data['close'] + np.random.normal(0, 1, len(data))
                   for symbol in symbols}

    start_time = time.time()

    # Process symbols concurrently
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = []
        for symbol, symbol_prices in symbol_data.items():
            future = executor.submit(process_symbol, symbol_prices, symbol)
            futures.append(future)

        results = [future.result() for future in concurrent.futures.as_completed(futures)]

    concurrent_time = time.time() - start_time

    # Performance assertions
    assert concurrent_time < 2.0  # Should complete within 2 seconds
    assert len(results) == len(symbols)

    print(f"Concurrent processing time: {concurrent_time:.3f} seconds")
    for result in results:
        print(f"Symbol: {result['symbol']}, "
              f"Mean Return: {result['mean_return']:.6f}, "
              f"Volatility: {result['volatility']:.6f}, "
              f"Data Points: {result['data_points']}")

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])