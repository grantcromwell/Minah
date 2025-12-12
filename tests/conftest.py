"""
Pytest configuration and fixtures for the testing framework
"""
import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import shutil
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

@pytest.fixture(scope="session")
def test_data_dir():
    """Create and return path to test data directory"""
    test_dir = Path(__file__).parent / "fixtures"
    test_dir.mkdir(exist_ok=True)
    yield test_dir
    # Cleanup after tests if needed
    # shutil.rmtree(test_dir, ignore_errors=True)

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='1min')
    returns = np.random.normal(0.0001, 0.01, 1000)
    prices = 100 * (1 + returns).cumprod()
    
    return pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': prices * 1.001,
        'low': prices * 0.999,
        'close': prices * 1.0005,
        'volume': np.random.lognormal(mean=8, sigma=1, size=1000)
    }).set_index('timestamp')

@pytest.fixture
def sample_order_book():
    """Generate sample order book data"""
    levels = 10
    data = []
    
    for i in range(1000):
        ts = datetime.now() - timedelta(seconds=1000-i)
        for side in ['bid', 'ask']:
            for level in range(levels):
                price = 100 - level * 0.1 if side == 'bid' else 100.1 + level * 0.1
                size = np.random.uniform(0.1, 5.0)
                data.append({
                    'timestamp': ts,
                    'side': side,
                    'price': price,
                    'size': size,
                    'level': level
                })
    
    return pd.DataFrame(data)
