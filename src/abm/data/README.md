# ABM Data Module

This module provides comprehensive data handling, processing, and connectivity for the Agent-Based Modeling (ABM) system. It includes components for market data fetching, preprocessing, validation, gap filling, and technical analysis.

## Features

- **Market Data Connector**: Fetch real-time and historical market data from various exchanges (via CCXT)
- **Data Pipeline**: Process and transform raw market data with ML-based gap filling and anomaly detection
- **Data Validation**: Comprehensive validation and quality assurance for market data
- **Technical Analysis**: Calculate various technical indicators and performance metrics
- **Risk Metrics**: Compute risk and performance metrics for trading strategies
- **Configuration Management**: Flexible configuration system for data sources and processing parameters

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```python
from abm.data import MarketDataConnector, DataPipeline, MarketDataValidator
from abm.data.config import get_config

# Load configuration
config = get_config()

# Initialize components
connector = MarketDataConnector(config=config)
data_pipeline = DataPipeline(config=config)
validator = MarketDataValidator()

# Fetch OHLCV data
df = connector.fetch_ohlcv(
    symbol='BTC/USDT',
    timeframe='1d',
    limit=1000
)

# Process data
processed_df = data_pipeline.process_data(df)

# Validate data
report = validator.validate_ohlcv(processed_df)
print(f"Data quality score: {report.summary['data_quality_score']}")
```

### Configuration

Create a `config/abm_data.yaml` file to customize settings:

```yaml
data_sources:
  default: 'binance'
  available:
    binance:
      class: 'ccxt.binance'
      rate_limit: true
      enable_rate_limit: true
      timeout: 30000
    
processing:
  resample_freq: '1m'
  min_liquidity: 1e-6
  max_price_change: 0.1
  max_volume_change: 5.0
  zscore_threshold: 3.0
  min_data_points: 100
  max_gap_minutes: 60
  confidence_level: 0.95

caching:
  enabled: true
  cache_dir: '.cache/abm_data'
  max_size_mb: 1024
  ttl_days: 7
```

### Data Validation

```python
from abm.data import MarketDataValidator

validator = MarketDataValidator()
report = validator.validate_ohlcv(df)

# Print summary
print(f"Status: {report.summary['status']}")
print(f"Total issues: {report.summary['total_issues']}")
print(f"High priority issues: {report.summary['high_priority_issues']}")

# Get issues as DataFrame
issues_df = report.to_dataframe()
if not issues_df.empty:
    print("\nIssues found:")
    print(issues_df[['type', 'severity', 'message']])
```

### Technical Analysis

```python
from abm.data.utils import calculate_technical_indicators

# Calculate technical indicators
df_with_indicators = calculate_technical_indicators(
    df,
    price_col='close',
    volume_col='volume'
)

# Available indicators:
# - Moving Averages (SMA, EMA)
# - Bollinger Bands
# - RSI (Relative Strength Index)
# - MACD (Moving Average Convergence Divergence)
# - VWAP (Volume Weighted Average Price)
# - ATR (Average True Range)
```

### Risk Metrics

```python
from abm.data.utils import calculate_common_risk_metrics

# Calculate risk metrics
metrics = calculate_common_risk_metrics(
    returns=df['returns'].values,
    benchmark_returns=benchmark_df['returns'].values,
    risk_free_rate=0.02  # 2% annual risk-free rate
)

print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
print(f"Profit Factor: {metrics['profit_factor']:.2f}")
```

## Components

### MarketDataConnector

Fetches market data from various sources (exchanges, APIs, files) with built-in caching and rate limiting.

### DataPipeline

Processes raw market data, handling:
- Resampling and alignment
- Missing data imputation
- Anomaly detection and handling
- Feature engineering
- ML-based gap filling

### MarketDataValidator

Validates data quality and detects issues:
- Missing values
- Outliers and anomalies
- Price and volume spikes
- Time gaps and irregularities
- Order book integrity

### Utils

Utility functions for:
- Technical analysis indicators
- Risk and performance metrics
- Data transformation and manipulation
- Statistical analysis

## Examples

### Fetch and Process Data

```python
from abm.data import MarketDataConnector, DataPipeline

# Initialize components
connector = MarketDataConnector()
data_pipeline = DataPipeline()

# Fetch data
df = connector.fetch_ohlcv('BTC/USDT', '1d', limit=1000)

# Process data
processed_df = data_pipeline.process_data(df)

# Add technical indicators
from abm.data.utils import calculate_technical_indicators
df_with_ta = calculate_technical_indicators(processed_df)
```

### Backtesting with Risk Metrics

```python
import pandas as pd
from abm.data.utils import calculate_common_risk_metrics

# Simulate strategy returns
returns = pd.Series(np.random.normal(0.001, 0.02, 1000))
benchmark_returns = pd.Series(np.random.normal(0.0005, 0.015, 1000))

# Calculate metrics
metrics = calculate_common_risk_metrics(
    returns=returns,
    benchmark_returns=benchmark_returns,
    risk_free_rate=0.02
)

# Display key metrics
print(f"Annualized Return: {metrics['cagr']:.2%}")
print(f"Volatility: {metrics['volatility']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
print(f"Win Rate: {metrics['win_rate']:.2%}")
```

## Configuration

The module can be configured via:
1. YAML configuration file (recommended)
2. Environment variables
3. Programmatic configuration

### Environment Variables

- `ABM_DATA_CONFIG`: Path to configuration file
- `ABM_DATA_CACHE_DIR`: Cache directory path
- `ABM_DATA_SOURCE`: Default data source (e.g., 'binance', 'file')

## Dependencies

- pandas
- numpy
- scipy
- scikit-learn
- ccxt
- pyyaml
- numba (for performance optimization)
- prophet (for ML-based gap filling)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
