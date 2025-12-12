""
Utility functions for data processing and analysis.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numba
import logging
from datetime import datetime, timedelta
import pytz
import hashlib
import json
import os
from pathlib import Path
import pickle
import gzip
import shutil

logger = logging.getLogger(__name__)

# Type aliases
ArrayLike = Union[np.ndarray, pd.Series, List[float]]
Numeric = Union[int, float, np.number]

def resample_data(df: pd.DataFrame, 
                 freq: str = '1min', 
                 agg: Optional[Dict[str, Union[str, List[str]]]] = None) -> pd.DataFrame:
    """
    Resample time series data to a new frequency.
    
    Args:
        df: Input DataFrame with a DatetimeIndex
        freq: Target frequency (e.g., '1min', '5min', '1H', '1D')
        agg: Dictionary mapping columns to aggregation functions
            
    Returns:
        Resampled DataFrame
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex")
    
    # Default aggregation if not specified
    if agg is None:
        agg = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'count': 'sum'
        }
    
    # Apply resampling
    resampled = df.resample(freq)
    
    # Apply aggregations
    result = pd.DataFrame()
    for col, func in agg.items():
        if col in df.columns:
            if isinstance(func, (list, tuple)):
                # Multiple aggregation functions
                result[[f"{col}_{f}" for f in func]] = resampled[col].agg(func)
            else:
                # Single aggregation function
                result[col] = resampled[col].agg(func)
    
    # Forward fill NaN values for OHLC data
    ohlc_cols = [c for c in ['open', 'high', 'low', 'close'] if c in result.columns]
    if ohlc_cols:
        result[ohlc_cols] = result[ohlc_cols].ffill()
    
    return result

@numba.jit(nopython=True)
def calculate_returns(prices: np.ndarray, 
                     period: int = 1, 
                     log_returns: bool = False) -> np.ndarray:
    """
    Calculate returns from a price series.
    
    Args:
        prices: Array of prices
        period: Number of periods to calculate returns over
        log_returns: If True, calculate log returns instead of simple returns
        
    Returns:
        Array of returns
    """
    n = len(prices)
    returns = np.empty(n)
    returns[:period] = np.nan
    
    if log_returns:
        for i in range(period, n):
            returns[i] = np.log(prices[i] / prices[i - period])
    else:
        for i in range(period, n):
            returns[i] = (prices[i] - prices[i - period]) / prices[i - period]
    
    return returns

def calculate_technical_indicators(df: pd.DataFrame, 
                                 price_col: str = 'close',
                                 volume_col: str = 'volume') -> pd.DataFrame:
    """
    Calculate common technical indicators.
    
    Args:
        df: Input DataFrame with OHLCV data
        price_col: Name of the price column
        volume_col: Name of the volume column
        
    Returns:
        DataFrame with added technical indicators
    """
    if df.empty:
        return df
    
    result = df.copy()
    prices = result[price_col].values
    
    # Simple Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        result[f'sma_{window}'] = result[price_col].rolling(window=window).mean()
    
    # Exponential Moving Averages
    for window in [5, 10, 20, 50, 100, 200]:
        result[f'ema_{window}'] = result[price_col].ewm(span=window, adjust=False).mean()
    
    # Bollinger Bands
    window = 20
    result['bb_middle'] = result[price_col].rolling(window=window).mean()
    result['bb_std'] = result[price_col].rolling(window=window).std()
    result['bb_upper'] = result['bb_middle'] + 2 * result['bb_std']
    result['bb_lower'] = result['bb_middle'] - 2 * result['bb_std']
    
    # RSI (Relative Strength Index)
    delta = result[price_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    result['rsi'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = result[price_col].ewm(span=12, adjust=False).mean()
    exp2 = result[price_col].ewm(span=26, adjust=False).mean()
    result['macd'] = exp1 - exp2
    result['macd_signal'] = result['macd'].ewm(span=9, adjust=False).mean()
    result['macd_hist'] = result['macd'] - result['macd_signal']
    
    # Volume Weighted Average Price (VWAP)
    if volume_col in result.columns:
        result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
        result['tp_volume'] = result['typical_price'] * result[volume_col]
        result['vwap'] = result['tp_volume'].cumsum() / result[volume_col].cumsum()
        result.drop(['typical_price', 'tp_volume'], axis=1, inplace=True)
    
    # ATR (Average True Range)
    high_low = result['high'] - result['low']
    high_close = (result['high'] - result['close'].shift()).abs()
    low_close = (result['low'] - result['close'].shift()).abs()
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    result['atr'] = true_range.rolling(window=14).mean()
    
    # Remove any columns with all NaN values
    result = result.dropna(axis=1, how='all')
    
    return result

def detect_regimes(prices: ArrayLike, 
                  n_regimes: int = 3,
                  method: str = 'hmm') -> np.ndarray:
    """
    Detect market regimes in price data.
    
    Args:
        prices: Array-like of prices or returns
        n_regimes: Number of regimes to detect
        method: Method to use ('hmm' for Hidden Markov Model, 'kmeans' for K-means)
        
    Returns:
        Array of regime labels (0 to n_regimes-1)
    """
    from sklearn.mixture import GaussianMixture
    from hmmlearn import hmm
    
    if not isinstance(prices, np.ndarray):
        prices = np.asarray(prices)
    
    # Calculate returns if prices are provided
    if len(prices) > 1 and np.all(prices > 0):
        returns = np.diff(np.log(prices))
    else:
        returns = prices
    
    # Reshape for sklearn
    X = returns.reshape(-1, 1)
    
    if method.lower() == 'hmm' and len(X) > 10:
        try:
            # Use Gaussian HMM
            model = hmm.GaussianHMM(
                n_components=n_regimes,
                covariance_type="diag",
                n_iter=1000,
                random_state=42
            )
            model.fit(X)
            regimes = model.predict(X)
        except:
            # Fall back to GMM if HMM fails
            model = GaussianMixture(n_components=n_regimes, random_state=42)
            regimes = model.fit_predict(X)
    else:
        # Use Gaussian Mixture Model
        model = GaussianMixture(n_components=n_regimes, random_state=42)
        regimes = model.fit_predict(X)
    
    return regimes

def calculate_volatility(returns: ArrayLike, 
                        window: int = 21, 
                        annualize: bool = True) -> np.ndarray:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Array-like of returns
        window: Rolling window size
        annualize: Whether to annualize the volatility
        
    Returns:
        Array of volatility values
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Calculate rolling standard deviation
    vol = pd.Series(returns).rolling(window=window).std()
    
    # Annualize if requested (assuming daily data)
    if annualize:
        vol = vol * np.sqrt(252)  # 252 trading days in a year
    
    return vol.values

def calculate_correlation(series1: ArrayLike, 
                         series2: ArrayLike, 
                         window: int = 21) -> np.ndarray:
    """
    Calculate rolling correlation between two series.
    
    Args:
        series1: First time series
        series2: Second time series
        window: Rolling window size
        
    Returns:
        Array of correlation values
    """
    if len(series1) != len(series2):
        raise ValueError("Input series must have the same length")
    
    s1 = pd.Series(series1)
    s2 = pd.Series(series2)
    
    return s1.rolling(window=window).corr(s2).values

def calculate_drawdown(prices: ArrayLike) -> np.ndarray:
    """
    Calculate drawdown from a price series.
    
    Args:
        prices: Array-like of prices
        
    Returns:
        Array of drawdown values (as positive percentages)
    """
    if not isinstance(prices, np.ndarray):
        prices = np.asarray(prices)
    
    # Calculate cumulative maximum
    cummax = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (cummax - prices) / cummax
    
    return np.where(np.isfinite(drawdown), drawdown, 0.0)

def calculate_sharpe_ratio(returns: ArrayLike, 
                          risk_free_rate: float = 0.0,
                          annualize: bool = True) -> float:
    """
    Calculate the Sharpe ratio.
    
    Args:
        returns: Array-like of returns
        risk_free_rate: Annual risk-free rate (as a decimal)
        annualize: Whether to annualize the Sharpe ratio
        
    Returns:
        Sharpe ratio
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252 if annualize else risk_free_rate)
    
    # Calculate mean and standard deviation
    mean_return = np.mean(excess_returns)
    std_return = np.std(returns, ddof=1)
    
    # Handle edge cases
    if std_return == 0:
        return np.inf if mean_return > 0 else -np.inf
    
    # Calculate Sharpe ratio
    sharpe = mean_return / std_return
    
    # Annualize if requested (assuming daily returns)
    if annualize:
        sharpe *= np.sqrt(252)
    
    return float(sharpe)

def calculate_sortino_ratio(returns: ArrayLike, 
                           risk_free_rate: float = 0.0,
                           annualize: bool = True) -> float:
    """
    Calculate the Sortino ratio.
    
    Args:
        returns: Array-like of returns
        risk_free_rate: Annual risk-free rate (as a decimal)
        annualize: Whether to annualize the ratio
        
    Returns:
        Sortino ratio
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252 if annualize else risk_free_rate)
    
    # Calculate mean and downside deviation
    mean_return = np.mean(excess_returns)
    downside_returns = returns[returns < 0]
    
    if len(downside_returns) == 0:
        return np.inf if mean_return > 0 else -np.inf
    
    downside_dev = np.std(downside_returns, ddof=1)
    
    # Handle edge cases
    if downside_dev == 0:
        return np.inf if mean_return > 0 else -np.inf
    
    # Calculate Sortino ratio
    sortino = mean_return / downside_dev
    
    # Annualize if requested (assuming daily returns)
    if annualize:
        sortino *= np.sqrt(252)
    
    return float(sortino)

def calculate_max_drawdown(prices: ArrayLike) -> float:
    """
    Calculate the maximum drawdown from a price series.
    
    Args:
        prices: Array-like of prices
        
    Returns:
        Maximum drawdown (as a positive decimal)
    """
    if not isinstance(prices, np.ndarray):
        prices = np.asarray(prices)
    
    if len(prices) < 2:
        return 0.0
    
    # Calculate cumulative maximum
    cummax = np.maximum.accumulate(prices)
    
    # Calculate drawdown
    drawdown = (cummax - prices) / cummax
    
    return float(np.nanmax(drawdown)) if len(drawdown) > 0 else 0.0

def calculate_calmar_ratio(returns: ArrayLike, 
                          prices: Optional[ArrayLike] = None,
                          risk_free_rate: float = 0.0,
                          annualize: bool = True) -> float:
    """
    Calculate the Calmar ratio.
    
    Args:
        returns: Array-like of returns
        prices: Optional array-like of prices (used if returns not provided)
        risk_free_rate: Annual risk-free rate (as a decimal)
        annualize: Whether to annualize the ratio
        
    Returns:
        Calmar ratio
    """
    if prices is not None:
        max_dd = calculate_max_drawdown(prices)
    else:
        if not isinstance(returns, np.ndarray):
            returns = np.asarray(returns)
        
        # Calculate cumulative returns to get prices
        cum_returns = np.cumprod(1 + returns) - 1
        max_dd = calculate_max_drawdown(1 + cum_returns)
    
    if max_dd == 0:
        return np.inf
    
    # Calculate annualized return
    if annualize:
        annual_return = np.mean(returns) * 252 - risk_free_rate
    else:
        annual_return = np.mean(returns) - (risk_free_rate / 252 if annualize else risk_free_rate)
    
    return float(annual_return / max_dd)

def calculate_omega_ratio(returns: ArrayLike, 
                         threshold: float = 0.0,
                         annualize: bool = True) -> float:
    """
    Calculate the Omega ratio.
    
    Args:
        returns: Array-like of returns
        threshold: Return threshold (default is 0)
        annualize: Whether to annualize the ratio
        
    Returns:
        Omega ratio
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate excess returns over threshold
    excess_returns = returns - threshold
    
    # Calculate gains and losses
    gains = excess_returns[excess_returns > 0].sum()
    losses = -excess_returns[excess_returns < 0].sum()
    
    # Handle edge cases
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    
    # Calculate Omega ratio
    omega = gains / losses
    
    return float(omega)

def calculate_value_at_risk(returns: ArrayLike, 
                          confidence_level: float = 0.95,
                          method: str = 'historical') -> float:
    """
    Calculate Value at Risk (VaR).
    
    Args:
        returns: Array-like of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        method: Method to use ('historical' or 'parametric')
        
    Returns:
        VaR (as a negative number representing a loss)
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return np.nan
    
    if method.lower() == 'historical':
        # Historical VaR
        return float(np.percentile(returns, (1 - confidence_level) * 100))
    elif method.lower() == 'parametric':
        # Parametric (Gaussian) VaR
        mean = np.mean(returns)
        std = np.std(returns, ddof=1)
        from scipy.stats import norm
        z_score = norm.ppf(1 - confidence_level)
        return float(mean + z_score * std)
    else:
        raise ValueError(f"Unknown VaR method: {method}")

def calculate_expected_shortfall(returns: ArrayLike, 
                               confidence_level: float = 0.95) -> float:
    """
    Calculate Expected Shortfall (ES) / Conditional Value at Risk (CVaR).
    
    Args:
        returns: Array-like of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        Expected Shortfall (as a negative number representing a loss)
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return np.nan
    
    # Sort returns
    sorted_returns = np.sort(returns)
    
    # Calculate VaR
    var = np.percentile(returns, (1 - confidence_level) * 100)
    
    # Calculate ES as the average of returns worse than VaR
    es = sorted_returns[sorted_returns <= var].mean()
    
    return float(es)

def calculate_beta(returns: ArrayLike, 
                  benchmark_returns: ArrayLike) -> float:
    """
    Calculate beta against a benchmark.
    
    Args:
        returns: Array-like of asset returns
        benchmark_returns: Array-like of benchmark returns
        
    Returns:
        Beta coefficient
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate covariance and variance
    cov_matrix = np.cov(returns, benchmark_returns, ddof=1)
    
    # Beta = cov(r, m) / var(m)
    beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] != 0 else np.nan
    
    return float(beta)

def calculate_tracking_error(returns: ArrayLike, 
                           benchmark_returns: ArrayLike) -> float:
    """
    Calculate tracking error.
    
    Args:
        returns: Array-like of asset returns
        benchmark_returns: Array-like of benchmark returns
        
    Returns:
        Tracking error (annualized standard deviation of active returns)
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate active returns
    active_returns = returns - benchmark_returns
    
    # Calculate standard deviation of active returns and annualize (assuming daily data)
    tracking_error = np.std(active_returns, ddof=1) * np.sqrt(252)
    
    return float(tracking_error)

def calculate_information_ratio(returns: ArrayLike, 
                              benchmark_returns: ArrayLike) -> float:
    """
    Calculate the Information Ratio.
    
    Args:
        returns: Array-like of asset returns
        benchmark_returns: Array-like of benchmark returns
        
    Returns:
        Information Ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate active returns
    active_returns = returns - benchmark_returns
    
    # Calculate tracking error
    tracking_error = np.std(active_returns, ddof=1)
    
    # Handle edge case where tracking error is zero
    if tracking_error == 0:
        return np.inf if np.mean(active_returns) > 0 else -np.inf
    
    # Information Ratio = mean(active_returns) / tracking_error
    info_ratio = np.mean(active_returns) / tracking_error
    
    # Annualize (assuming daily data)
    info_ratio *= np.sqrt(252)
    
    return float(info_ratio)

def calculate_ulcer_index(prices: ArrayLike) -> float:
    """
    Calculate the Ulcer Index.
    
    Args:
        prices: Array-like of prices
        
    Returns:
        Ulcer Index
    """
    if not isinstance(prices, np.ndarray):
        prices = np.asarray(prices)
    
    if len(prices) < 2:
        return 0.0
    
    # Calculate drawdowns
    cummax = np.maximum.accumulate(prices)
    drawdowns = (cummax - prices) / cummax
    
    # Calculate Ulcer Index
    ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
    
    return float(ulcer_index)

def calculate_sterling_ratio(returns: ArrayLike, 
                           risk_free_rate: float = 0.0,
                           annualize: bool = True) -> float:
    """
    Calculate the Sterling Ratio.
    
    Args:
        returns: Array-like of returns
        risk_free_rate: Annual risk-free rate (as a decimal)
        annualize: Whether to annualize the ratio
        
    Returns:
        Sterling Ratio
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252 if annualize else risk_free_rate)
    
    # Calculate average excess return
    avg_return = np.mean(excess_returns)
    
    # Calculate average drawdown (using negative of max drawdown)
    avg_drawdown = -calculate_max_drawdown(np.cumprod(1 + returns) - 1)
    
    # Handle edge case where average drawdown is zero
    if avg_drawdown == 0:
        return np.inf if avg_return > 0 else -np.inf
    
    # Calculate Sterling Ratio
    sterling_ratio = avg_return / avg_drawdown
    
    # Annualize if requested (assuming daily returns)
    if annualize:
        sterling_ratio *= np.sqrt(252)
    
    return float(sterling_ratio)

def calculate_treynor_ratio(returns: ArrayLike, 
                          benchmark_returns: ArrayLike,
                          risk_free_rate: float = 0.0,
                          annualize: bool = True) -> float:
    """
    Calculate the Treynor Ratio.
    
    Args:
        returns: Array-like of asset returns
        benchmark_returns: Array-like of benchmark returns
        risk_free_rate: Annual risk-free rate (as a decimal)
        annualize: Whether to annualize the ratio
        
    Returns:
        Treynor Ratio
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate beta
    beta = calculate_beta(returns, benchmark_returns)
    
    # Handle edge case where beta is zero
    if beta == 0:
        return np.inf if np.mean(returns) > 0 else -np.inf
    
    # Calculate excess returns
    excess_returns = returns - (risk_free_rate / 252 if annualize else risk_free_rate)
    
    # Calculate Treynor Ratio
    treynor_ratio = np.mean(excess_returns) / beta
    
    # Annualize if requested (assuming daily returns)
    if annualize:
        treynor_ratio *= 252
    
    return float(treynor_ratio)

def calculate_jensens_alpha(returns: ArrayLike, 
                          benchmark_returns: ArrayLike,
                          risk_free_rate: float = 0.0,
                          annualize: bool = True) -> float:
    """
    Calculate Jensen's Alpha.
    
    Args:
        returns: Array-like of asset returns
        benchmark_returns: Array-like of benchmark returns
        risk_free_rate: Annual risk-free rate (as a decimal)
        annualize: Whether to annualize the alpha
        
    Returns:
        Jensen's Alpha
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate beta
    beta = calculate_beta(returns, benchmark_returns)
    
    # Calculate average returns
    avg_return = np.mean(returns)
    avg_benchmark_return = np.mean(benchmark_returns)
    
    # Calculate risk-free rate for the period
    rf = risk_free_rate / 252 if annualize else risk_free_rate
    
    # Calculate Jensen's Alpha
    alpha = avg_return - (rf + beta * (avg_benchmark_return - rf))
    
    # Annualize if requested (assuming daily returns)
    if annualize:
        alpha *= 252
    
    return float(alpha)

def calculate_m2_ratio(returns: ArrayLike, 
                     benchmark_returns: ArrayLike,
                     risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Modigliani-Modigliani (M²) Ratio.
    
    Args:
        returns: Array-like of asset returns
        benchmark_returns: Array-like of benchmark returns
        risk_free_rate: Annual risk-free rate (as a decimal)
        
    Returns:
        M² Ratio (as a decimal, e.g., 0.05 for 5%)
    """
    if len(returns) != len(benchmark_returns):
        raise ValueError("Returns and benchmark returns must have the same length")
    
    # Remove NaN values
    mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
    returns = returns[mask]
    benchmark_returns = benchmark_returns[mask]
    
    if len(returns) < 2:
        return np.nan
    
    # Calculate Sharpe ratios
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate)
    benchmark_sharpe = calculate_sharpe_ratio(benchmark_returns, risk_free_rate)
    
    # Calculate M² Ratio
    m2_ratio = risk_free_rate + (sharpe_ratio - benchmark_sharpe) * np.std(benchmark_returns) * np.sqrt(252)
    
    return float(m2_ratio)

def calculate_win_rate(returns: ArrayLike) -> float:
    """
    Calculate the win rate (percentage of positive returns).
    
    Args:
        returns: Array-like of returns
        
    Returns:
        Win rate (as a decimal between 0 and 1)
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate win rate
    win_rate = np.mean(returns > 0)
    
    return float(win_rate)

def calculate_profit_factor(returns: ArrayLike) -> float:
    """
    Calculate the profit factor (gross profits / gross losses).
    
    Args:
        returns: Array-like of returns
        
    Returns:
        Profit factor (1.0 is break-even, >1.0 is profitable, <1.0 is losing)
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate gross profits and losses
    gross_profits = returns[returns > 0].sum()
    gross_losses = -returns[returns < 0].sum()
    
    # Handle edge case of no losses
    if gross_losses == 0:
        return np.inf if gross_profits > 0 else 0.0
    
    # Calculate profit factor
    profit_factor = gross_profits / gross_losses
    
    return float(profit_factor)

def calculate_average_win_loss(returns: ArrayLike) -> Tuple[float, float]:
    """
    Calculate the average win and average loss.
    
    Args:
        returns: Array-like of returns
        
    Returns:
        Tuple of (average_win, average_loss) where both are positive numbers
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0, 0.0
    
    # Calculate average win and loss
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    avg_win = np.mean(wins) if len(wins) > 0 else 0.0
    avg_loss = -np.mean(losses) if len(losses) > 0 else 0.0
    
    return float(avg_win), float(avg_loss)

def calculate_expectancy(returns: ArrayLike) -> float:
    """
    Calculate the expectancy of a trading strategy.
    
    Args:
        returns: Array-like of returns
        
    Returns:
        Expectancy (expected return per trade)
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) == 0:
        return 0.0
    
    # Calculate win rate, average win, and average loss
    win_rate = calculate_win_rate(returns)
    avg_win, avg_loss = calculate_average_win_loss(returns)
    
    # Calculate expectancy
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
    
    return float(expectancy)

def calculate_kelly_criterion(returns: ArrayLike) -> float:
    """
    Calculate the Kelly Criterion for optimal position sizing.
    
    Args:
        returns: Array-like of returns
        
    Returns:
        Kelly fraction (suggested fraction of capital to bet)
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 0.0
    
    # Calculate win rate, average win, and average loss
    win_rate = calculate_win_rate(returns)
    avg_win, avg_loss = calculate_average_win_loss(returns)
    
    # Handle edge cases
    if avg_loss == 0:
        return 1.0 if win_rate > 0.5 else 0.0
    
    # Calculate Kelly fraction
    win_loss_ratio = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Ensure the result is between 0 and 1
    return float(max(0.0, min(kelly, 1.0)))

def calculate_ulcer_performance_index(returns: ArrayLike, 
                                    prices: Optional[ArrayLike] = None,
                                    risk_free_rate: float = 0.0) -> float:
    """
    Calculate the Ulcer Performance Index (UPI) / Martin Ratio.
    
    Args:
        returns: Array-like of returns
        prices: Optional array-like of prices (used if returns not provided)
        risk_free_rate: Annual risk-free rate (as a decimal)
        
    Returns:
        Ulcer Performance Index
    """
    if prices is not None:
        ulcer_index = calculate_ulcer_index(prices)
    else:
        if not isinstance(returns, np.ndarray):
            returns = np.asarray(returns)
        
        # Calculate cumulative returns to get prices
        cum_returns = np.cumprod(1 + returns) - 1
        ulcer_index = calculate_ulcer_index(1 + cum_returns)
    
    if ulcer_index == 0:
        return np.inf
    
    # Calculate excess return (assuming returns are annualized)
    excess_return = np.mean(returns) * 252 - risk_free_rate
    
    # Calculate UPI
    upi = excess_return / ulcer_index
    
    return float(upi)

def calculate_tail_ratio(returns: ArrayLike, 
                        percentile: float = 5.0) -> float:
    """
    Calculate the Tail Ratio (ratio of right tail to left tail).
    
    Args:
        returns: Array-like of returns
        percentile: Percentile to use for tail calculation (e.g., 5 for 5th and 95th percentiles)
        
    Returns:
        Tail Ratio
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return 1.0
    
    # Calculate percentiles
    left_tail = np.percentile(returns, percentile)
    right_tail = np.percentile(returns, 100 - percentile)
    
    # Handle edge case where left tail is zero
    if left_tail == 0:
        return np.inf if right_tail > 0 else 1.0
    
    # Calculate Tail Ratio
    tail_ratio = abs(right_tail / left_tail)
    
    return float(tail_ratio)

def calculate_common_risk_metrics(returns: ArrayLike, 
                                benchmark_returns: Optional[ArrayLike] = None,
                                risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Calculate a comprehensive set of risk and performance metrics.
    
    Args:
        returns: Array-like of returns
        benchmark_returns: Optional array-like of benchmark returns
        risk_free_rate: Annual risk-free rate (as a decimal)
        
    Returns:
        Dictionary of risk metrics
    """
    if not isinstance(returns, np.ndarray):
        returns = np.asarray(returns)
    
    # Remove NaN values
    returns = returns[~np.isnan(returns)]
    
    if len(returns) < 2:
        return {}
    
    # Calculate cumulative returns to get prices for some metrics
    cum_returns = np.cumprod(1 + returns) - 1
    prices = 1 + cum_returns  # Normalized to start at 1.0
    
    # Calculate metrics
    metrics = {
        'total_return': float((1 + returns).prod() - 1),
        'cagr': float((1 + returns).prod() ** (252 / len(returns)) - 1) if len(returns) > 0 else 0.0,
        'volatility': float(np.std(returns, ddof=1) * np.sqrt(252)),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate),
        'max_drawdown': calculate_max_drawdown(prices),
        'calmar_ratio': calculate_calmar_ratio(returns, prices=prices, risk_free_rate=risk_free_rate),
        'omega_ratio': calculate_omega_ratio(returns, threshold=risk_free_rate / 252),
        'var_95': calculate_value_at_risk(returns, confidence_level=0.95),
        'expected_shortfall_95': calculate_expected_shortfall(returns, confidence_level=0.95),
        'win_rate': calculate_win_rate(returns),
        'profit_factor': calculate_profit_factor(returns),
        'expectancy': calculate_expectancy(returns),
        'kelly_criterion': calculate_kelly_criterion(returns),
        'ulcer_index': calculate_ulcer_index(prices),
        'ulcer_performance_index': calculate_ulcer_performance_index(returns, prices=prices, risk_free_rate=risk_free_rate),
        'tail_ratio_5': calculate_tail_ratio(returns, percentile=5.0)
    }
    
    # Calculate benchmark-relative metrics if benchmark is provided
    if benchmark_returns is not None:
        if not isinstance(benchmark_returns, np.ndarray):
            benchmark_returns = np.asarray(benchmark_returns)
        
        # Remove NaN values and ensure same length
        mask = ~(np.isnan(returns) | np.isnan(benchmark_returns))
        returns_aligned = returns[mask]
        benchmark_aligned = benchmark_returns[mask]
        
        if len(returns_aligned) > 1:
            metrics.update({
                'beta': calculate_beta(returns_aligned, benchmark_aligned),
                'alpha': calculate_jensens_alpha(returns_aligned, benchmark_aligned, risk_free_rate),
                'tracking_error': calculate_tracking_error(returns_aligned, benchmark_aligned),
                'information_ratio': calculate_information_ratio(returns_aligned, benchmark_aligned),
                'treynor_ratio': calculate_treynor_ratio(returns_aligned, benchmark_aligned, risk_free_rate),
                'm2_ratio': calculate_m2_ratio(returns_aligned, benchmark_aligned, risk_free_rate)
            })
    
    return metrics
