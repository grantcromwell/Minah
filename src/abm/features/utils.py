""
Feature Engineering Utilities for Financial Time Series

This module provides utility functions for creating common financial features.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
import talib
import logging

logger = logging.getLogger(__name__)

def create_lag_features(
    data: pd.DataFrame,
    columns: List[str],
    lags: List[int],
    fill_method: str = 'ffill'
) -> pd.DataFrame:
    """
    Create lagged features for specified columns.
    
    Args:
        data: Input DataFrame
        columns: List of column names to create lags for
        lags: List of lag periods
        fill_method: Method to handle NaN values ('ffill', 'bfill', or 'drop')
        
    Returns:
        DataFrame with lagged features
    """
    result = data.copy()
    
    for col in columns:
        if col not in data.columns:
            logger.warning(f"Column '{col}' not found in data")
            continue
            
        for lag in lags:
            result[f"{col}_lag{lag}"] = data[col].shift(lag)
    
    # Handle NaN values
    if fill_method == 'ffill':
        result = result.ffill()
    elif fill_method == 'bfill':
        result = result.bfill()
    elif fill_method == 'drop':
        result = result.dropna()
    
    return result

def create_rolling_features(
    data: pd.DataFrame,
    columns: List[str],
    windows: List[int],
    aggregations: List[str] = ['mean', 'std', 'min', 'max'],
    min_periods: int = 1
) -> pd.DataFrame:
    """
    Create rolling window features for specified columns.
    
    Args:
        data: Input DataFrame
        columns: List of column names to create rolling features for
        windows: List of window sizes
        aggregations: List of aggregation functions to apply
        min_periods: Minimum number of observations in window required
        
    Returns:
        DataFrame with rolling features
    """
    result = data.copy()
    
    for col in columns:
        if col not in data.columns:
            logger.warning(f"Column '{col}' not found in data")
            continue
            
        for window in windows:
            for agg in aggregations:
                if hasattr(pd.Series, agg):
                    result[f"{col}_roll_{window}_{agg}"] = (
                        data[col]
                        .rolling(window=window, min_periods=min_periods)
                        .agg(agg)
                    )
    
    return result

def create_ta_features(
    data: pd.DataFrame,
    open_col: str = 'open',
    high_col: str = 'high',
    low_col: str = 'low',
    close_col: str = 'close',
    volume_col: Optional[str] = None,
    include_indicators: List[str] = None,
    indicator_params: Optional[Dict[str, Dict[str, Any]]] = None
) -> pd.DataFrame:
    """
    Create technical analysis features using TA-Lib.
    
    Args:
        data: Input DataFrame with OHLCV data
        open_col: Name of the open price column
        high_col: Name of the high price column
        low_col: Name of the low price column
        close_col: Name of the close price column
        volume_col: Name of the volume column (optional)
        include_indicators: List of indicators to include (None for all)
        indicator_params: Dictionary of indicator-specific parameters
        
    Returns:
        DataFrame with technical indicators
    """
    result = data.copy()
    indicator_params = indicator_params or {}
    
    # Default indicators to include if not specified
    if include_indicators is None:
        include_indicators = [
            'sma', 'ema', 'rsi', 'macd', 'bbands', 'stoch', 'atr', 'obv', 'adx'
        ]
    
    # Ensure we have required columns
    required_cols = [open_col, high_col, low_col, close_col]
    if not all(col in data.columns for col in required_cols):
        missing = [col for col in required_cols if col not in data.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Get OHLCV data
    open_prices = data[open_col].values
    high_prices = data[high_col].values
    low_prices = data[low_col].values
    close_prices = data[close_col].values
    volumes = data[volume_col].values if volume_col and volume_col in data.columns else None
    
    # Calculate indicators
    if 'sma' in include_indicators:
        periods = indicator_params.get('sma', {}).get('periods', [20, 50, 200])
        for period in periods:
            result[f'sma_{period}'] = talib.SMA(close_prices, timeperiod=period)
    
    if 'ema' in include_indicators:
        periods = indicator_params.get('ema', {}).get('periods', [12, 26])
        for period in periods:
            result[f'ema_{period}'] = talib.EMA(close_prices, timeperiod=period)
    
    if 'rsi' in include_indicators:
        period = indicator_params.get('rsi', {}).get('period', 14)
        result['rsi'] = talib.RSI(close_prices, timeperiod=period)
    
    if 'macd' in include_indicators:
        fast = indicator_params.get('macd', {}).get('fastperiod', 12)
        slow = indicator_params.get('macd', {}).get('slowperiod', 26)
        signal = indicator_params.get('macd', {}).get('signalperiod', 9)
        
        macd, macd_signal, macd_hist = talib.MACD(
            close_prices,
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=signal
        )
        
        result['macd'] = macd
        result['macd_signal'] = macd_signal
        result['macd_hist'] = macd_hist
    
    if 'bbands' in include_indicators:
        period = indicator_params.get('bbands', {}).get('period', 20)
        nbdevup = indicator_params.get('bbands', {}).get('nbdevup', 2)
        nbdevdn = indicator_params.get('bbands', {}).get('nbdevdn', 2)
        
        upper, middle, lower = talib.BBANDS(
            close_prices,
            timeperiod=period,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )
        
        result['bb_upper'] = upper
        result['bb_middle'] = middle
        result['bb_lower'] = lower
    
    if 'stoch' in include_indicators:
        fastk = indicator_params.get('stoch', {}).get('fastk_period', 5)
        slowk = indicator_params.get('stoch', {}).get('slowk_period', 3)
        slowd = indicator_params.get('stoch', {}).get('slowd_period', 3)
        
        slowk, slowd = talib.STOCH(
            high_prices,
            low_prices,
            close_prices,
            fastk_period=fastk,
            slowk_period=slowk,
            slowk_matype=0,
            slowd_period=slowd,
            slowd_matype=0
        )
        
        result['stoch_slowk'] = slowk
        result['stoch_slowd'] = slowd
    
    if 'atr' in include_indicators:
        period = indicator_params.get('atr', {}).get('period', 14)
        result['atr'] = talib.ATR(high_prices, low_prices, close_prices, timeperiod=period)
    
    if 'obv' in include_indicators and volumes is not None:
        result['obv'] = talib.OBV(close_prices, volumes)
    
    if 'adx' in include_indicators:
        period = indicator_params.get('adx', {}).get('period', 14)
        result['adx'] = talib.ADX(high_prices, low_prices, close_prices, timeperiod=period)
    
    return result

def create_crossing_features(
    data: pd.DataFrame,
    fast_col: str,
    slow_col: str,
    prefix: str = ''
) -> pd.DataFrame:
    """
    Create crossing features between two columns.
    
    Args:
        data: Input DataFrame
        fast_col: Name of the faster moving column
        slow_col: Name of the slower moving column
        prefix: Prefix for the output column names
        
    Returns:
        DataFrame with crossing features
    """
    result = data.copy()
    
    # Create crossing signals
    result[f'{prefix}cross_above'] = (data[fast_col] > data[slow_col]) & (data[fast_col].shift(1) <= data[slow_col].shift(1))
    result[f'{prefix}cross_below'] = (data[fast_col] < data[slow_col]) & (data[fast_col].shift(1) >= data[slow_col].shift(1))
    
    # Create crossover flags
    result[f'{prefix}crossover'] = 0
    result.loc[result[f'{prefix}cross_above'], f'{prefix}crossover'] = 1
    result.loc[result[f'{prefix}cross_below'], f'{prefix}crossover'] = -1
    
    return result

def create_time_features(
    data: pd.DataFrame,
    datetime_col: str = 'timestamp',
    include: List[str] = None
) -> pd.DataFrame:
    """
    Create time-based features from a datetime column.
    
    Args:
        data: Input DataFrame with a datetime column
        datetime_col: Name of the datetime column
        include: List of time features to include:
            - 'hour': Hour of day
            - 'day': Day of month
            - 'weekday': Day of week (0-6)
            - 'month': Month (1-12)
            - 'quarter': Quarter (1-4)
            - 'year': Year
            - 'is_weekend': Boolean for weekend days
            - 'is_month_start': Boolean for first day of month
            - 'is_month_end': Boolean for last day of month
            
    Returns:
        DataFrame with time features
    """
    if include is None:
        include = ['hour', 'weekday', 'month', 'is_weekend']
    
    result = data.copy()
    
    if datetime_col not in data.columns:
        raise ValueError(f"Datetime column '{datetime_col}' not found in data")
    
    dt_series = pd.to_datetime(data[datetime_col])
    
    if 'hour' in include:
        result['hour'] = dt_series.dt.hour
    if 'day' in include:
        result['day'] = dt_series.dt.day
    if 'weekday' in include:
        result['weekday'] = dt_series.dt.weekday
    if 'month' in include:
        result['month'] = dt_series.dt.month
    if 'quarter' in include:
        result['quarter'] = dt_series.dt.quarter
    if 'year' in include:
        result['year'] = dt_series.dt.year
    if 'is_weekend' in include:
        result['is_weekend'] = dt_series.dt.weekday >= 5
    if 'is_month_start' in include:
        result['is_month_start'] = dt_series.dt.is_month_start
    if 'is_month_end' in include:
        result['is_month_end'] = dt_series.dt.is_month_end
    
    return result
