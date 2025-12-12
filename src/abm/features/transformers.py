"""
Feature Transformers for Financial Time Series

This module implements various feature transformers for financial time series data.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Union
from scipy import stats
import talib
import logging

from .pipeline import FeatureTransformer

logger = logging.getLogger(__name__)

class LogReturns(FeatureTransformer):
    """Calculate log returns for specified columns."""
    
    def __init__(self, columns: List[str] = None, prefix: str = 'log_ret_'):
        """
        Initialize the log returns transformer.
        
        Args:
            columns: List of column names to calculate log returns for
            prefix: Prefix for the new column names
        """
        super().__init__(name='LogReturns')
        self.columns = columns
        self.prefix = prefix
        self.feature_names_ = []
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'LogReturns':
        """Fit the transformer (no-op for this transformer)."""
        if self.columns is None:
            self.columns = data.select_dtypes(include=[np.number]).columns.tolist()
        return super().fit(data, **kwargs)
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate log returns for the specified columns."""
        result = data.copy()
        
        for col in self.columns:
            if col in data.columns:
                new_col = f"{self.prefix}{col}"
                result[new_col] = np.log(data[col] / data[col].shift(1))
                self.feature_names_.append(new_col)
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the features created by this transformer."""
        return self.feature_names_

class TechnicalIndicators(FeatureTransformer):
    """Calculate technical indicators using TA-Lib."""
    
    def __init__(self, indicators: List[Dict[str, Any]] = None):
        """
        Initialize the technical indicators transformer.
        
        Args:
            indicators: List of indicator configurations. Each configuration is a dict with:
                - name: Name of the indicator (e.g., 'SMA', 'RSI')
                - params: Dictionary of parameters for the indicator
                - columns: List of column names to apply the indicator to
                - prefix: Optional prefix for the output column names
        """
        super().__init__(name='TechnicalIndicators')
        self.indicators = indicators or []
        self.feature_names_ = []
    
    def add_indicator(self, name: str, params: Dict[str, Any], 
                     columns: List[str] = None, prefix: str = None) -> 'TechnicalIndicators':
        """
        Add an indicator to the transformer.
        
        Args:
            name: Name of the indicator (e.g., 'SMA', 'RSI')
            params: Parameters for the indicator
            columns: Columns to apply the indicator to
            prefix: Optional prefix for the output column names
            
        Returns:
            self: Returns the transformer instance
        """
        self.indicators.append({
            'name': name,
            'params': params,
            'columns': columns or ['close'],
            'prefix': prefix or f"{name.lower()}_"
        })
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate technical indicators for the specified columns."""
        result = data.copy()
        
        for indicator in self.indicators:
            name = indicator['name'].upper()
            params = indicator['params']
            columns = indicator['columns']
            prefix = indicator['prefix']
            
            for col in columns:
                if col not in data.columns:
                    logger.warning(f"Column '{col}' not found for indicator {name}")
                    continue
                
                try:
                    if name == 'SMA':
                        period = params.get('timeperiod', 14)
                        result[f"{prefix}{col}_{period}"] = talib.SMA(
                            data[col].values, 
                            timeperiod=period
                        )
                        
                    elif name == 'EMA':
                        period = params.get('timeperiod', 14)
                        result[f"{prefix}{col}_{period}"] = talib.EMA(
                            data[col].values, 
                            timeperiod=period
                        )
                        
                    elif name == 'RSI':
                        period = params.get('timeperiod', 14)
                        result[f"{prefix}{col}_{period}"] = talib.RSI(
                            data[col].values, 
                            timeperiod=period
                        )
                        
                    elif name == 'MACD':
                        fastperiod = params.get('fastperiod', 12)
                        slowperiod = params.get('slowperiod', 26)
                        signalperiod = params.get('signalperiod', 9)
                        
                        macd, signal, hist = talib.MACD(
                            data[col].values,
                            fastperiod=fastperiod,
                            slowperiod=slowperiod,
                            signalperiod=signalperiod
                        )
                        
                        result[f"{prefix}{col}_macd"] = macd
                        result[f"{prefix}{col}_signal"] = signal
                        result[f"{prefix}{col}_hist"] = hist
                        
                        self.feature_names_.extend([
                            f"{prefix}{col}_macd",
                            f"{prefix}{col}_signal",
                            f"{prefix}{col}_hist"
                        ])
                        continue
                        
                    elif name == 'BBANDS':
                        timeperiod = params.get('timeperiod', 20)
                        nbdevup = params.get('nbdevup', 2)
                        nbdevdn = params.get('nbdevdn', 2)
                        
                        upper, middle, lower = talib.BBANDS(
                            data[col].values,
                            timeperiod=timeperiod,
                            nbdevup=nbdevup,
                            nbdevdn=nbdevdn
                        )
                        
                        result[f"{prefix}{col}_upper"] = upper
                        result[f"{prefix}{col}_middle"] = middle
                        result[f"{prefix}{col}_lower"] = lower
                        
                        self.feature_names_.extend([
                            f"{prefix}{col}_upper",
                            f"{prefix}{col}_middle",
                            f"{prefix}{col}_lower"
                        ])
                        continue
                        
                    elif name == 'STOCH':
                        fastk_period = params.get('fastk_period', 5)
                        slowk_period = params.get('slowk_period', 3)
                        slowd_period = params.get('slowd_period', 3)
                        
                        high = data.get('high', data[col])
                        low = data.get('low', data[col])
                        
                        slowk, slowd = talib.STOCH(
                            high.values,
                            low.values,
                            data[col].values,
                            fastk_period=fastk_period,
                            slowk_period=slowk_period,
                            slowk_matype=0,
                            slowd_period=slowd_period,
                            slowd_matype=0
                        )
                        
                        result[f"{prefix}{col}_slowk"] = slowk
                        result[f"{prefix}{col}_slowd"] = slowd
                        
                        self.feature_names_.extend([
                            f"{prefix}{col}_slowk",
                            f"{prefix}{col}_slowd"
                        ])
                        continue
                        
                    elif name == 'ATR':
                        timeperiod = params.get('timeperiod', 14)
                        high = data.get('high', data[col])
                        low = data.get('low', data[col])
                        
                        atr = talib.ATR(
                            high.values,
                            low.values,
                            data[col].values,
                            timeperiod=timeperiod
                        )
                        
                        result[f"{prefix}{col}_{timeperiod}"] = atr
                        self.feature_names_.append(f"{prefix}{col}_{timeperiod}")
                        continue
                        
                    else:
                        logger.warning(f"Unsupported indicator: {name}")
                        continue
                    
                    # For simple indicators, add the feature name
                    self.feature_names_.append(f"{prefix}{col}_{params.get('timeperiod', '')}".rstrip('_'))
                    
                except Exception as e:
                    logger.error(f"Error calculating {name} for {col}: {str(e)}")
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the features created by this transformer."""
        return self.feature_names_

class StatisticalFeatures(FeatureTransformer):
    """Calculate statistical features for time series data."""
    
    def __init__(self, columns: List[str] = None, lookback: int = 20):
        """
        Initialize the statistical features transformer.
        
        Args:
            columns: List of column names to calculate features for
            lookback: Number of periods to look back for rolling calculations
        """
        super().__init__(name='StatisticalFeatures')
        self.columns = columns
        self.lookback = lookback
        self.feature_names_ = []
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'StatisticalFeatures':
        """Fit the transformer (no-op for this transformer)."""
        if self.columns is None:
            self.columns = data.select_dtypes(include=[np.number]).columns.tolist()
        return super().fit(data, **kwargs)
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Calculate statistical features for the specified columns."""
        result = data.copy()
        
        for col in self.columns:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found in data")
                continue
                
            # Rolling statistics
            result[f"{col}_roll_mean_{self.lookback}"] = data[col].rolling(window=self.lookback).mean()
            result[f"{col}_roll_std_{self.lookback}"] = data[col].rolling(window=self.lookback).std()
            result[f"{col}_roll_min_{self.lookback}"] = data[col].rolling(window=self.lookback).min()
            result[f"{col}_roll_max_{self.lookback}"] = data[col].rolling(window=self.lookback).max()
            
            # Z-score (standard score)
            roll_mean = data[col].rolling(window=self.lookback).mean()
            roll_std = data[col].rolling(window=self.lookback).std()
            result[f"{col}_zscore_{self.lookback}"] = (data[col] - roll_mean) / (roll_std + 1e-10)
            
            # Rolling quantiles
            result[f"{col}_roll_q25_{self.lookback}"] = data[col].rolling(window=self.lookback).quantile(0.25)
            result[f"{col}_roll_median_{self.lookback}"] = data[col].rolling(window=self.lookback).median()
            result[f"{col}_roll_q75_{self.lookback}"] = data[col].rolling(window=self.lookback).quantile(0.75)
            
            # Add feature names
            self.feature_names_.extend([
                f"{col}_roll_mean_{self.lookback}",
                f"{col}_roll_std_{self.lookback}",
                f"{col}_roll_min_{self.lookback}",
                f"{col}_roll_max_{self.lookback}",
                f"{col}_zscore_{self.lookback}",
                f"{col}_roll_q25_{self.lookback}",
                f"{col}_roll_median_{self.lookback}",
                f"{col}_roll_q75_{self.lookback}"
            ])
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the features created by this transformer."""
        return self.feature_names_

class TimeFeatures(FeatureTransformer):
    """Extract time-based features from datetime index."""
    
    def __init__(self, include: List[str] = None):
        """
        Initialize the time features transformer.
        
        Args:
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
        """
        super().__init__(name='TimeFeatures')
        self.include = include or ['hour', 'weekday', 'month', 'is_weekend']
        self.feature_names_ = []
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Extract time features from the DataFrame index."""
        result = data.copy()
        
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be a DatetimeIndex")
        
        if 'hour' in self.include:
            result['hour'] = data.index.hour
            self.feature_names_.append('hour')
            
        if 'day' in self.include:
            result['day'] = data.index.day
            self.feature_names_.append('day')
            
        if 'weekday' in self.include:
            result['weekday'] = data.index.weekday
            self.feature_names_.append('weekday')
            
        if 'month' in self.include:
            result['month'] = data.index.month
            self.feature_names_.append('month')
            
        if 'quarter' in self.include:
            result['quarter'] = data.index.quarter
            self.feature_names_.append('quarter')
            
        if 'year' in self.include:
            result['year'] = data.index.year
            self.feature_names_.append('year')
            
        if 'is_weekend' in self.include:
            result['is_weekend'] = data.index.weekday >= 5
            self.feature_names_.append('is_weekend')
            
        if 'is_month_start' in self.include:
            result['is_month_start'] = data.index.is_month_start
            self.feature_names_.append('is_month_start')
            
        if 'is_month_end' in self.include:
            result['is_month_end'] = data.index.is_month_end
            self.feature_names_.append('is_month_end')
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the features created by this transformer."""
        return self.feature_names_
