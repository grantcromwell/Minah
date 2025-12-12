"""
Market Data Pipeline with Hybrid ML Gap-Filling and Uncertainty Estimation
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.ensemble import IsolationForest
from prophet import Prophet
import logging
import warnings
from functools import lru_cache

# Suppress Prophet's verbose output
logging.getLogger('cmdstanpy').setLevel(logging.WARNING)
logging.getLogger('prophet').setLevel(logging.WARNING)
warnings.filterwarnings('ignore')

class DataPipeline:
    """
    Hybrid data pipeline combining ML and traditional methods for gap-filling
    and uncertainty estimation in market data.
    """
    
    def __init__(self, 
                 confidence_level: float = 0.95,
                 max_gap_minutes: int = 60,
                 min_data_points: int = 100,
                 **kwargs):
        """
        Initialize the data pipeline.
        
        Args:
            confidence_level: Confidence level for uncertainty intervals (0-1)
            max_gap_minutes: Maximum gap size to fill (in minutes)
            min_data_points: Minimum data points required for ML model training
        """
        self.confidence_level = confidence_level
        self.max_gap_seconds = max_gap_minutes * 60
        self.min_data_points = min_data_points
        self.models = {}
        self.anomaly_detector = IsolationForest(
            contamination=0.01,
            random_state=42,
            n_estimators=100
        )
        self.is_fitted = False
    
    def process_data(self, 
                    data: pd.DataFrame,
                    symbol: str,
                    timestamp_col: str = 'timestamp',
                    price_col: str = 'price',
                    volume_col: str = 'volume') -> Tuple[pd.DataFrame, Dict]:
        """
        Process market data, filling gaps and estimating uncertainty.
        
        Args:
            data: DataFrame with market data
            symbol: Trading symbol
            timestamp_col: Name of timestamp column
            price_col: Name of price column
            volume_col: Name of volume column
            
        Returns:
            Tuple of (processed_data, metadata)
        """
        # Make a copy to avoid modifying the original
        df = data.copy()
        
        # Ensure timestamp is datetime and sort
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values(timestamp_col).reset_index(drop=True)
        
        # Detect and handle anomalies
        df = self._detect_anomalies(df, price_col)
        
        # Resample to regular time intervals
        resampled = self._resample_data(df, timestamp_col, price_col, volume_col)
        
        # Fill gaps using hybrid approach
        filled_data, gap_info = self._fill_gaps_hybrid(
            resampled, 
            symbol, 
            timestamp_col, 
            price_col, 
            volume_col
        )
        
        # Calculate uncertainty
        filled_data = self._estimate_uncertainty(filled_data, price_col)
        
        # Generate metadata
        metadata = self._generate_metadata(filled_data, gap_info, price_col, volume_col)
        
        return filled_data, metadata
    
    def _detect_anomalies(self, 
                         df: pd.DataFrame, 
                         price_col: str,
                         window: int = 20) -> pd.DataFrame:
        """Detect and handle price anomalies."""
        if len(df) < window * 2:
            return df
            
        # Calculate rolling statistics
        rolling = df[price_col].rolling(window=window)
        df['price_ma'] = rolling.mean()
        df['price_std'] = rolling.std()
        
        # Identify anomalies
        z_scores = np.abs((df[price_col] - df['price_ma']) / (df['price_std'] + 1e-6))
        df['is_anomaly'] = z_scores > 3.0  # 3 sigma
        
        # Replace anomalies with NaN (will be filled later)
        df.loc[df['is_anomaly'], price_col] = np.nan
        
        # Clean up
        df = df.drop(columns=['price_ma', 'price_std'])
        
        return df
    
    def _resample_data(self, 
                      df: pd.DataFrame, 
                      timestamp_col: str,
                      price_col: str,
                      volume_col: str,
                      freq: str = '1min') -> pd.DataFrame:
        """Resample data to regular time intervals."""
        # Set timestamp as index
        df = df.set_index(timestamp_col)
        
        # Resample
        resampled = pd.DataFrame()
        
        # OHLCV resampling for price data
        ohlc = df[price_col].resample(freq).ohlc()
        resampled = ohlc
        
        # Volume resampling (sum)
        if volume_col in df.columns:
            resampled[volume_col] = df[volume_col].resample(freq).sum()
        
        # Reset index to make timestamp a column again
        resampled = resampled.reset_index()
        
        return resampled
    
    def _fill_gaps_hybrid(self,
                         df: pd.DataFrame,
                         symbol: str,
                         timestamp_col: str,
                         price_col: str,
                         volume_col: str) -> Tuple[pd.DataFrame, Dict]:
        """Fill gaps using a hybrid approach (ML + traditional)."""
        gap_info = {
            'total_gaps': 0,
            'ml_filled': 0,
            'linear_filled': 0,
            'ffill_filled': 0,
            'gap_sizes': []
        }
        
        if df.empty:
            return df, gap_info
        
        # Ensure timestamp is the index
        df = df.set_index(timestamp_col)
        
        # Find gaps
        df['time_diff'] = df.index.to_series().diff().dt.total_seconds()
        gap_indices = df[df['time_diff'] > 60].index  # Gaps > 1 minute
        gap_info['total_gaps'] = len(gap_indices)
        
        # Process each gap
        for i, gap_start in enumerate(gap_indices):
            gap_end = df.index[df.index.get_loc(gap_start) + 1] if i < len(gap_indices) - 1 else df.index[-1]
            gap_size = (gap_end - gap_start).total_seconds()
            gap_info['gap_sizes'].append(gap_size)
            
            # Skip if gap is too large
            if gap_size > self.max_gap_seconds:
                continue
                
            # Get data before the gap
            pre_gap_data = df.loc[:gap_start].iloc[-self.min_data_points:]
            
            # Choose filling method based on data availability
            if len(pre_gap_data) >= self.min_data_points:
                # Use ML model for filling
                filled = self._fill_with_ml(
                    pre_gap_data, 
                    gap_start, 
                    gap_end, 
                    price_col,
                    symbol
                )
                if filled is not None:
                    df.loc[gap_start:gap_end, price_col] = filled[price_col]
                    gap_info['ml_filled'] += 1
                    continue
            
            # Fall back to linear interpolation
            try:
                df[price_col] = df[price_col].interpolate(method='linear')
                gap_info['linear_filled'] += 1
            except:
                # Final fallback to forward fill
                df[price_col] = df[price_col].ffill()
                gap_info['ffill_filled'] += 1
        
        # Fill any remaining NaNs with forward/backward fill
        df[price_col] = df[price_col].ffill().bfill()
        
        # Reset index
        df = df.reset_index()
        
        return df, gap_info
    
    @lru_cache(maxsize=10)
    def _get_prophet_model(self, symbol: str):
        """Get or create a Prophet model for the given symbol."""
        if symbol not in self.models:
            self.models[symbol] = Prophet(
                yearly_seasonality=False,
                weekly_seasonality=True,
                daily_seasonality=True,
                seasonality_mode='multiplicative',
                interval_width=self.confidence_level
            )
        return self.models[symbol]
    
    def _fill_with_ml(self,
                     pre_gap_data: pd.DataFrame,
                     gap_start: pd.Timestamp,
                     gap_end: pd.Timestamp,
                     price_col: str,
                     symbol: str) -> Optional[pd.DataFrame]:
        """Fill gap using ML model (Prophet)."""
        try:
            # Prepare data for Prophet
            prophet_df = pd.DataFrame({
                'ds': pre_gap_data.index,
                'y': pre_gap_data[price_col]
            })
            
            # Get or create model
            model = self._get_prophet_model(symbol)
            
            # Fit model
            model.fit(prophet_df)
            self.is_fitted = True
            
            # Create future dataframe for prediction
            future_dates = pd.date_range(
                start=gap_start,
                end=gap_end,
                freq='1min'
            )
            future = pd.DataFrame({'ds': future_dates})
            
            # Make prediction
            forecast = model.predict(future)
            
            # Create result dataframe
            result = pd.DataFrame({
                price_col: forecast['yhat'].values,
                f'{price_col}_lower': forecast['yhat_lower'].values,
                f'{price_col}_upper': forecast['yhat_upper'].values
            }, index=future_dates)
            
            return result
            
        except Exception as e:
            print(f"ML gap-filling failed: {str(e)}")
            return None
    
    def _estimate_uncertainty(self, 
                             df: pd.DataFrame, 
                             price_col: str) -> pd.DataFrame:
        """Estimate uncertainty for filled data points."""
        if f'{price_col}_upper' not in df.columns or f'{price_col}_lower' not in df.columns:
            # If we don't have ML-based uncertainty, use rolling std
            window = min(20, len(df) // 10)  # Use 10% of data or 20 points, whichever is smaller
            if window > 1:
                rolling_std = df[price_col].rolling(window=window).std()
                df[f'{price_col}_std'] = rolling_std.ffill().bfill()
                df[f'{price_col}_upper'] = df[price_col] + 2 * df[f'{price_col}_std']
                df[f'{price_col}_lower'] = df[price_col] - 2 * df[f'{price_col}_std']
        
        return df
    
    def _generate_metadata(self,
                          df: pd.DataFrame,
                          gap_info: Dict,
                          price_col: str,
                          volume_col: str) -> Dict:
        """Generate metadata about the data processing."""
        metadata = {
            'symbol': 'unknown',
            'start_time': df.index[0] if not df.empty else None,
            'end_time': df.index[-1] if not df.empty else None,
            'num_points': len(df),
            'price_stats': {},
            'gap_info': gap_info,
            'data_quality': {}
        }
        
        if not df.empty:
            # Price statistics
            metadata['price_stats'] = {
                'mean': df[price_col].mean(),
                'std': df[price_col].std(),
                'min': df[price_col].min(),
                'max': df[price_col].max(),
                'median': df[price_col].median()
            }
            
            # Data quality metrics
            metadata['data_quality'] = {
                'missing_values': df[price_col].isna().sum(),
                'zero_volume': (df[volume_col] == 0).sum() if volume_col in df.columns else 0,
                'duplicate_timestamps': df.index.duplicated().sum()
            }
            
            # Add gap statistics
            if gap_info['gap_sizes']:
                gap_sizes = gap_info['gap_sizes']
                metadata['gap_info'].update({
                    'avg_gap_size': np.mean(gap_sizes) if gap_sizes else 0,
                    'max_gap_size': max(gap_sizes) if gap_sizes else 0,
                    'min_gap_size': min(gap_sizes) if gap_sizes else 0
                })
        
        return metadata
