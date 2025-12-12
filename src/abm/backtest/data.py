""
Data handling for backtesting.

This module provides functionality for loading, processing, and serving
historical market data during backtesting.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Iterator, Tuple, Any
from datetime import datetime, timedelta
import logging
import os

logger = logging.getLogger(__name__)

class DataError(Exception):
    """Exception raised for errors in the data handling process."""
    pass

class HistoricalDataHandler:
    """
    Handles loading and serving historical market data for backtesting.
    
    This class is responsible for:
    - Loading historical market data from various sources
    - Aligning and normalizing timestamps
    - Serving data in a consistent format to the backtest engine
    """
    
    def __init__(
        self,
        data_source: str = 'local',
        data_dir: Optional[str] = None,
        symbols: Optional[List[str]] = None,
        data_frequency: str = '1min',
        adjust_prices: bool = True,
        **kwargs
    ):
        """
        Initialize the historical data handler.
        
        Args:
            data_source: Source of the historical data ('local', 'yfinance', 'alpaca', etc.)
            data_dir: Directory containing local data files (required for 'local' data source)
            symbols: List of symbols to load data for
            data_frequency: Frequency of the data ('1min', '5min', '1h', '1d', etc.)
            adjust_prices: Whether to adjust prices for splits and dividends
            **kwargs: Additional parameters specific to the data source
        """
        self.data_source = data_source
        self.data_dir = data_dir or os.path.join(os.path.dirname(__file__), '..', '..', 'data')
        self.symbols = symbols or []
        self.data_frequency = data_frequency
        self.adjust_prices = adjust_prices
        self.kwargs = kwargs
        
        # Data storage
        self.data = {}
        self.current_idx = 0
        self.start_date = None
        self.end_date = None
        
        # Initialize data loading
        self._load_data()
    
    def _load_data(self) -> None:
        """Load historical data based on the specified data source."""
        if not self.symbols:
            raise DataError("No symbols provided for data loading")
        
        logger.info(f"Loading {self.data_frequency} data for symbols: {', '.join(self.symbols)}")
        
        if self.data_source == 'local':
            self._load_local_data()
        elif self.data_source == 'yfinance':
            self._load_yfinance_data()
        elif self.data_source == 'alpaca':
            self._load_alpaca_data()
        else:
            raise DataError(f"Unsupported data source: {self.data_source}")
        
        # Validate and align data
        self._validate_and_align_data()
    
    def _load_local_data(self) -> None:
        """Load data from local CSV or Parquet files."""
        import glob
        
        for symbol in self.symbols:
            # Look for files matching the pattern: {symbol}_{frequency}.*
            pattern = os.path.join(self.data_dir, f"{symbol.lower()}_{self.data_frequency}.*")
            files = glob.glob(pattern)
            
            if not files:
                logger.warning(f"No data files found for {symbol} with pattern: {pattern}")
                continue
                
            # Use the first matching file
            file_path = files[0]
            
            try:
                if file_path.endswith('.parquet'):
                    df = pd.read_parquet(file_path)
                elif file_path.endswith('.csv'):
                    df = pd.read_csv(file_path, parse_dates=['timestamp'], index_col='timestamp')
                elif file_path.endswith('.h5') or file_path.endswith('.hdf5'):
                    df = pd.read_hdf(file_path, key='data')
                else:
                    logger.warning(f"Unsupported file format: {file_path}")
                    continue
                
                # Ensure timestamp is the index
                if not isinstance(df.index, pd.DatetimeIndex):
                    if 'timestamp' in df.columns:
                        df = df.set_index('timestamp')
                    elif 'date' in df.columns:
                        df = df.set_index('date')
                    elif 'time' in df.columns:
                        df = df.set_index('time')
                    else:
                        raise DataError(f"Could not determine timestamp column in {file_path}")
                
                # Ensure the index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # Store the data
                self.data[symbol] = df
                logger.info(f"Loaded {len(df)} rows for {symbol} from {os.path.basename(file_path)}")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol} from {file_path}: {str(e)}")
    
    def _load_yfinance_data(self) -> None:
        """Load data using yfinance library."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required for Yahoo Finance data. Install with: pip install yfinance")
        
        # Map frequency to yfinance interval
        interval_map = {
            '1min': '1m', '5min': '5m', '15min': '15m', '30min': '30m',
            '1h': '1h', '1d': '1d', '1wk': '1wk', '1mo': '1mo'
        }
        
        interval = interval_map.get(self.data_frequency, '1d')
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                
                # Get historical data
                df = ticker.history(
                    period="max" if not self.kwargs.get('period') else self.kwargs['period'],
                    interval=interval,
                    auto_adjust=self.adjust_prices,
                    prepost=self.kwargs.get('prepost', False),
                    threads=True,
                    proxy=self.kwargs.get('proxy')
                )
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Rename columns to match our standard format
                df = df.rename(columns={
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Dividends': 'dividend',
                    'Stock Splits': 'split'
                })
                
                # Ensure the index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # Store the data
                self.data[symbol] = df
                logger.info(f"Loaded {len(df)} rows for {symbol} from Yahoo Finance")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol} from Yahoo Finance: {str(e)}")
    
    def _load_alpaca_data(self) -> None:
        """Load data from Alpaca API."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame
        except ImportError:
            raise ImportError("alpaca-py is required for Alpaca data. Install with: pip install alpaca-py")
        
        # Initialize Alpaca client
        api_key = self.kwargs.get('api_key')
        secret_key = self.kwargs.get('secret_key')
        
        if not api_key or not secret_key:
            raise DataError("API key and secret key are required for Alpaca data source")
        
        client = StockHistoricalDataClient(api_key, secret_key)
        
        # Map frequency to Alpaca TimeFrame
        timeframe_map = {
            '1min': TimeFrame.Minute,
            '5min': TimeFrame.FiveMinutes,
            '15min': TimeFrame.FifteenMinutes,
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day
        }
        
        timeframe = timeframe_map.get(self.data_frequency, TimeFrame.Day)
        
        # Determine date range
        end_date = self.kwargs.get('end_date', pd.Timestamp.now(tz='UTC'))
        start_date = self.kwargs.get('start_date', end_date - pd.Timedelta(days=365))
        
        for symbol in self.symbols:
            try:
                # Request data from Alpaca
                request_params = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=timeframe,
                    start=start_date,
                    end=end_date,
                    adjustment='all' if self.adjust_prices else 'raw'
                )
                
                bars = client.get_stock_bars(request_params)
                
                # Convert to DataFrame
                df = bars.df
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                
                # Ensure the index is timezone-aware
                if df.index.tz is None:
                    df.index = df.index.tz_localize('UTC')
                
                # Store the data
                self.data[symbol] = df
                logger.info(f"Loaded {len(df)} rows for {symbol} from Alpaca")
                
            except Exception as e:
                logger.error(f"Error loading data for {symbol} from Alpaca: {str(e)}")
    
    def _validate_and_align_data(self) -> None:
        """Validate and align data across all symbols."""
        if not self.data:
            raise DataError("No data loaded for any symbols")
        
        # Find common index across all symbols
        common_index = None
        for symbol, df in self.data.items():
            if common_index is None:
                common_index = df.index
            else:
                common_index = common_index.intersection(df.index)
        
        if common_index.empty:
            raise DataError("No common timestamps found across all symbols")
        
        # Trim data to common index
        for symbol in list(self.data.keys()):
            self.data[symbol] = self.data[symbol].loc[common_index].sort_index()
            logger.info(f"Aligned {symbol} to {len(common_index)} timestamps")
        
        # Set date range
        self.start_date = common_index.min()
        self.end_date = common_index.max()
    
    def set_date_range(
        self,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None
    ) -> None:
        """
        Set the date range for the backtest.
        
        Args:
            start_date: Start date of the backtest
            end_date: End date of the backtest
        """
        if start_date is not None:
            self.start_date = pd.to_datetime(start_date).tz_localize('UTC')
        if end_date is not None:
            self.end_date = pd.to_datetime(end_date).tz_localize('UTC')
        
        logger.info(f"Date range set to: {self.start_date} - {self.end_date}")
    
    def get_symbols(self) -> List[str]:
        """Get the list of symbols with available data."""
        return list(self.data.keys())
    
    def get_data(self, symbol: str) -> pd.DataFrame:
        """
        Get the full historical data for a symbol.
        
        Args:
            symbol: Symbol to get data for
            
        Returns:
            DataFrame containing the historical data
        """
        return self.data.get(symbol, pd.DataFrame())
    
    def get_latest_data(self, symbol: str, n: int = 1) -> pd.DataFrame:
        """
        Get the latest n data points for a symbol.
        
        Args:
            symbol: Symbol to get data for
            n: Number of data points to return
            
        Returns:
            DataFrame containing the latest n data points
        """
        if symbol not in self.data:
            return pd.DataFrame()
        
        return self.data[symbol].iloc[-n:]
    
    def get_data_range(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """
        Get data for a symbol within a specific date range.
        
        Args:
            symbol: Symbol to get data for
            start: Start date
            end: End date
            
        Returns:
            DataFrame containing the data within the specified range
        """
        if symbol not in self.data:
            return pd.DataFrame()
        
        return self.data[symbol].loc[start:end]
    
    def __iter__(self) -> Iterator[Tuple[datetime, Dict[str, pd.Series]]]:
        """
        Iterate through the data one timestamp at a time.
        
        Yields:
            Tuple of (timestamp, data_dict) where data_dict maps symbols to their data at that timestamp
        """
        if not self.data:
            raise DataError("No data available to iterate over")
        
        # Get the first symbol's index (all symbols should have the same index after alignment)
        first_symbol = next(iter(self.data))
        timestamps = self.data[first_symbol].index
        
        # Filter by date range if specified
        if self.start_date is not None:
            timestamps = timestamps[timestamps >= self.start_date]
        if self.end_date is not None:
            timestamps = timestamps[timestamps <= self.end_date]
        
        # Iterate through each timestamp
        for timestamp in timestamps:
            data_slice = {}
            
            for symbol, df in self.data.items():
                if timestamp in df.index:
                    data_slice[symbol] = df.loc[timestamp]
            
            if data_slice:  # Only yield if we have data for at least one symbol
                yield timestamp, data_slice
    
    def get_slice(self, timestamp: datetime) -> Dict[str, pd.Series]:
        """
        Get data for all symbols at a specific timestamp.
        
        Args:
            timestamp: Timestamp to get data for
            
        Returns:
            Dictionary mapping symbols to their data at the specified timestamp
        """
        slice_data = {}
        
        for symbol, df in self.data.items():
            if timestamp in df.index:
                slice_data[symbol] = df.loc[timestamp]
        
        return slice_data
    
    def get_available_data_range(self) -> Tuple[datetime, datetime]:
        """
        Get the available date range for the loaded data.
        
        Returns:
            Tuple of (start_date, end_date)
        """
        if not self.data:
            raise DataError("No data available")
        
        return self.start_date, self.end_date
