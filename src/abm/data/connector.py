""
Market Data Connector for fetching and preprocessing market data from various sources.
"""
import pandas as pd
import numpy as np
import ccxt
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import pytz
import os
import json
from .pipeline import DataPipeline

logger = logging.getLogger(__name__)

class MarketDataConnector:
    """
    Connector for fetching and managing market data from various sources.
    Supports both live and historical data with caching.
    """
    
    def __init__(self, 
                 exchange_id: str = 'binance',
                 data_dir: str = 'market_data',
                 cache_enabled: bool = True,
                 **kwargs):
        """
        Initialize the market data connector.
        
        Args:
            exchange_id: Exchange ID (e.g., 'binance', 'coinbasepro')
            data_dir: Directory to store cached data
            cache_enabled: Whether to enable data caching
        """
        self.exchange_id = exchange_id
        self.data_dir = data_dir
        self.cache_enabled = cache_enabled
        self.exchange = self._init_exchange(exchange_id, **kwargs)
        self.pipeline = DataPipeline(**kwargs)
        self.supported_intervals = {
            '1m': '1m', '5m': '5m', '15m': '15m',
            '1h': '1h', '4h': '4h', '1d': '1d'
        }
        
        # Create data directory if it doesn't exist
        os.makedirs(self.data_dir, exist_ok=True)
    
    def _init_exchange(self, exchange_id: str, **kwargs) -> ccxt.Exchange:
        """Initialize the exchange connection."""
        try:
            # Get exchange class from ccxt
            exchange_class = getattr(ccxt, exchange_id)
            
            # Initialize with rate limiting enabled
            exchange = exchange_class({
                'enableRateLimit': True,
                'options': {
                    'adjustForTimeDifference': True,
                    'recvWindow': 60000,  # 60 seconds
                },
                **kwargs
            })
            
            # Test connectivity
            exchange.fetch_time()
            logger.info(f"Successfully connected to {exchange_id}")
            
            return exchange
            
        except Exception as e:
            logger.error(f"Failed to initialize {exchange_id}: {str(e)}")
            raise
    
    def fetch_ohlcv(self, 
                   symbol: str, 
                   timeframe: str = '1h',
                   since: Optional[int] = None,
                   limit: int = 1000,
                   use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            since: Timestamp in milliseconds since epoch
            limit: Maximum number of candles to fetch
            use_cache: Whether to use cached data if available
            
        Returns:
            DataFrame with OHLCV data
        """
        # Normalize symbol and timeframe
        symbol = symbol.upper().replace('-', '/')
        timeframe = self.supported_intervals.get(timeframe, '1h')
        
        # Generate cache key
        cache_key = f"{self.exchange_id}_{symbol.replace('/', '')}_{timeframe}"
        cache_file = os.path.join(self.data_dir, f"{cache_key}.parquet")
        
        # Try to load from cache first
        if use_cache and self.cache_enabled and os.path.exists(cache_file):
            try:
                df = pd.read_parquet(cache_file)
                if since is not None and not df.empty:
                    df = df[df.index >= pd.to_datetime(since, unit='ms')]
                if len(df) >= limit:
                    return df.tail(limit)
                # If we don't have enough data, we'll fetch more below
            except Exception as e:
                logger.warning(f"Error reading cache file {cache_file}: {str(e)}"
        
        # Calculate time range
        now = int(time.time() * 1000)
        if since is None:
            # Default to 1 week of hourly data
            since = now - (7 * 24 * 60 * 60 * 1000)
        
        # Fetch data from exchange
        logger.info(f"Fetching {limit} {timeframe} candles for {symbol} since {pd.to_datetime(since, unit='ms')}")
        
        all_ohlcv = []
        current_since = since
        
        while len(all_ohlcv) < limit:
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=current_since,
                    limit=min(1000, limit - len(all_ohlcv))
                )
                
                if not ohlcv:
                    break
                    
                all_ohlcv.extend(ohlcv)
                
                # Update since for next request
                current_since = ohlcv[-1][0] + 1
                
                # Check if we've reached the current time
                if current_since > now:
                    break
                    
                # Respect rate limits
                time.sleep(self.exchange.rateLimit / 1000)
                
            except Exception as e:
                logger.error(f"Error fetching OHLCV data: {str(e)}")
                break
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        if df.empty:
            return pd.DataFrame()
        
        # Convert timestamp to datetime and set as index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Convert string values to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Cache the data
        if self.cache_enabled:
            try:
                # Read existing cache if it exists
                if os.path.exists(cache_file):
                    existing_df = pd.read_parquet(cache_file)
                    # Combine with new data, keeping the most recent values
                    df = pd.concat([existing_df, df]).drop_duplicates(keep='last')
                # Save to cache
                df.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to cache data: {str(e)}")
        
        return df.tail(limit)
    
    def fetch_order_book(self, 
                        symbol: str, 
                        limit: int = 100) -> Dict[str, Any]:
        """
        Fetch order book data.
        
        Args:
            symbol: Trading pair symbol
            limit: Number of order book levels to fetch
            
        Returns:
            Dictionary with order book data
        """
        symbol = symbol.upper().replace('-', '/')
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit=limit)
            return {
                'bids': orderbook['bids'],
                'asks': orderbook['asks'],
                'timestamp': orderbook['timestamp'],
                'datetime': orderbook['datetime']
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {str(e)}")
            return {'bids': [], 'asks': [], 'timestamp': None, 'datetime': None}
    
    def fetch_trades(self, 
                    symbol: str, 
                    since: Optional[int] = None,
                    limit: int = 1000) -> pd.DataFrame:
        """
        Fetch recent trades.
        
        Args:
            symbol: Trading pair symbol
            since: Timestamp in milliseconds since epoch
            limit: Maximum number of trades to fetch
            
        Returns:
            DataFrame with trade data
        """
        symbol = symbol.upper().replace('-', '/')
        try:
            trades = self.exchange.fetch_trades(symbol, since=since, limit=limit)
            
            if not trades:
                return pd.DataFrame()
                
            df = pd.DataFrame(trades)
            
            # Convert timestamp to datetime
            if 'timestamp' in df.columns:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def get_historical_data(self,
                          symbol: str,
                          start_date: str,
                          end_date: Optional[str] = None,
                          timeframe: str = '1h') -> Tuple[pd.DataFrame, Dict]:
        """
        Get historical price data with gap filling and uncertainty estimation.
        
        Args:
            symbol: Trading pair symbol
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format (default: current date)
            timeframe: Timeframe string (e.g., '1m', '1h', '1d')
            
        Returns:
            Tuple of (processed_data, metadata)
        """
        # Convert dates to timestamps
        start_dt = pd.to_datetime(start_date)
        start_ts = int(start_dt.timestamp() * 1000)
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = pd.Timestamp.now()
        end_ts = int(end_dt.timestamp() * 1000)
        
        # Fetch raw data
        df = self.fetch_ohlcv(
            symbol=symbol,
            timeframe=timeframe,
            since=start_ts,
            limit=10000  # Adjust based on expected data size
        )
        
        if df.empty:
            return pd.DataFrame(), {'error': 'No data available'}
        
        # Process data with the pipeline
        processed_data, metadata = self.pipeline.process_data(
            data=df.reset_index(),
            symbol=symbol,
            timestamp_col='datetime',
            price_col='close',
            volume_col='volume'
        )
        
        return processed_data, metadata
    
    def get_market_snapshot(self, 
                          symbols: List[str],
                          orderbook_depth: int = 10) -> Dict[str, Any]:
        """
        Get a snapshot of current market conditions.
        
        Args:
            symbols: List of trading pair symbols
            orderbook_depth: Depth of order book to fetch
            
        Returns:
            Dictionary with market snapshot
        """
        snapshot = {
            'timestamp': int(time.time() * 1000),
            'datetime': datetime.utcnow().isoformat(),
            'markets': {}
        }
        
        for symbol in symbols:
            try:
                # Fetch ticker data
                ticker = self.exchange.fetch_ticker(symbol)
                
                # Fetch order book
                orderbook = self.fetch_order_book(symbol, limit=orderbook_depth)
                
                # Add to snapshot
                snapshot['markets'][symbol] = {
                    'price': {
                        'last': ticker['last'],
                        'bid': ticker['bid'],
                        'ask': ticker['ask'],
                        'open': ticker['open'],
                        'high': ticker['high'],
                        'low': ticker['low'],
                        'close': ticker['close'],
                        'change': ticker['percentage'],
                        'average': ticker['average']
                    },
                    'volume': ticker['baseVolume'],
                    'orderbook': orderbook
                }
                
            except Exception as e:
                logger.error(f"Error fetching data for {symbol}: {str(e)}")
                snapshot['markets'][symbol] = {'error': str(e)}
        
        return snapshot
    
    def close(self):
        """Close the exchange connection and cleanup."""
        if hasattr(self, 'exchange') and hasattr(self.exchange, 'close'):
            self.exchange.close()
            logger.info("Closed exchange connection")
