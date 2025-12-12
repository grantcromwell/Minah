import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict, Any, Optional
from pathlib import Path
import yaml
from datetime import datetime, timedelta
import pytz

class TradingDataset(Dataset):
    """
    Dataset for loading and processing financial time series data for training and evaluation.
    """
    
    def __init__(
        self,
        data_path: str,
        seq_len: int = 60,
        prediction_horizon: int = 5,
        train: bool = True,
        val_split: float = 0.1,
        test_split: float = 0.1,
        random_seed: int = 42,
        normalize: bool = True,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to the data file or directory
            seq_len: Length of input sequences
            prediction_horizon: Number of time steps to predict ahead
            train: Whether to use training or test data
            val_split: Fraction of data to use for validation
            test_split: Fraction of data to use for testing
            random_seed: Random seed for reproducibility
            normalize: Whether to normalize the data
            feature_columns: List of column names to use as features
            target_columns: List of column names to use as targets
        """
        self.seq_len = seq_len
        self.prediction_horizon = prediction_horizon
        self.train = train
        self.normalize = normalize
        self.rng = np.random.RandomState(random_seed)
        
        # Load and preprocess data
        self.data, self.feature_columns, self.target_columns = self._load_data(
            data_path, feature_columns, target_columns
        )
        
        # Split data into train/val/test
        self._split_data(val_split, test_split)
        
        # Compute normalization parameters on training data
        if self.normalize:
            self._compute_normalization_params()
        
        # Store sequence indices
        self._create_sequences()
    
    def _load_data(
        self,
        data_path: str,
        feature_columns: Optional[List[str]] = None,
        target_columns: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, List[str], List[str]]:
        """Load and preprocess the data."""
        # Load data
        if os.path.isdir(data_path):
            # Load from directory of CSV files
            data_files = sorted([f for f in os.listdir(data_path) if f.endswith('.csv')])
            dfs = []
            for f in data_files:
                df = pd.read_csv(os.path.join(data_path, f))
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                dfs.append(df)
            data = pd.concat(dfs).sort_values('timestamp')
        else:
            # Load single file
            data = pd.read_csv(data_path)
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        
        # Set timestamp as index
        data = data.set_index('timestamp')
        
        # Handle missing values
        data = self._handle_missing_values(data)
        
        # Feature engineering
        data = self._create_features(data)
        
        # Set default feature and target columns if not provided
        if feature_columns is None:
            # Default to OHLCV columns if they exist
            default_features = ['open', 'high', 'low', 'close', 'volume']
            feature_columns = [col for col in default_features if col in data.columns]
            
            # Add technical indicators if they exist
            tech_indicators = [
                'rsi', 'macd', 'bollinger_upper', 'bollinger_lower',
                'atr', 'obv', 'vwap', 'sma_20', 'ema_50', 'ema_200'
            ]
            feature_columns.extend([col for col in tech_indicators if col in data.columns])
        
        if target_columns is None:
            # Default to predicting next period returns
            if 'returns' in data.columns:
                target_columns = ['returns']
            else:
                target_columns = ['close']
        
        # Ensure all specified columns exist
        missing_cols = set(feature_columns + target_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing columns: {missing_cols}")
        
        return data, feature_columns, target_columns
    
    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the data."""
        # Forward fill then backward fill any remaining NaNs
        data = data.ffill().bfill()
        
        # If any NaNs remain, drop those rows
        if data.isna().any().any():
            print(f"Warning: Dropping {data.isna().any(axis=1).sum()} rows with missing values")
            data = data.dropna()
        
        return data
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators and other features."""
        df = data.copy()
        
        # Calculate returns
        if 'close' in df.columns:
            df['returns'] = df['close'].pct_change()
        
        # Simple Moving Averages
        for window in [5, 10, 20, 50, 100, 200]:
            if 'close' in df.columns:
                df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
                df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
        
        # RSI
        if 'close' in df.columns:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        if 'close' in df.columns:
            exp1 = df['close'].ewm(span=12, adjust=False).mean()
            exp2 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = exp1 - exp2
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        if 'close' in df.columns:
            df['bollinger_mid'] = df['close'].rolling(window=20).mean()
            df['bollinger_std'] = df['close'].rolling(window=20).std()
            df['bollinger_upper'] = df['bollinger_mid'] + (df['bollinger_std'] * 2)
            df['bollinger_lower'] = df['bollinger_mid'] - (df['bollinger_std'] * 2)
        
        # Average True Range (ATR)
        if all(col in df.columns for col in ['high', 'low', 'close']):
            high_low = df['high'] - df['low']
            high_close = (df['high'] - df['close'].shift()).abs()
            low_close = (df['low'] - df['close'].shift()).abs()
            df['atr'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1).rolling(14).mean()
        
        # On-Balance Volume (OBV)
        if 'volume' in df.columns and 'close' in df.columns:
            df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        
        # Volume Weighted Average Price (VWAP)
        if all(col in df.columns for col in ['high', 'low', 'close', 'volume']):
            tp = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (tp * df['volume']).cumsum() / df['volume'].cumsum()
        
        # Time-based features
        if isinstance(df.index, pd.DatetimeIndex):
            df['hour'] = df.index.hour
            df['day_of_week'] = df.index.dayofweek
            df['day_of_month'] = df.index.day
            df['month'] = df.index.month
        
        return df
    
    def _split_data(self, val_split: float, test_split: float) -> None:
        """Split data into train/validation/test sets."""
        n = len(self.data)
        test_size = int(n * test_split)
        val_size = int(n * val_split)
        train_size = n - test_size - val_size
        
        # Split indices
        indices = np.arange(n)
        if self.train:
            # Use training + validation data
            self.indices = indices[:train_size + val_size]
            self.split_idx = train_size
        else:
            # Use test data
            self.indices = indices[train_size + val_size:]
        
        # Store data splits
        self.train_data = self.data.iloc[:train_size]
        self.val_data = self.data.iloc[train_size:train_size + val_size]
        self.test_data = self.data.iloc[train_size + val_size:]
    
    def _compute_normalization_params(self) -> None:
        """Compute normalization parameters (mean and std) on training data."""
        self.means = self.train_data[self.feature_columns].mean()
        self.stds = self.train_data[self.feature_columns].std()
        
        # Avoid division by zero
        self.stds = self.stds.replace(0, 1.0)
    
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize the data using precomputed mean and std."""
        if not hasattr(self, 'means') or not hasattr(self, 'stds'):
            raise RuntimeError("Normalization parameters not computed. Call _compute_normalization_params first.")
        
        return (df[self.feature_columns] - self.means) / self.stds
    
    def _create_sequences(self) -> None:
        """Create sequences of data for training."""
        # Get the appropriate data split
        data = self.train_data if self.train else self.val_data if hasattr(self, 'val_data') else self.test_data
        
        # Normalize features if needed
        if self.normalize:
            features = self._normalize(data).values
        else:
            features = data[self.feature_columns].values
        
        # Create sequences
        self.sequences = []
        n = len(features)
        
        for i in range(n - self.seq_len - self.prediction_horizon + 1):
            # Input sequence (x): [seq_len, n_features]
            x = features[i:i + self.seq_len]
            
            # Target (y): [prediction_horizon, n_targets]
            # For classification: predict direction of future returns
            if 'returns' in self.target_columns:
                future_returns = data['returns'].iloc[i + self.seq_len:i + self.seq_len + self.prediction_horizon].values
                future_direction = 1 * (future_returns > 0)  # 1 for positive, 0 for negative
                y_class = np.argmax(np.bincount(future_direction))  # Majority direction
                y_value = np.prod(1 + future_returns) - 1  # Cumulative return
            else:
                # Default to predicting next value of the first target column
                y_class = 1 * (data[self.target_columns[0]].iloc[i + self.seq_len] > data[self.target_columns[0]].iloc[i + self.seq_len - 1])
                y_value = data[self.target_columns[0]].iloc[i + self.seq_len]
            
            self.sequences.append((x, y_class, y_value))
    
    def __len__(self) -> int:
        """Return the number of sequences in the dataset."""
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence and its target."""
        x, y_class, y_value = self.sequences[idx]
        
        # Convert to PyTorch tensors
        x_tensor = torch.FloatTensor(x)
        y_class_tensor = torch.LongTensor([y_class])
        y_value_tensor = torch.FloatTensor([y_value])
        
        return x_tensor, y_class_tensor, y_value_tensor


def create_data_loaders(
    data_path: str,
    batch_size: int = 32,
    seq_len: int = 60,
    prediction_horizon: int = 5,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    **kwargs
) -> Tuple[DataLoader, Optional[DataLoader], DataLoader]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_path: Path to the data file or directory
        batch_size: Batch size for data loaders
        seq_len: Length of input sequences
        prediction_horizon: Number of time steps to predict ahead
        val_split: Fraction of data to use for validation
        test_split: Fraction of data to use for testing
        num_workers: Number of worker processes for data loading
        **kwargs: Additional arguments to pass to TradingDataset
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = TradingDataset(
        data_path=data_path,
        seq_len=seq_len,
        prediction_horizon=prediction_horizon,
        train=True,
        val_split=val_split,
        test_split=test_split,
        **kwargs
    )
    
    val_dataset = TradingDataset(
        data_path=data_path,
        seq_len=seq_len,
        prediction_horizon=prediction_horizon,
        train=False,  # This will use validation data
        val_split=val_split,
        test_split=test_split,
        **kwargs
    )
    
    test_dataset = TradingDataset(
        data_path=data_path,
        seq_len=seq_len,
        prediction_horizon=prediction_horizon,
        train=False,  # This will use test data
        val_split=val_split,
        test_split=test_split,
        **kwargs
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, test_loader
