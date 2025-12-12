"""
Regime Detection Module for the Agent-Based Modeling system.

This module implements machine learning-based regime detection using both
supervised and unsupervised learning techniques.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
import pytz

# ML imports
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras import layers, models

# Type aliases
ArrayLike = Union[np.ndarray, pd.DataFrame, List[float]]

class RegimeType(Enum):
    """Market regime types."""
    TRENDING_UP = auto()
    TRENDING_DOWN = auto()
    MEAN_REVERTING = auto()
    VOLATILE = auto()
    LOW_VOLATILITY = auto()
    CRASH = auto()
    RALLY = auto()
    SIDEWAYS = auto()
    
    @classmethod
    def from_string(cls, regime_str: str) -> 'RegimeType':
        """Convert string to RegimeType enum."""
        try:
            return cls[regime_str.upper()]
        except KeyError:
            raise ValueError(f"Unknown regime type: {regime_str}")

@dataclass
class Regime:
    """Market regime with confidence and features."""
    regime_type: RegimeType
    confidence: float  # 0.0 to 1.0
    start_time: datetime
    end_time: Optional[datetime] = None
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'regime': self.regime_type.name,
            'confidence': self.confidence,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'features': self.features
        }

class RegimeDetector:
    """
    Machine learning-based market regime detector.
    
    This class implements various algorithms for detecting market regimes:
    - Gaussian Mixture Models (GMM) for unsupervised regime detection
    - Random Forest for supervised regime classification
    - Deep learning with LSTM for sequential regime prediction
    - Isolation Forest for anomaly detection
    """
    
    def __init__(self, 
                 n_regimes: int = 3,
                 lookback_window: int = 20,
                 method: str = 'gmm',
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes: Number of regimes to detect (for unsupervised methods)
            lookback_window: Number of periods to use for feature calculation
            method: Detection method ('gmm', 'rf', 'lstm', 'isolation')
            config: Configuration dictionary
        """
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.method = method
        self.config = config or {}
        self.scaler = StandardScaler()
        self.regime_history: List[Regime] = []
        self.current_regime: Optional[Regime] = None
        self._initialize_models()
        
        # Initialize metrics
        self.metrics = {
            'total_predictions': 0,
            'regime_changes': 0,
            'avg_confidence': 0.0,
            'last_update': None,
            'start_time': datetime.now(pytz.utc)
        }
    
    def _initialize_models(self) -> None:
        """Initialize the machine learning models."""
        # Common parameters
        random_state = self.config.get('random_state', 42)
        
        # Initialize the appropriate model based on method
        if self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                random_state=random_state,
                covariance_type='full',
                max_iter=1000,
                n_init=3
            )
            self.regime_names = [f'Regime {i+1}' for i in range(self.n_regimes)]
            
        elif self.method == 'rf':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=5,
                random_state=random_state,
                class_weight='balanced'
            )
            # For supervised learning, we need labeled data
            self.regime_names = [r.name for r in RegimeType]
            
        elif self.method == 'lstm':
            # Will be built in _build_lstm_model()
            self.model = self._build_lstm_model()
            self.regime_names = [r.name for r in RegimeType]
            
        elif self.method == 'isolation':
            self.model = IsolationForest(
                n_estimators=100,
                contamination=0.1,
                random_state=random_state
            )
            self.regime_names = ['Normal', 'Anomaly']
            
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _build_lstm_model(self) -> tf.keras.Model:
        """Build an LSTM model for regime classification."""
        input_shape = (self.lookback_window, 5)  # Example: 5 features
        
        model = models.Sequential([
            layers.LSTM(64, return_sequences=True, input_shape=input_shape),
            layers.Dropout(0.2),
            layers.LSTM(32, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(len(RegimeType), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def extract_features(self, 
                        prices: ArrayLike,
                        volumes: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Extract features for regime detection.
        
        Args:
            prices: Array-like of price data
            volumes: Optional array-like of volume data
            
        Returns:
            Feature matrix (n_samples, n_features)
        """
        if isinstance(prices, pd.Series) or isinstance(prices, pd.DataFrame):
            prices = prices.values
            
        if volumes is not None and (isinstance(volumes, pd.Series) or isinstance(volumes, pd.DataFrame)):
            volumes = volumes.values
        
        # Calculate returns and other features
        returns = np.diff(prices) / prices[:-1]
        log_returns = np.log(prices[1:] / prices[:-1])
        
        # Initialize feature matrix
        n_samples = len(returns)
        features = []
        
        # 1. Basic statistics
        features.append(returns)  # Raw returns
        
        # 2. Volatility (rolling standard deviation of returns)
        if n_samples >= self.lookback_window:
            rolling_vol = pd.Series(returns).rolling(window=self.lookback_window).std().values
            features.append(rolling_vol)
        else:
            features.append(np.full_like(returns, np.nan))
        
        # 3. Trend (rolling mean of returns)
        if n_samples >= self.lookback_window:
            rolling_mean = pd.Series(returns).rolling(window=self.lookback_window).mean().values
            features.append(rolling_mean)
        else:
            features.append(np.full_like(returns, np.nan))
        
        # 4. Volume (if available)
        if volumes is not None and len(volumes) == len(prices):
            volume_changes = np.diff(volumes) / (volumes[:-1] + 1e-10)  # Avoid division by zero
            features.append(volume_changes)
            
            # Volume-weighted price change
            if n_samples >= self.lookback_window:
                vwap = pd.Series(prices * volumes).rolling(window=self.lookback_window).sum() / \
                       pd.Series(volumes).rolling(window=self.lookback_window).sum()
                vwap_returns = vwap.pct_change().values
                features.append(vwap_returns)
            else:
                features.append(np.full_like(returns, np.nan))
        
        # 5. Technical indicators (simplified)
        if n_samples >= 14:  # Minimum for RSI
            # RSI (Relative Strength Index)
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs)).values
            features.append(rsi[-len(returns):])
        else:
            features.append(np.full_like(returns, np.nan))
        
        # Stack features and handle NaNs
        feature_matrix = np.column_stack(features)
        valid_mask = ~np.isnan(feature_matrix).any(axis=1)
        
        return feature_matrix[valid_mask]
    
    def fit(self, 
            prices: ArrayLike,
            volumes: Optional[ArrayLike] = None,
            y: Optional[ArrayLike] = None) -> 'RegimeDetector':
        """
        Fit the regime detection model.
        
        Args:
            prices: Array-like of price data
            volumes: Optional array-like of volume data
            y: Optional array-like of target labels (for supervised learning)
            
        Returns:
            self
        """
        # Extract features
        X = self.extract_features(prices, volumes)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the appropriate model
        if self.method in ['gmm', 'isolation']:
            self.model.fit(X_scaled)
            
        elif self.method == 'rf' and y is not None:
            # For supervised learning, we need labels
            if len(y) != len(X):
                raise ValueError("Length of y must match the number of samples in X")
            self.model.fit(X_scaled, y)
            
        elif self.method == 'lstm' and y is not None:
            # Reshape data for LSTM (samples, time steps, features)
            X_reshaped = self._create_sequences(X_scaled, self.lookback_window)
            y_reshaped = y[self.lookback_window-1:]  # Align with sequences
            
            # Convert to one-hot encoding for categorical crossentropy
            y_one_hot = tf.keras.utils.to_categorical(y_reshaped, num_classes=len(RegimeType))
            
            # Train the model
            self.model.fit(
                X_reshaped, 
                y_one_hot,
                epochs=self.config.get('epochs', 50),
                batch_size=32,
                validation_split=0.2,
                verbose=0
            )
        
        return self
    
    def predict(self, 
                prices: ArrayLike,
                volumes: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Predict market regimes.
        
        Args:
            prices: Array-like of price data
            volumes: Optional array-like of volume data
            
        Returns:
            Array of predicted regime indices
        """
        # Extract features
        X = self.extract_features(prices, volumes)
        
        if len(X) == 0:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        if self.method == 'gmm':
            # For GMM, we get probabilities and take the most likely regime
            probs = self.model.predict_proba(X_scaled)
            predictions = np.argmax(probs, axis=1)
            
        elif self.method == 'rf':
            # For Random Forest, we get class predictions
            predictions = self.model.predict(X_scaled)
            
        elif self.method == 'lstm':
            # For LSTM, we need to create sequences
            X_reshaped = self._create_sequences(X_scaled, self.lookback_window)
            if len(X_reshaped) == 0:
                return np.array([])
                
            # Get predictions
            probs = self.model.predict(X_reshaped, verbose=0)
            predictions = np.argmax(probs, axis=1)
            
            # Pad the beginning with the first prediction
            predictions = np.concatenate([
                np.full(self.lookback_window - 1, predictions[0]),
                predictions
            ])[:len(X_scaled)]  # Ensure correct length
            
        elif self.method == 'isolation':
            # For Isolation Forest, we get anomaly scores (-1 for anomalies, 1 for normal)
            predictions = self.model.predict(X_scaled)
            # Convert to 0 (normal) and 1 (anomaly)
            predictions = (predictions == -1).astype(int)
        
        # Update metrics
        self.metrics['total_predictions'] += len(predictions)
        
        return predictions
    
    def predict_proba(self, 
                     prices: ArrayLike,
                     volumes: Optional[ArrayLike] = None) -> np.ndarray:
        """
        Predict regime probabilities.
        
        Args:
            prices: Array-like of price data
            volumes: Optional array-like of volume data
            
        Returns:
            Array of predicted regime probabilities (n_samples, n_regimes)
        """
        # Extract features
        X = self.extract_features(prices, volumes)
        
        if len(X) == 0:
            return np.array([])
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get probabilities
        if self.method == 'gmm':
            return self.model.predict_proba(X_scaled)
            
        elif self.method == 'rf':
            return self.model.predict_proba(X_scaled)
            
        elif self.method == 'lstm':
            X_reshaped = self._create_sequences(X_scaled, self.lookback_window)
            if len(X_reshaped) == 0:
                return np.array([])
                
            probs = self.model.predict(X_reshaped, verbose=0)
            
            # Pad the beginning with the first prediction
            probs_padded = np.zeros((len(X_scaled), probs.shape[1]))
            probs_padded[self.lookback_window-1:] = probs
            probs_padded[:self.lookback_window-1] = probs[0]
            
            return probs_padded
            
        elif self.method == 'isolation':
            # For Isolation Forest, we can use decision function as a proxy for probability
            decision = self.model.decision_function(X_scaled)
            # Convert to probability-like scores between 0 and 1
            probs = 1 / (1 + np.exp(-decision))  # Sigmoid
            return np.column_stack([1 - probs, probs])  # [P(normal), P(anomaly)]
    
    def detect_regime_change(self,
                           prices: ArrayLike,
                           volumes: Optional[ArrayLike] = None,
                           threshold: float = 0.7) -> Optional[Regime]:
        """
        Detect if the market regime has changed.
        
        Args:
            prices: Array-like of price data
            volumes: Optional array-like of volume data
            threshold: Confidence threshold for regime change
            
        Returns:
            New Regime if a change is detected, None otherwise
        """
        # Get probabilities for the most recent period
        probs = self.predict_proba(prices, volumes)
        
        if len(probs) == 0:
            return None
            
        # Get the most likely regime and its probability
        current_probs = probs[-1]  # Most recent probabilities
        predicted_regime = np.argmax(current_probs)
        confidence = current_probs[predicted_regime]
        
        # Check if we have a previous regime to compare with
        if self.current_regime is None:
            # First detection
            self.current_regime = Regime(
                regime_type=RegimeType(predicted_regime + 1),  # +1 because RegimeType starts at 1
                confidence=float(confidence),
                start_time=datetime.now(pytz.utc),
                features=self._get_current_features(prices, volumes)
            )
            self.regime_history.append(self.current_regime)
            self.metrics['regime_changes'] += 1
            return self.current_regime
            
        # Check if the regime has changed
        current_regime_idx = self.current_regime.regime_type.value - 1  # Convert to 0-based index
        
        if predicted_regime != current_regime_idx and confidence >= threshold:
            # Regime change detected
            old_regime = self.current_regime
            
            # Update end time of previous regime
            old_regime.end_time = datetime.now(pytz.utc)
            
            # Create new regime
            new_regime = Regime(
                regime_type=RegimeType(predicted_regime + 1),
                confidence=float(confidence),
                start_time=datetime.now(pytz.utc),
                features=self._get_current_features(prices, volumes)
            )
            
            # Update current regime
            self.current_regime = new_regime
            self.regime_history.append(new_regime)
            self.metrics['regime_changes'] += 1
            self.metrics['avg_confidence'] = (
                (self.metrics['avg_confidence'] * (self.metrics['regime_changes'] - 1) + confidence) / 
                self.metrics['regime_changes']
            )
            
            logger.info(f"Regime changed from {old_regime.regime_type.name} to {new_regime.regime_type.name} "
                       f"(confidence: {confidence:.2f})")
            
            return new_regime
            
        return None
    
    def _get_current_features(self, 
                            prices: ArrayLike, 
                            volumes: Optional[ArrayLike] = None) -> Dict[str, float]:
        """Extract features for the current time period."""
        # This is a simplified version that just returns basic statistics
        returns = np.diff(prices) / prices[:-1]
        
        features = {
            'mean_return': float(np.nanmean(returns)),
            'volatility': float(np.nanstd(returns)),
            'skewness': float(pd.Series(returns).skew()),
            'kurtosis': float(pd.Series(returns).kurt())
        }
        
        if volumes is not None and len(volumes) > 1:
            volume_changes = np.diff(volumes) / (volumes[:-1] + 1e-10)
            features.update({
                'volume_change': float(volume_changes[-1] if len(volume_changes) > 0 else 0.0),
                'volume_volatility': float(np.nanstd(volume_changes))
            })
            
        return features
    
    def _create_sequences(self, 
                         data: np.ndarray, 
                         seq_length: int) -> np.ndarray:
        """
        Create sequences for time series data.
        
        Args:
            data: Input data (n_samples, n_features)
            seq_length: Length of each sequence
            
        Returns:
            Array of sequences (n_sequences, seq_length, n_features)
        """
        sequences = []
        for i in range(len(data) - seq_length + 1):
            sequences.append(data[i:i + seq_length])
        return np.array(sequences)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get regime detection metrics.
        
        Returns:
            Dictionary of metrics
        """
        elapsed_hours = (datetime.now(pytz.utc) - self.metrics['start_time']).total_seconds() / 3600
        
        metrics = self.metrics.copy()
        metrics.update({
            'elapsed_hours': elapsed_hours,
            'predictions_per_hour': metrics['total_predictions'] / max(1, elapsed_hours),
            'regime_duration_hours': elapsed_hours / max(1, metrics['regime_changes']),
            'current_regime': self.current_regime.regime_type.name if self.current_regime else None,
            'current_confidence': self.current_regime.confidence if self.current_regime else 0.0,
            'last_update': datetime.now(pytz.utc).isoformat()
        })
        
        return metrics


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Generate some sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create different regimes
    regimes = [
        {'type': 'TRENDING_UP', 'mu': 0.001, 'sigma': 0.005, 'length': 200},
        {'type': 'VOLATILE', 'mu': 0.0001, 'sigma': 0.02, 'length': 200},
        {'type': 'TRENDING_DOWN', 'mu': -0.001, 'sigma': 0.008, 'length': 200},
        {'type': 'MEAN_REVERTING', 'mu': 0.0, 'sigma': 0.01, 'length': 200},
        {'type': 'LOW_VOLATILITY', 'mu': 0.0002, 'sigma': 0.002, 'length': 200}
    ]
    
    # Generate price series
    prices = [100.0]
    regime_labels = []
    
    for regime in regimes:
        regime_type = RegimeType[regime['type']]
        for _ in range(regime['length']):
            ret = np.random.normal(regime['mu'], regime['sigma'])
            prices.append(prices[-1] * (1 + ret))
            regime_labels.append(regime_type.value - 1)  # Convert to 0-based index
    
    # Create volumes (random for this example)
    volumes = np.random.lognormal(mean=8, sigma=0.5, size=len(prices))
    
    # Initialize and fit the regime detector
    detector = RegimeDetector(n_regimes=5, lookback_window=20, method='gmm')
    detector.fit(prices, volumes)
    
    # Predict regimes
    predictions = detector.predict(prices, volumes)
    
    # Print some results
    print(f"Detected {len(set(predictions))} different regimes")
    
    # If we have ground truth, calculate accuracy
    if len(regime_labels) == len(predictions):
        # For demonstration, we'll just print the classification report
        print("\nClassification Report:")
        print(classification_report(regime_labels, predictions, zero_division=0))
    
    # Get current regime
    current_regime = detector.detect_regime_change(prices, volumes)
    if current_regime:
        print(f"\nCurrent market regime: {current_regime.regime_type.name} "
              f"(confidence: {current_regime.confidence:.2f})")
    
    # Get metrics
    metrics = detector.get_metrics()
    print("\nRegime Detection Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
