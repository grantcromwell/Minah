"""
Regime Detection Module

This module implements market regime detection using Hidden Markov Models (HMM)
and other statistical methods to identify different market conditions.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from hmmlearn import hmm
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class MarketRegimeDetector:
    """
    Detects and classifies market regimes using various statistical methods.
    
    This class implements multiple regime detection algorithms including:
    - Hidden Markov Models (HMM)
    - Gaussian Mixture Models (GMM)
    - Statistical breakpoint detection
    """
    
    def __init__(
        self,
        method: str = 'hmm',
        n_regimes: int = 3,
        lookback_window: int = 252,
        features: Optional[List[str]] = None,
        **kwargs
    ):
        """
        Initialize the regime detector.
        
        Args:
            method: Detection method ('hmm', 'gmm', or 'statistical')
            n_regimes: Number of market regimes to detect
            lookback_window: Number of periods to use for regime detection
            features: List of feature names to use for regime detection
            **kwargs: Additional parameters for the underlying model
        """
        self.method = method.lower()
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.features = features or ['returns', 'volatility']
        self.scaler = StandardScaler()
        self.model = None
        self.regime_labels = None
        self.current_regime = None
        self.regime_probs = None
        self.kwargs = kwargs
        
        self._initialize_model()
    
    def _initialize_model(self) -> None:
        """Initialize the regime detection model based on the selected method."""
        if self.method == 'hmm':
            self.model = hmm.GaussianHMM(
                n_components=self.n_regimes,
                covariance_type="full",
                n_iter=1000,
                random_state=42,
                **self.kwargs
            )
        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                covariance_type='full',
                random_state=42,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported regime detection method: {self.method}")
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """
        Extract relevant features for regime detection.
        
        Args:
            data: DataFrame containing price/return data
            
        Returns:
            Array of features for regime detection
        """
        features = []
        
        if 'returns' in self.features:
            if 'returns' not in data.columns:
                if 'close' in data.columns:
                    data['returns'] = data['close'].pct_change()
                else:
                    raise ValueError("Could not calculate returns: 'close' column not found")
            features.append(data['returns'].values.reshape(-1, 1))
        
        if 'volatility' in self.features:
            if 'returns' not in data.columns:
                if 'close' in data.columns:
                    data['returns'] = data['close'].pct_change()
                else:
                    raise ValueError("Could not calculate volatility: 'close' column not found")
            data['volatility'] = data['returns'].rolling(window=20).std()
            features.append(data['volatility'].values.reshape(-1, 1))
        
        if 'volume' in self.features and 'volume' in data.columns:
            features.append(data['volume'].values.reshape(-1, 1))
        
        if not features:
            raise ValueError("No valid features could be extracted")
        
        # Combine features and handle NaN values
        X = np.hstack(features)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        return X
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray]) -> 'MarketRegimeDetector':
        """
        Fit the regime detection model to historical data.
        
        Args:
            data: Historical market data (DataFrame or numpy array)
            
        Returns:
            self: Returns an instance of self
        """
        if isinstance(data, pd.DataFrame):
            X = self.extract_features(data)
        else:
            X = data
        
        # Scale the features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        if self.method == 'hmm':
            self.model.fit(X_scaled)
            # Get the most likely sequence of states
            self.regime_labels = self.model.predict(X_scaled)
            self.regime_probs = self.model.predict_proba(X_scaled)
        elif self.method == 'gmm':
            self.model.fit(X_scaled)
            self.regime_labels = self.model.predict(X_scaled)
            self.regime_probs = self.model.predict_proba(X_scaled)
        
        # Set the current regime to the most recent one
        if len(self.regime_labels) > 0:
            self.current_regime = self.regime_labels[-1]
        
        return self
    
    def update(self, new_data: Union[pd.DataFrame, np.ndarray]) -> int:
        """
        Update the regime detection with new data.
        
        Args:
            new_data: New market data to update the regime detection
            
        Returns:
            int: The updated regime label
        """
        if isinstance(new_data, pd.DataFrame):
            X_new = self.extract_features(new_data)
        else:
            X_new = new_data
        
        # Scale the new data
        X_scaled = self.scaler.transform(X_new)
        
        # Update the regime detection
        if self.method == 'hmm':
            new_regime = self.model.predict(X_scaled[-1].reshape(1, -1))[0]
            new_probs = self.model.predict_proba(X_scaled[-1].reshape(1, -1))[0]
        elif self.method == 'gmm':
            new_regime = self.model.predict(X_scaled[-1].reshape(1, -1))[0]
            new_probs = self.model.predict_proba(X_scaled[-1].reshape(1, -1))[0]
        
        # Update the current regime and probabilities
        self.current_regime = new_regime
        self.regime_probs = new_probs
        
        return new_regime
    
    def get_regime_metrics(self) -> Dict:
        """
        Get metrics about the current regime.
        
        Returns:
            Dictionary containing regime metrics
        """
        if self.regime_probs is None:
            return {}
        
        return {
            'current_regime': int(self.current_regime) if self.current_regime is not None else None,
            'regime_probabilities': self.regime_probs.tolist() if hasattr(self.regime_probs, 'tolist') else self.regime_probs,
            'regime_confidence': float(np.max(self.regime_probs)) if self.regime_probs is not None else None,
            'n_regimes': self.n_regimes
        }
    
    def plot_regimes(self, price_series: pd.Series, save_path: Optional[str] = None) -> None:
        """
        Plot the price series with regime shading.
        
        Args:
            price_series: Time series of prices
            save_path: Optional path to save the plot
        """
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        
        if self.regime_labels is None or len(self.regime_labels) != len(price_series):
            raise ValueError("Regime labels not available or length mismatch with price series")
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot price series
        ax1.plot(price_series.index, price_series, 'k-', linewidth=1.5, label='Price')
        ax1.set_ylabel('Price')
        ax1.set_title('Market Regimes')
        
        # Shade regions by regime
        regime_changes = np.diff(np.concatenate(([0], self.regime_labels, [0])))
        regime_starts = np.where(regime_changes != 0)[0]
        
        for i in range(len(regime_starts) - 1):
            start_idx = regime_starts[i]
            end_idx = regime_starts[i + 1] if i < len(regime_starts) - 1 else len(price_series)
            
            if start_idx >= len(price_series):
                continue
                
            regime = self.regime_labels[start_idx]
            color = plt.cm.tab10(regime % 10)
            
            # Handle datetime index
            if hasattr(price_series.index, 'values'):
                start_date = price_series.index[start_idx]
                end_date = price_series.index[min(end_idx, len(price_series) - 1)]
            else:
                start_date = start_idx
                end_date = end_idx
            
            ax1.axvspan(start_date, end_date, alpha=0.2, color=color, label=f'Regime {regime}' if i == 0 else "")
        
        # Plot regime probabilities
        if self.regime_probs is not None and len(self.regime_probs) == len(price_series):
            for i in range(self.n_regimes):
                probs = [p[i] for p in self.regime_probs]
                ax2.plot(price_series.index, probs, label=f'Regime {i}')
        
        ax2.set_ylabel('Probability')
        ax2.set_ylim(0, 1)
        
        # Format x-axis
        if len(price_series) > 0:
            if hasattr(price_series.index, 'strftime'):
                # Handle datetime index
                if len(price_series) > 252:  # More than a year of daily data
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax1.xaxis.set_major_locator(mdates.YearLocator())
                elif len(price_series) > 60:  # More than 3 months of daily data
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
                    ax1.xaxis.set_major_locator(mdates.MonthLocator())
                else:
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(price_series) // 10)))
                
                plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
                plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Add legend
        handles, labels = ax1.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax1.legend(by_label.values(), by_label.keys(), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
