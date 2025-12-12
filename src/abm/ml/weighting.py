"""
Dynamic Model Weighting

This module implements dynamic model weighting strategies for ensemble models
based on regime detection and performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import logging
from scipy.special import softmax
from sklearn.metrics import mean_squared_error

logger = logging.getLogger(__name__)

@dataclass
class ModelPerformance:
    """Tracks performance metrics for a single model."""
    name: str
    recent_errors: List[float] = None
    window_size: int = 100
    decay_factor: float = 0.95
    
    def __post_init__(self):
        self.recent_errors = []
        self.cumulative_loss = 0.0
        self.weight = 1.0
        self.last_updated = 0
    
    def update(self, y_true: np.ndarray, y_pred: np.ndarray, timestamp: int) -> None:
        """Update performance metrics with new predictions."""
        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")
        
        # Calculate error (MSE)
        error = mean_squared_error(y_true, y_pred)
        
        # Update recent errors with exponential decay
        if self.recent_errors:
            self.recent_errors = [e * self.decay_factor for e in self.recent_errors]
        
        self.recent_errors.append(error)
        self.recent_errors = self.recent_errors[-self.window_size:]
        
        # Update cumulative loss
        self.cumulative_loss = (self.cumulative_loss * self.decay_factor) + error
        self.last_updated = timestamp
    
    def get_performance_score(self) -> float:
        """Calculate a performance score (lower is better)."""
        if not self.recent_errors:
            return float('inf')
        
        # Use exponentially weighted moving average of recent errors
        weights = np.array([self.decay_factor ** i for i in range(len(self.recent_errors))][::-1])
        weighted_errors = np.array(self.recent_errors) * weights
        return np.sum(weighted_errors) / np.sum(weights)

class DynamicWeighting:
    """
    Implements dynamic weighting of ensemble models based on performance and regime.
    """
    
    def __init__(
        self,
        model_names: List[str],
        regime_detector: Optional[Any] = None,
        min_weight: float = 0.01,
        window_size: int = 100,
        decay_factor: float = 0.95,
        cold_start: str = 'uniform',
        cold_start_period: int = 10
    ):
        """
        Initialize the dynamic weighting system.
        
        Args:
            model_names: List of model names in the ensemble
            regime_detector: Optional regime detector instance
            min_weight: Minimum weight for any model
            window_size: Size of the sliding window for performance tracking
            decay_factor: Decay factor for older errors (0-1, higher = slower decay)
            cold_start: Strategy for initial weights ('uniform' or 'first')
            cold_start_period: Number of observations before dynamic weighting starts
        """
        self.models = {name: ModelPerformance(name, window_size, decay_factor) 
                      for name in model_names}
        self.regime_detector = regime_detector
        self.min_weight = min_weight
        self.window_size = window_size
        self.decay_factor = decay_factor
        self.cold_start = cold_start
        self.cold_start_period = cold_start_period
        self.observation_count = 0
        self.weights_history = []
        self.regime_history = []
        
        # Initialize weights
        self.weights = self._initialize_weights()
    
    def _initialize_weights(self) -> Dict[str, float]:
        """Initialize model weights based on cold start strategy."""
        n_models = len(self.models)
        
        if self.cold_start == 'uniform':
            return {name: 1.0 / n_models for name in self.models}
        elif self.cold_start == 'first':
            weights = {name: self.min_weight for name in self.models}
            first_model = next(iter(self.models))
            weights[first_model] = 1.0 - (self.min_weight * (n_models - 1))
            return weights
        else:
            raise ValueError(f"Unknown cold start strategy: {self.cold_start}")
    
    def update(
        self,
        y_true: np.ndarray,
        predictions: Dict[str, np.ndarray],
        timestamp: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Update model weights based on new predictions.
        
        Args:
            y_true: True target values
            predictions: Dictionary of model predictions {model_name: y_pred}
            timestamp: Optional timestamp for tracking
            
        Returns:
            Updated model weights
        """
        if timestamp is None:
            timestamp = self.observation_count
        
        self.observation_count += 1
        
        # Update model performances
        for name, model in self.models.items():
            if name in predictions:
                model.update(y_true, predictions[name], timestamp)
        
        # Skip weight update during cold start
        if self.observation_count < self.cold_start_period:
            return self.weights
        
        # Get current regime if detector is available
        current_regime = None
        if self.regime_detector is not None:
            try:
                current_regime = self.regime_detector.current_regime
                self.regime_history.append({
                    'timestamp': timestamp,
                    'regime': current_regime,
                    'regime_probs': getattr(self.regime_detector, 'regime_probs', None)
                })
            except Exception as e:
                logger.warning(f"Error getting current regime: {e}")
        
        # Calculate new weights based on performance
        if current_regime is not None and hasattr(self, f'_calculate_weights_regime_{current_regime}'):
            # Use regime-specific weighting if available
            weights = getattr(self, f'_calculate_weights_regime_{current_regime}')()
        else:
            # Default weighting based on performance
            weights = self._calculate_weights_performance()
        
        # Apply minimum weight constraint
        weights = {name: max(self.min_weight, weight) for name, weight in weights.items()}
        
        # Normalize weights
        total_weight = sum(weights.values())
        self.weights = {name: w / total_weight for name, w in weights.items()}
        
        # Track weight history
        self.weights_history.append({
            'timestamp': timestamp,
            'weights': self.weights.copy(),
            'regime': current_regime
        })
        
        return self.weights
    
    def _calculate_weights_performance(self) -> Dict[str, float]:
        """Calculate weights based on model performance."""
        # Get performance scores (lower is better)
        scores = {name: model.get_performance_score() 
                 for name, model in self.models.items()}
        
        # Convert to weights (inverse of scores)
        # Add small epsilon to avoid division by zero
        inv_scores = {name: 1.0 / (score + 1e-10) for name, score in scores.items()}
        total = sum(inv_scores.values())
        
        return {name: score / total for name, score in inv_scores.items()}
    
    def get_weights(self) -> Dict[str, float]:
        """Get current model weights."""
        return self.weights.copy()
    
    def get_weight_history(self) -> pd.DataFrame:
        """Get weight history as a DataFrame."""
        if not self.weights_history:
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(self.weights_history)
        
        # Explode weights dictionary into columns
        weights_df = pd.DataFrame(df['weights'].tolist(), index=df.index)
        df = pd.concat([df.drop('weights', axis=1), weights_df], axis=1)
        
        return df
    
    def get_regime_history(self) -> pd.DataFrame:
        """Get regime history as a DataFrame."""
        if not self.regime_history:
            return pd.DataFrame()
        return pd.DataFrame(self.regime_history)
    
    def plot_weights(self, figsize=(12, 6)) -> None:
        """Plot the evolution of model weights over time."""
        import matplotlib.pyplot as plt
        
        df = self.get_weight_history()
        if df.empty:
            logger.warning("No weight history to plot")
            return
        
        plt.figure(figsize=figsize)
        
        # Plot weights
        weight_cols = [col for col in df.columns if col not in ['timestamp', 'regime']]
        for col in weight_cols:
            plt.plot(df['timestamp'], df[col], label=col, linewidth=2)
        
        # Add regime changes if available
        if 'regime' in df.columns and df['regime'].notna().any():
            regime_changes = df[df['regime'] != df['regime'].shift(1)]
            for _, row in regime_changes.iterrows():
                plt.axvline(x=row['timestamp'], color='gray', linestyle='--', alpha=0.5)
                plt.text(
                    row['timestamp'], 1.02, f"Regime {row['regime']}",
                    ha='center', va='bottom', rotation=45, fontsize=8
                )
        
        plt.title('Model Weights Over Time')
        plt.xlabel('Time')
        plt.ylabel('Weight')
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

class RegimeAwareWeighting(DynamicWeighting):
    """
    Extends DynamicWeighting with regime-specific weighting strategies.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regime_models = {}  # Track best models per regime
    
    def _calculate_weights_regime_0(self) -> Dict[str, float]:
        """Weighting strategy for regime 0 (e.g., trending market)."""
        # In trending markets, give more weight to momentum-based models
        weights = {}
        for name, model in self.models.items():
            if 'momentum' in name.lower() or 'trend' in name.lower():
                weights[name] = model.get_performance_score() * 0.5  # Higher weight
            else:
                weights[name] = model.get_performance_score()
        
        # Invert scores to get weights (lower score = higher weight)
        inv_weights = {name: 1.0 / (score + 1e-10) for name, score in weights.items()}
        total = sum(inv_weights.values())
        return {name: w / total for name, w in inv_weights.items()}
    
    def _calculate_weights_regime_1(self) -> Dict[str, float]:
        """Weighting strategy for regime 1 (e.g., mean-reverting market)."""
        # In mean-reverting markets, give more weight to mean-reversion models
        weights = {}
        for name, model in self.models.items():
            if 'mean' in name.lower() or 'reversion' in name.lower():
                weights[name] = model.get_performance_score() * 0.5  # Higher weight
            else:
                weights[name] = model.get_performance_score()
        
        inv_weights = {name: 1.0 / (score + 1e-10) for name, score in weights.items()}
        total = sum(inv_weights.values())
        return {name: w / total for name, w in inv_weights.items()}
    
    def _calculate_weights_regime_2(self) -> Dict[str, float]:
        """Weighting strategy for regime 2 (e.g., high volatility)."""
        # In high volatility, give more weight to models that handle volatility well
        weights = {}
        for name, model in self.models.items():
            if 'volatility' in name.lower() or 'garch' in name.lower():
                weights[name] = model.get_performance_score() * 0.5  # Higher weight
            else:
                weights[name] = model.get_performance_score()
        
        inv_weights = {name: 1.0 / (score + 1e-10) for name, score in weights.items()}
        total = sum(inv_weights.values())
        return {name: w / total for name, w in inv_weights.items()}

def create_weighting_strategy(
    strategy: str = 'dynamic',
    model_names: List[str] = None,
    **kwargs
) -> Union[DynamicWeighting, RegimeAwareWeighting]:
    """
    Factory function to create a weighting strategy.
    
    Args:
        strategy: Weighting strategy ('dynamic' or 'regime')
        model_names: List of model names
        **kwargs: Additional arguments for the weighting strategy
        
    Returns:
        An instance of the specified weighting strategy
    """
    if strategy == 'dynamic':
        return DynamicWeighting(model_names=model_names, **kwargs)
    elif strategy == 'regime':
        return RegimeAwareWeighting(model_names=model_names, **kwargs)
    else:
        raise ValueError(f"Unknown weighting strategy: {strategy}")
