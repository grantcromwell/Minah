"""
Ensemble Model Framework

This module implements an ensemble of models with dynamic weighting based on
market regimes and model confidence. It combines predictions from multiple models
to improve robustness and adapt to changing market conditions.
"""

from typing import Dict, List, Tuple, Optional, Any, Union, Callable
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

@dataclass
class ModelPrediction:
    """Container for model predictions with confidence scores."""
    prediction: np.ndarray
    confidence: float
    model_name: str
    features: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseModel(ABC):
    """Abstract base class for all prediction models in the ensemble."""
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Train the model on the given data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """Make predictions on new data."""
        pass
    
    @abstractmethod
    def update(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseModel':
        """Update the model with new data (online learning)."""
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {}
    
    def set_params(self, **params) -> 'BaseModel':
        """Set model parameters."""
        return self

class EnsembleModel:
    """
    Ensemble of models with dynamic weighting based on market regimes.
    
    This class combines predictions from multiple models, with weights that can
    adapt based on market conditions and model performance.
    """
    
    def __init__(
        self,
        models: Optional[Dict[str, BaseModel]] = None,
        regime_detector: Optional[Any] = None,
        initial_weights: Optional[Dict[str, float]] = None,
        regime_weights: Optional[Dict[Any, Dict[str, float]]] = None,
        decay_factor: float = 0.99,
        min_weight: float = 0.01,
        max_models: int = 10,
        **kwargs
    ):
        """
        Initialize the ensemble model.
        
        Args:
            models: Dictionary of models to include in the ensemble
            regime_detector: Optional regime detector for dynamic weighting
            initial_weights: Initial weights for each model
            regime_weights: Dictionary mapping regimes to model weights
            decay_factor: Factor for exponential decay of model performance
            min_weight: Minimum weight for any model
            max_models: Maximum number of models to keep in the ensemble
            **kwargs: Additional keyword arguments
        """
        self.models = models or {}
        self.regime_detector = regime_detector
        self.initial_weights = initial_weights or {}
        self.regime_weights = regime_weights or {}
        self.decay_factor = decay_factor
        self.min_weight = min_weight
        self.max_models = max_models
        
        # Track model performance
        self.model_performance = {}
        self.weights_history = []
        self.regime_history = []
        
        # Initialize weights if not provided
        if not self.initial_weights and self.models:
            self.initial_weights = {name: 1.0/len(self.models) for name in self.models}
        
        # Current weights (dynamically updated)
        self.current_weights = self.initial_weights.copy()
    
    def add_model(self, name: str, model: BaseModel, weight: Optional[float] = None) -> None:
        """
        Add a model to the ensemble.
        
        Args:
            name: Name of the model
            model: Model instance
            weight: Initial weight (if None, will be set automatically)
        """
        if name in self.models:
            logger.warning(f"Model '{name}' already exists in the ensemble. It will be replaced.")
        
        self.models[name] = model
        
        if weight is not None:
            self.initial_weights[name] = weight
        elif not self.initial_weights:
            self.initial_weights[name] = 1.0 / (len(self.models) + 1)
        
        # Normalize weights
        self._normalize_weights()
        
        # Initialize performance tracking
        if name not in self.model_performance:
            self.model_performance[name] = {
                'recent_errors': [],
                'total_predictions': 0,
                'cumulative_loss': 0.0,
                'last_updated': 0
            }
    
    def remove_model(self, name: str) -> None:
        """
        Remove a model from the ensemble.
        
        Args:
            name: Name of the model to remove
        """
        if name in self.models:
            del self.models[name]
            if name in self.initial_weights:
                del self.initial_weights[name]
            if name in self.model_performance:
                del self.model_performance[name]
            
            # Redistribute weights
            self._normalize_weights()
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleModel':
        """
        Train all models in the ensemble.
        
        Args:
            X: Input features
            y: Target values
            **kwargs: Additional arguments to pass to model fit methods
            
        Returns:
            self: Returns an instance of self
        """
        for name, model in self.models.items():
            logger.info(f"Training model: {name}")
            try:
                model.fit(X, y, **kwargs)
                self.model_performance[name] = {
                    'recent_errors': [],
                    'total_predictions': 0,
                    'cumulative_loss': 0.0,
                    'last_updated': 0
                }
            except Exception as e:
                logger.error(f"Error training model {name}: {str(e)}")
                if name in self.models:
                    del self.models[name]
        
        # Update regime weights if regime detector is available
        if self.regime_detector is not None and hasattr(X, 'index') and hasattr(self.regime_detector, 'fit'):
            self.regime_detector.fit(X)
            self._update_regime_weights()
        
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """
        Make predictions using the ensemble.
        
        Args:
            X: Input features
            **kwargs: Additional arguments to pass to model predict methods
            
        Returns:
            ModelPrediction: Combined prediction with confidence
        """
        if not self.models:
            raise ValueError("No models in the ensemble")
        
        # Get current regime if detector is available
        current_regime = None
        if self.regime_detector is not None:
            try:
                if hasattr(self.regime_detector, 'update'):
                    current_regime = self.regime_detector.update(X)
                else:
                    current_regime = self.regime_detector.current_regime
                self.regime_history.append({
                    'timestamp': len(self.regime_history),
                    'regime': current_regime
                })
            except Exception as e:
                logger.error(f"Error updating regime detector: {str(e)}")
        
        # Get predictions from all models
        predictions = {}
        confidences = {}
        
        for name, model in self.models.items():
            try:
                pred = model.predict(X, **kwargs)
                predictions[name] = pred.prediction
                confidences[name] = pred.confidence
            except Exception as e:
                logger.error(f"Error getting prediction from model {name}: {str(e)}")
                predictions[name] = None
                confidences[name] = 0.0
        
        # Filter out None predictions
        valid_models = [name for name, pred in predictions.items() if pred is not None]
        if not valid_models:
            raise ValueError("No valid predictions from any model")
        
        # Get current weights (considering regime if available)
        if current_regime is not None and current_regime in self.regime_weights:
            weights = self.regime_weights[current_regime].copy()
        else:
            weights = self.current_weights.copy()
        
        # Apply confidence-based adjustments
        for name in valid_models:
            if name in confidences and name in weights:
                weights[name] *= confidences[name]
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w/total_weight for name, w in weights.items()}
        
        # Store weights history
        self.weights_history.append({
            'timestamp': len(self.weights_history),
            'weights': weights.copy(),
            'regime': current_regime
        })
        
        # Combine predictions using weighted average
        combined_pred = np.zeros_like(next(p for p in predictions.values() if p is not None))
        total_weight = 0.0
        
        for name, pred in predictions.items():
            if pred is not None and name in weights and weights[name] > 0:
                combined_pred += pred * weights[name]
                total_weight += weights[name]
        
        if total_weight > 0:
            combined_pred /= total_weight
        
        # Calculate ensemble confidence (weighted average of confidences)
        avg_confidence = np.average(
            [confidences[name] for name in valid_models],
            weights=[weights.get(name, 0) for name in valid_models]
        )
        
        return ModelPrediction(
            prediction=combined_pred,
            confidence=avg_confidence,
            model_name='ensemble',
            metadata={
                'model_weights': weights,
                'model_confidences': confidences,
                'regime': current_regime
            }
        )
    
    def update(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleModel':
        """
        Update models with new data (online learning).
        
        Args:
            X: New input features
            y: New target values
            **kwargs: Additional arguments to pass to model update methods
            
        Returns:
            self: Returns an instance of self
        """
        # Get predictions before updating
        try:
            pred = self.predict(X, **kwargs)
            error = np.mean((pred.prediction - y) ** 2)  # MSE
        except Exception as e:
            logger.error(f"Error getting predictions for update: {str(e)}")
            error = 1.0  # Default high error if prediction fails
        
        # Update each model
        for name, model in self.models.items():
            try:
                if hasattr(model, 'update'):
                    model.update(X, y, **kwargs)
                # Update performance metrics
                if name in self.model_performance:
                    self.model_performance[name]['recent_errors'].append(error)
                    self.model_performance[name]['total_predictions'] += 1
                    self.model_performance[name]['cumulative_loss'] += error
                    self.model_performance[name]['last_updated'] = len(self.weights_history) - 1
            except Exception as e:
                logger.error(f"Error updating model {name}: {str(e)}")
        
        # Update model weights based on performance
        self._update_weights()
        
        # Update regime weights if regime detector is available
        if self.regime_detector is not None and hasattr(self.regime_detector, 'update'):
            try:
                self.regime_detector.update(X)
                self._update_regime_weights()
            except Exception as e:
                logger.error(f"Error updating regime detector: {str(e)}")
        
        return self
    
    def _update_weights(self) -> None:
        """Update model weights based on recent performance."""
        if not self.model_performance:
            return
        
        # Calculate performance scores (lower is better)
        scores = {}
        for name, perf in self.model_performance.items():
            if perf['recent_errors']:
                # Use exponential moving average of recent errors
                ema = 0.0
                for i, error in enumerate(reversed(perf['recent_errors'])):
                    ema = self.decay_factor * ema + (1 - self.decay_factor) * error
                scores[name] = ema
            else:
                scores[name] = 1.0  # Default score for new models
        
        # Convert scores to weights (lower score = higher weight)
        min_score = min(scores.values()) if scores else 1.0
        max_score = max(scores.values()) if scores else 1.0
        
        if max_score > min_score:
            # Normalize scores to [0, 1] and invert (so lower scores become higher weights)
            weights = {
                name: 1.0 - (score - min_score) / (max_score - min_score + 1e-10)
                for name, score in scores.items()
            }
        else:
            # All scores are equal, use uniform weights
            weights = {name: 1.0/len(scores) for name in scores}
        
        # Apply minimum weight
        for name in weights:
            weights[name] = max(self.min_weight, weights[name])
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            self.current_weights = {name: w/total_weight for name, w in weights.items()}
    
    def _update_regime_weights(self) -> None:
        """Update regime-specific weights based on model performance in each regime."""
        if self.regime_detector is None or not hasattr(self.regime_detector, 'regime_labels'):
            return
        
        # This is a placeholder - in a real implementation, you would track
        # model performance separately for each regime and update weights accordingly
        pass
    
    def _normalize_weights(self) -> None:
        """Normalize model weights to sum to 1."""
        if not self.current_weights:
            return
        
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            self.current_weights = {name: w/total_weight for name, w in self.current_weights.items()}
    
    def get_model_weights(self) -> Dict[str, float]:
        """Get the current model weights."""
        return self.current_weights.copy()
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all models."""
        return {
            name: {
                'recent_errors': perf['recent_errors'][-100:],  # Last 100 errors
                'total_predictions': perf['total_predictions'],
                'average_loss': perf['cumulative_loss'] / max(1, perf['total_predictions']),
                'last_updated': perf['last_updated']
            }
            for name, perf in self.model_performance.items()
        }
    
    def get_weights_history(self) -> pd.DataFrame:
        """Get the history of model weights over time."""
        if not self.weights_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.weights_history)
        weights_df = pd.DataFrame(df['weights'].tolist(), index=df.index)
        df = pd.concat([df.drop('weights', axis=1), weights_df], axis=1)
        return df
    
    def get_regime_history(self) -> pd.DataFrame:
        """Get the history of market regimes."""
        if not self.regime_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.regime_history)
