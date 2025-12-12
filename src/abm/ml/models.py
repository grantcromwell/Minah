"""
Financial Time Series Models

This module implements various ML models for financial time series prediction,
all compatible with the ensemble framework.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from abc import ABC, abstractmethod
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

from .ensemble import BaseModel, ModelPrediction

logger = logging.getLogger(__name__)

class FinancialModel(BaseModel, ABC):
    """Base class for financial time series models with common functionality."""
    
    def __init__(self, name: str, lookback: int = 10, **kwargs):
        super().__init__(name=name)
        self.lookback = lookback
        self.scaler = StandardScaler()
        self.feature_importance_ = None
    
    def _prepare_sequences(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> tuple:
        """Convert time series into sequences for supervised learning."""
        Xs, ys = [], []
        
        for i in range(len(X) - self.lookback):
            Xs.append(X[i:(i + self.lookback)])
            if y is not None:
                ys.append(y[i + self.lookback])
        
        Xs = np.array(Xs)
        if y is not None:
            ys = np.array(ys)
            return Xs, ys
        return Xs
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """Get feature importance if available."""
        return self.feature_importance_
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            'name': self.name,
            'lookback': self.lookback,
            'is_fitted': self.is_fitted
        }

class RandomForestModel(FinancialModel):
    """Random Forest model for financial time series prediction."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 5,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name=f"RandomForest_{n_estimators}", **kwargs)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """Train the model on the given data."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, y)
            
            # Reshape for RF (flatten the lookback dimension)
            n_samples, lookback, n_features = X_seq.shape
            X_rf = X_seq.reshape((n_samples, lookback * n_features))
            
            # Train model
            self.model.fit(X_rf, y_seq)
            self.is_fitted = True
            
            # Store feature importance
            self.feature_importance_ = self.model.feature_importances_
            
            logger.info(f"Trained {self.name} with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error fitting RandomForestModel: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Prepare sequences
            X_seq = self._prepare_sequences(X_scaled)
            
            # Reshape for RF
            n_samples, lookback, n_features = X_seq.shape
            X_rf = X_seq.reshape((n_samples, lookback * n_features))
            
            # Make predictions
            y_pred = self.model.predict(X_rf)
            
            # Get prediction confidence (using out-of-bag score as proxy)
            confidence = self.model.oob_score_ if hasattr(self.model, 'oob_score_') else 0.7
            
            return ModelPrediction(
                prediction=y_pred,
                confidence=float(confidence),
                model_name=self.name
            )
            
        except Exception as e:
            logger.error(f"Error predicting with RandomForestModel: {str(e)}")
            raise
    
    def update(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """Update the model with new data (partial fit)."""
        try:
            # For RandomForest, we need to retrain on the entire dataset
            # In a production setting, you might want to implement a more efficient update mechanism
            self.fit(X, y, **kwargs)
            return self
        except Exception as e:
            logger.error(f"Error updating RandomForestModel: {str(e)}")
            raise

class LSTMModel(FinancialModel):
    """LSTM model for financial time series prediction."""
    
    def __init__(
        self,
        units: int = 50,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        validation_split: float = 0.1,
        **kwargs
    ):
        super().__init__(name=f"LSTM_{units}U", **kwargs)
        self.units = units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.model = None
    
    def _build_model(self, input_shape: tuple) -> None:
        """Build the LSTM model architecture."""
        self.model = Sequential([
            LSTM(
                self.units,
                return_sequences=True,
                input_shape=input_shape,
                kernel_regularizer=l2(0.01),
                recurrent_regularizer=l2(0.01)
            ),
            Dropout(self.dropout),
            LSTM(
                self.units // 2,
                kernel_regularizer=l2(0.01),
                recurrent_regularizer=l2(0.01)
            ),
            Dropout(self.dropout),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=self.learning_rate)
        self.model.compile(optimizer=optimizer, loss='mse')
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LSTMModel':
        """Train the LSTM model on the given data."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, y)
            
            # Build model if not already built
            if self.model is None:
                self._build_model(input_shape=(X_seq.shape[1], X_seq.shape[2]))
            
            # Define callbacks
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                )
            ]
            
            # Train model
            history = self.model.fit(
                X_seq, y_seq,
                epochs=self.epochs,
                batch_size=self.batch_size,
                validation_split=self.validation_split,
                callbacks=callbacks,
                verbose=0
            )
            
            self.is_fitted = True
            self.training_history_ = history.history
            
            logger.info(f"Trained {self.name} with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error fitting LSTMModel: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """Make predictions on new data."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Prepare sequences
            X_seq = self._prepare_sequences(X_scaled)
            
            # Make predictions
            y_pred = self.model.predict(X_seq, verbose=0).flatten()
            
            # Estimate confidence using model's loss on training data
            confidence = 1.0 / (1.0 + self.training_history_['loss'][-1])
            
            return ModelPrediction(
                prediction=y_pred,
                confidence=float(confidence),
                model_name=self.name
            )
            
        except Exception as e:
            logger.error(f"Error predicting with LSTMModel: {str(e)}")
            raise
    
    def update(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LSTMModel':
        """Update the model with new data (online learning)."""
        try:
            # For LSTM, we can perform a partial fit on new data
            if not self.is_fitted or self.model is None:
                return self.fit(X, y, **kwargs)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, y)
            
            # Perform a small number of training steps on new data
            self.model.fit(
                X_seq, y_seq,
                epochs=5,  # Few epochs for online learning
                batch_size=min(32, len(X_seq)),
                verbose=0
            )
            
            logger.info(f"Updated {self.name} with {len(X)} new samples")
            
            return self
            
        except Exception as e:
            logger.error(f"Error updating LSTMModel: {str(e)}")
            raise

class GradientBoostingModel(FinancialModel):
    """Gradient Boosting model for financial time series prediction."""
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        subsample: float = 0.8,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(name=f"GradientBoosting_{n_estimators}", **kwargs)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'GradientBoostingModel':
        """Train the model on the given data."""
        try:
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, y)
            
            # Reshape for GBM (flatten the lookback dimension)
            n_samples, lookback, n_features = X_seq.shape
            X_gbm = X_seq.reshape((n_samples, lookback * n_features))
            
            # Train model
            self.model.fit(X_gbm, y_seq)
            self.is_fitted = True
            
            # Store feature importance
            self.feature_importance_ = self.model.feature_importances_
            
            logger.info(f"Trained {self.name} with {len(X)} samples")
            
        except Exception as e:
            logger.error(f"Error fitting GradientBoostingModel: {str(e)}")
            raise
        
        return self
    
    def predict(self, X: np.ndarray, **kwargs) -> ModelPrediction:
        """Make predictions on new data."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        try:
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Prepare sequences
            X_seq = self._prepare_sequences(X_scaled)
            
            # Reshape for GBM
            n_samples, lookback, n_features = X_seq.shape
            X_gbm = X_seq.reshape((n_samples, lookback * n_features))
            
            # Make predictions
            y_pred = self.model.predict(X_gbm)
            
            # Get prediction confidence (using out-of-bag score as proxy)
            confidence = 0.8  # Default confidence
            
            return ModelPrediction(
                prediction=y_pred,
                confidence=confidence,
                model_name=self.name
            )
            
        except Exception as e:
            logger.error(f"Error predicting with GradientBoostingModel: {str(e)}")
            raise
    
    def update(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'GradientBoostingModel':
        """Update the model with new data (partial fit)."""
        try:
            # For GBM, we can use warm_start to continue training
            if not self.is_fitted:
                return self.fit(X, y, **kwargs)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Prepare sequences
            X_seq, y_seq = self._prepare_sequences(X_scaled, y)
            
            # Reshape for GBM
            n_samples, lookback, n_features = X_seq.shape
            X_gbm = X_seq.reshape((n_samples, lookback * n_features))
            
            # Update model
            self.model.n_estimators += 10  # Add more trees
            self.model.warm_start = True
            self.model.fit(X_gbm, y_seq)
            
            logger.info(f"Updated {self.name} with {len(X)} new samples")
            
            return self
            
        except Exception as e:
            logger.error(f"Error updating GradientBoostingModel: {str(e)}")
            raise

# Factory function for creating models
def create_model(model_type: str, **kwargs) -> FinancialModel:
    """Factory function to create a model by name."""
    model_map = {
        'random_forest': RandomForestModel,
        'lstm': LSTMModel,
        'gradient_boosting': GradientBoostingModel,
    }
    
    if model_type not in model_map:
        raise ValueError(f"Unknown model type: {model_type}. Available types: {list(model_map.keys())}")
    
    return model_map[model_type](**kwargs)
