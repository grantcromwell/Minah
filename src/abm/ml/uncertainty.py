"""
Uncertainty-Aware Training

This module implements uncertainty estimation techniques for machine learning models,
including Monte Carlo Dropout, Deep Ensembles, and Bayesian Neural Networks.
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict, List, Optional, Tuple, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        logger.warning(f"Could not set GPU memory growth: {e}")

@dataclass
class PredictionWithUncertainty:
    """Container for predictions with uncertainty estimates."""
    prediction: np.ndarray
    uncertainty: np.ndarray
    lower_bound: np.ndarray
    upper_bound: np.ndarray
    samples: Optional[np.ndarray] = None
    confidence_interval: float = 0.95

class UncertaintyModel(ABC):
    """Abstract base class for uncertainty-aware models."""
    
    @abstractmethod
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        n_samples: int = 100,
        confidence_interval: float = 0.95
    ) -> PredictionWithUncertainty:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: Input features
            n_samples: Number of samples for uncertainty estimation
            confidence_interval: Desired confidence interval (0-1)
            
        Returns:
            PredictionWithUncertainty object
        """
        pass
    
    @abstractmethod
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform a single training step."""
        pass

class MCDropoutModel(UncertaintyModel):
    """
    Monte Carlo Dropout for uncertainty estimation.
    
    References:
        "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning"
        Yarin Gal, Zoubin Ghahramani (2016)
    """
    
    def __init__(
        self,
        model: tf.keras.Model,
        dropout_layers: Optional[List[int]] = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        dropout_rate: float = 0.1
    ):
        """
        Initialize the MC Dropout model.
        
        Args:
            model: Base Keras model (should include Dropout layers)
            dropout_layers: Indices of layers to apply MC Dropout
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization factor
            dropout_rate: Dropout rate (if model needs to be modified)
        """
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dropout_rate = dropout_rate
        self.dropout_layers = dropout_layers or self._find_dropout_layers()
        
        # Ensure dropout is enabled at test time
        self._set_dropout_trainable(True)
        
        # Compile model
        self._compile_model()
    
    def _find_dropout_layers(self) -> List[int]:
        """Find indices of Dropout layers in the model."""
        dropout_layers = []
        for i, layer in enumerate(self.model.layers):
            if isinstance(layer, tf.keras.layers.Dropout):
                dropout_layers.append(i)
        
        if not dropout_layers:
            logger.warning("No Dropout layers found. Adding Dropout before the output layer.")
            # Add a Dropout layer before the output
            x = self.model.layers[-2].output
            x = tf.keras.layers.Dropout(self.dropout_rate)(x, training=True)
            output = self.model.layers[-1](x)
            self.model = tf.keras.Model(inputs=self.model.inputs, outputs=output)
            dropout_layers = [len(self.model.layers) - 2]
        
        return dropout_layers
    
    def _set_dropout_trainable(self, trainable: bool) -> None:
        """Set dropout layers to be trainable or not."""
        for i in self.dropout_layers:
            if i < len(self.model.layers):
                self.model.layers[i].trainable = trainable
                # Ensure training=True for MC Dropout
                if hasattr(self.model.layers[i], 'training'):
                    self.model.layers[i].training = True
    
    def _compile_model(self) -> None:
        """Compile the model with appropriate loss and metrics."""
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        n_samples: int = 100,
        confidence_interval: float = 0.95
    ) -> PredictionWithUncertainty:
        """
        Make predictions with uncertainty estimates using MC Dropout.
        
        Args:
            X: Input features
            n_samples: Number of forward passes
            confidence_interval: Desired confidence interval (0-1)
            
        Returns:
            PredictionWithUncertainty object
        """
        # Generate multiple predictions
        samples = []
        for _ in range(n_samples):
            pred = self.model.predict(X, verbose=0)
            samples.append(pred)
        
        samples = np.array(samples)  # [n_samples, batch_size, n_outputs]
        
        # Calculate statistics
        mean_pred = np.mean(samples, axis=0)
        std_pred = np.std(samples, axis=0)
        
        # Calculate confidence intervals
        alpha = 1.0 - confidence_interval
        lower = np.percentile(samples, 100 * alpha/2, axis=0)
        upper = np.percentile(samples, 100 * (1 - alpha/2), axis=0)
        
        return PredictionWithUncertainty(
            prediction=mean_pred,
            uncertainty=std_pred,
            lower_bound=lower,
            upper_bound=upper,
            samples=samples,
            confidence_interval=confidence_interval
        )
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary of training metrics
        """
        # Enable dropout during training
        self._set_dropout_trainable(True)
        
        # Train for one epoch
        history = self.model.fit(
            X, y,
            batch_size=min(32, len(X)),
            epochs=1,
            verbose=0
        )
        
        return {k: v[0] for k, v in history.history.items()}

class DeepEnsemble(UncertaintyModel):
    """
    Deep Ensemble for uncertainty estimation.
    
    References:
        "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
        Balaji Lakshminarayanan, Alexander Pritzel, Charles Blundell (2017)
    """
    
    def __init__(
        self,
        model_fn: Callable[[], tf.keras.Model],
        n_models: int = 5,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        diversity_weight: float = 0.0
    ):
        """
        Initialize the Deep Ensemble.
        
        Args:
            model_fn: Function that returns a compiled Keras model
            n_models: Number of models in the ensemble
            learning_rate: Learning rate for optimizers
            weight_decay: L2 regularization factor
            diversity_weight: Weight for diversity term in loss (0 to disable)
        """
        self.model_fn = model_fn
        self.n_models = n_models
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.diversity_weight = diversity_weight
        
        # Initialize models
        self.models = []
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize the ensemble of models."""
        for i in range(self.n_models):
            model = self.model_fn()
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate,
                weight_decay=self.weight_decay
            )
            model.compile(
                optimizer=optimizer,
                loss='mse',
                metrics=['mae']
            )
            self.models.append(model)
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        n_samples: int = 1,
        confidence_interval: float = 0.95
    ) -> PredictionWithUncertainty:
        """
        Make predictions with uncertainty estimates using the ensemble.
        
        Args:
            X: Input features
            n_samples: Number of samples per model (for models with stochasticity)
            confidence_interval: Desired confidence interval (0-1)
            
        Returns:
            PredictionWithUncertainty object
        """
        all_samples = []
        
        for model in self.models:
            # Get multiple samples from each model
            model_samples = []
            for _ in range(n_samples):
                pred = model.predict(X, verbose=0)
                model_samples.append(pred)
            all_samples.extend(model_samples)
        
        all_samples = np.array(all_samples)  # [n_models * n_samples, batch_size, n_outputs]
        
        # Calculate statistics
        mean_pred = np.mean(all_samples, axis=0)
        std_pred = np.std(all_samples, axis=0)
        
        # Calculate confidence intervals
        alpha = 1.0 - confidence_interval
        lower = np.percentile(all_samples, 100 * alpha/2, axis=0)
        upper = np.percentile(all_samples, 100 * (1 - alpha/2), axis=0)
        
        return PredictionWithUncertainty(
            prediction=mean_pred,
            uncertainty=std_pred,
            lower_bound=lower,
            upper_bound=upper,
            samples=all_samples,
            confidence_interval=confidence_interval
        )
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform a single training step for all models in the ensemble.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary of training metrics (averaged across models)
        """
        metrics = {}
        
        for i, model in enumerate(self.models):
            # Train each model on a bootstrap sample of the data
            idx = np.random.choice(len(X), size=len(X), replace=True)
            X_boot = X[idx]
            y_boot = y[idx]
            
            # Train for one epoch
            history = model.fit(
                X_boot, y_boot,
                batch_size=min(32, len(X_boot)),
                epochs=1,
                verbose=0
            )
            
            # Track metrics
            for k, v in history.history.items():
                metrics[f"model_{i}_{k}"] = v[0]
        
        # Calculate average metrics
        avg_metrics = {}
        for k in history.history.keys():
            model_metrics = [v for kk, v in metrics.items() if k in kk]
            avg_metrics[k] = np.mean(model_metrics)
        
        return avg_metrics

class BayesianDenseLayer(tf.keras.layers.Layer):
    """Bayesian dense layer with weight uncertainty."""
    
    def __init__(
        self,
        units: int,
        prior_std: float = 1.0,
        posterior_std_init: float = 0.1,
        kl_weight: float = 1.0,
        activation: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the Bayesian dense layer.
        
        Args:
            units: Number of output units
            prior_std: Standard deviation of the prior distribution
            posterior_std_init: Initial standard deviation for the posterior
            kl_weight: Weight for the KL divergence term
            activation: Activation function
        """
        super().__init__(**kwargs)
        self.units = units
        self.prior_std = prior_std
        self.posterior_std_init = posterior_std_init
        self.kl_weight = kl_weight
        self.activation = tf.keras.activations.get(activation)
    
    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        # Define weight distributions
        self.kernel_mu = self.add_weight(
            name='kernel_mu',
            shape=(input_dim, self.units),
            initializer='glorot_normal',
            trainable=True
        )
        
        self.kernel_rho = self.add_weight(
            name='kernel_rho',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.Constant(
                np.log(np.exp(self.posterior_std_init) - 1.0)  # Softplus inverse
            ),
            trainable=True
        )
        
        self.bias_mu = self.add_weight(
            name='bias_mu',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        
        self.bias_rho = self.add_weight(
            name='bias_rho',
            shape=(self.units,),
            initializer=tf.keras.initializers.Constant(
                np.log(np.exp(self.posterior_std_init) - 1.0)  # Softplus inverse
            ),
            trainable=True
        )
        
        # Prior distribution (non-trainable)
        self.prior = tfp.distributions.Normal(0.0, self.prior_std)
        
        super().build(input_shape)
    
    def call(self, inputs, training=None):
        if training:
            # Sample weights from the posterior
            kernel_sigma = tf.nn.softplus(self.kernel_rho)
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(tf.shape(self.kernel_mu))
            
            bias_sigma = tf.nn.softplus(self.bias_rho)
            bias = self.bias_mu + bias_sigma * tf.random.normal(tf.shape(self.bias_mu))
            
            # Add KL divergence to loss
            posterior_kernel = tfp.distributions.Normal(self.kernel_mu, kernel_sigma)
            kl_kernel = tf.reduce_sum(tfp.distributions.kl_divergence(
                posterior_kernel, self.prior
            ))
            
            posterior_bias = tfp.distributions.Normal(self.bias_mu, bias_sigma)
            kl_bias = tf.reduce_sum(tfp.distributions.kl_divergence(
                posterior_bias, self.prior
            ))
            
            self.add_loss(self.kl_weight * (kl_kernel + kl_bias))
        else:
            # Use mean weights for prediction
            kernel = self.kernel_mu
            bias = self.bias_mu
        
        output = tf.matmul(inputs, kernel) + bias
        
        if self.activation is not None:
            output = self.activation(output)
            
        return output

class BayesianNeuralNetwork(UncertaintyModel):
    """
    Bayesian Neural Network for uncertainty estimation.
    
    Implements a BNN using variational inference with reparameterization trick.
    """
    
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        hidden_units: List[int] = [128, 64],
        activation: str = 'relu',
        learning_rate: float = 1e-3,
        kl_weight: float = 1.0,
        prior_std: float = 1.0,
        posterior_std_init: float = 0.1
    ):
        """
        Initialize the Bayesian Neural Network.
        
        Args:
            input_shape: Shape of input features
            hidden_units: List of hidden layer sizes
            activation: Activation function for hidden layers
            learning_rate: Learning rate for optimizer
            kl_weight: Weight for KL divergence term
            prior_std: Standard deviation of the prior distribution
            posterior_std_init: Initial standard deviation for the posterior
        """
        self.input_shape = input_shape
        self.hidden_units = hidden_units
        self.activation = activation
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.prior_std = prior_std
        self.posterior_std_init = posterior_std_init
        
        # Build the model
        self.model = self._build_model()
        self._compile_model()
    
    def _build_model(self) -> tf.keras.Model:
        """Build the BNN model."""
        inputs = tf.keras.layers.Input(shape=self.input_shape)
        x = inputs
        
        # Hidden layers
        for units in self.hidden_units:
            x = BayesianDenseLayer(
                units=units,
                prior_std=self.prior_std,
                posterior_std_init=self.posterior_std_init,
                kl_weight=self.kl_weight,
                activation=self.activation
            )(x)
        
        # Output layer (single output for regression)
        outputs = BayesianDenseLayer(
            units=1,
            prior_std=self.prior_std,
            posterior_std_init=self.posterior_std_init,
            kl_weight=self.kl_weight,
            activation=None
        )(x)
        
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    
    def _compile_model(self) -> None:
        """Compile the model with appropriate loss and metrics."""
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae']
        )
    
    def predict_with_uncertainty(
        self, 
        X: np.ndarray, 
        n_samples: int = 100,
        confidence_interval: float = 0.95
    ) -> PredictionWithUncertainty:
        """
        Make predictions with uncertainty estimates using the BNN.
        
        Args:
            X: Input features
            n_samples: Number of forward passes
            confidence_interval: Desired confidence interval (0-1)
            
        Returns:
            PredictionWithUncertainty object
        """
        # Generate multiple predictions
        samples = []
        for _ in range(n_samples):
            pred = self.model.predict(X, verbose=0)
            samples.append(pred)
        
        samples = np.array(samples)  # [n_samples, batch_size, 1]
        samples = np.squeeze(samples, axis=-1).T  # [batch_size, n_samples]
        
        # Calculate statistics
        mean_pred = np.mean(samples, axis=1, keepdims=True)
        std_pred = np.std(samples, axis=1, keepdims=True)
        
        # Calculate confidence intervals
        alpha = 1.0 - confidence_interval
        lower = np.percentile(samples, 100 * alpha/2, axis=1, keepdims=True)
        upper = np.percentile(samples, 100 * (1 - alpha/2), axis=1, keepdims=True)
        
        return PredictionWithUncertainty(
            prediction=mean_pred,
            uncertainty=std_pred,
            lower_bound=lower,
            upper_bound=upper,
            samples=samples.T[:, :, np.newaxis],  # [n_samples, batch_size, 1]
            confidence_interval=confidence_interval
        )
    
    def train_step(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            X: Input features
            y: Target values
            
        Returns:
            Dictionary of training metrics
        """
        history = self.model.fit(
            X, y,
            batch_size=min(32, len(X)),
            epochs=1,
            verbose=0
        )
        
        return {k: v[0] for k, v in history.history.items()}

def create_uncertainty_model(
    model_type: str = 'mc_dropout',
    **kwargs
) -> UncertaintyModel:
    """
    Factory function to create an uncertainty-aware model.
    
    Args:
        model_type: Type of uncertainty model ('mc_dropout', 'ensemble', or 'bnn')
        **kwargs: Additional arguments for the specific model
        
    Returns:
        An instance of the specified uncertainty model
    """
    if model_type == 'mc_dropout':
        return MCDropoutModel(**kwargs)
    elif model_type == 'ensemble':
        return DeepEnsemble(**kwargs)
    elif model_type == 'bnn':
        return BayesianNeuralNetwork(**kwargs)
    else:
        raise ValueError(f"Unknown uncertainty model type: {model_type}")
