""
Feature Pipeline Implementation

This module implements a flexible feature engineering pipeline for financial time series data.
"""

from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)

class FeatureTransformer(ABC):
    """Abstract base class for all feature transformers."""
    
    def __init__(self, name: Optional[str] = None):
        """
        Initialize the feature transformer.
        
        Args:
            name: Optional name for the transformer
        """
        self.name = name or self.__class__.__name__
        self.fitted = False
    
    @abstractmethod
    def fit(self, data: pd.DataFrame, **kwargs) -> 'FeatureTransformer':
        """
        Fit the transformer on the data.
        
        Args:
            data: Input DataFrame with time series data
            **kwargs: Additional arguments for fitting
            
        Returns:
            self: Returns the transformer instance
        """
        self.fitted = True
        return self
    
    @abstractmethod
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the data using the fitted transformer.
        
        Args:
            data: Input DataFrame with time series data
            **kwargs: Additional arguments for transformation
            
        Returns:
            DataFrame: Transformed data with new features
        """
        pass
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit the transformer and transform the data.
        
        Args:
            data: Input DataFrame with time series data
            **kwargs: Additional arguments for fitting and transformation
            
        Returns:
            DataFrame: Transformed data with new features
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of the features created by this transformer.
        
        Returns:
            List of feature names
        """
        return []


class FeaturePipeline:
    """
    A pipeline for chaining multiple feature transformers.
    
    This class allows you to chain multiple feature transformers together
    and apply them sequentially to the input data.
    """
    
    def __init__(self, steps: List[FeatureTransformer] = None):
        """
        Initialize the feature pipeline.
        
        Args:
            steps: List of feature transformers
        """
        self.steps = steps or []
        self.fitted = False
    
    def add_step(self, transformer: FeatureTransformer) -> 'FeaturePipeline':
        """
        Add a transformer to the pipeline.
        
        Args:
            transformer: Feature transformer to add
            
        Returns:
            self: Returns the pipeline instance
        """
        self.steps.append(transformer)
        return self
    
    def fit(self, data: pd.DataFrame, **kwargs) -> 'FeaturePipeline':
        """
        Fit all transformers in the pipeline.
        
        Args:
            data: Input DataFrame with time series data
            **kwargs: Additional arguments for fitting
            
        Returns:
            self: Returns the pipeline instance
        """
        current_data = data.copy()
        
        for step in self.steps:
            logger.info(f"Fitting transformer: {step.name}")
            step.fit(current_data, **kwargs)
            current_data = step.transform(current_data, **kwargs)
        
        self.fitted = True
        return self
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Transform the data using all fitted transformers.
        
        Args:
            data: Input DataFrame with time series data
            **kwargs: Additional arguments for transformation
            
        Returns:
            DataFrame: Transformed data with all features
        """
        if not self.fitted:
            raise RuntimeError("Pipeline not fitted. Call fit() first.")
        
        current_data = data.copy()
        
        for step in self.steps:
            logger.debug(f"Applying transformer: {step.name}")
            current_data = step.transform(current_data, **kwargs)
        
        return current_data
    
    def fit_transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Fit all transformers and transform the data.
        
        Args:
            data: Input DataFrame with time series data
            **kwargs: Additional arguments for fitting and transformation
            
        Returns:
            DataFrame: Transformed data with all features
        """
        return self.fit(data, **kwargs).transform(data, **kwargs)
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of all features created by the pipeline.
        
        Returns:
            List of all feature names
        """
        feature_names = []
        
        for step in self.steps:
            step_features = step.get_feature_names()
            if step_features:
                feature_names.extend(step_features)
        
        return feature_names
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from all transformers that support it.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        importances = {}
        
        for step in self.steps:
            if hasattr(step, 'feature_importances_'):
                step_importances = dict(zip(step.get_feature_names(), step.feature_importances_))
                importances.update(step_importances)
        
        return importances
    
    def __len__(self) -> int:
        """Get the number of steps in the pipeline."""
        return len(self.steps)
    
    def __getitem__(self, index: int) -> FeatureTransformer:
        """Get a transformer by index."""
        return self.steps[index]
