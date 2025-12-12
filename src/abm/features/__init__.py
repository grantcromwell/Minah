""
Feature Engineering Pipeline for Financial Time Series

This module provides a flexible and extensible pipeline for creating and managing
features from financial time series data.
"""

from .pipeline import FeaturePipeline, FeatureTransformer
from .transformers import *
from .utils import create_lag_features, create_rolling_features, create_ta_features

__all__ = [
    'FeaturePipeline',
    'FeatureTransformer',
    'create_lag_features',
    'create_rolling_features',
    'create_ta_features'
]
