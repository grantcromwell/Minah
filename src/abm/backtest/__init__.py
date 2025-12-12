"""
Backtesting module for the Agent-Based Modeling trading system.

This module provides the core functionality for backtesting trading strategies
using historical market data and simulating order execution in a realistic manner.
"""

from .engine import BacktestEngine
from .results import BacktestResult, PerformanceMetrics
from .strategies import Strategy, ParameterSet
from .data import HistoricalDataHandler

__all__ = [
    'BacktestEngine',
    'BacktestResult',
    'PerformanceMetrics',
    'Strategy',
    'ParameterSet',
    'HistoricalDataHandler'
]
