"""
Results and performance metrics for backtesting.

This module provides classes for storing and analyzing backtest results,
including performance metrics and statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime


@dataclass
class PerformanceMetrics:
    """
    Performance metrics for a backtest.

    Contains various performance measures calculated from backtest results.
    """
    total_return: float = 0.0
    annual_return: float = 0.0
    annual_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    num_trades: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            'total_return': self.total_return,
            'annual_return': self.annual_return,
            'annual_volatility': self.annual_volatility,
            'sharpe_ratio': self.sharpe_ratio,
            'max_drawdown': self.max_drawdown,
            'win_rate': self.win_rate,
            'profit_factor': self.profit_factor,
            'num_trades': self.num_trades
        }


@dataclass
class BacktestResult:
    """
    Results of a backtest run.

    Contains all the data from a backtest including trades, equity curve,
    performance metrics, and strategy parameters.
    """
    metrics: PerformanceMetrics
    trades: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    positions: Dict[str, float] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def __post_init__(self):
        """Set start and end times from equity curve."""
        if not self.equity_curve.empty:
            self.start_time = self.equity_curve.index[0]
            self.end_time = self.equity_curve.index[-1]

    def get_total_return(self) -> float:
        """Get the total return percentage."""
        return self.metrics.total_return

    def get_sharpe_ratio(self) -> float:
        """Get the Sharpe ratio."""
        return self.metrics.sharpe_ratio

    def get_max_drawdown(self) -> float:
        """Get the maximum drawdown percentage."""
        return self.metrics.max_drawdown

    def get_win_rate(self) -> float:
        """Get the win rate percentage."""
        return self.metrics.win_rate

    def get_num_trades(self) -> int:
        """Get the number of trades."""
        return self.metrics.num_trades

    def get_equity_curve(self) -> pd.DataFrame:
        """Get the equity curve DataFrame."""
        return self.equity_curve

    def get_trades(self) -> pd.DataFrame:
        """Get the trades DataFrame."""
        return self.trades

    def get_positions(self) -> Dict[str, float]:
        """Get final positions."""
        return self.positions

    def get_parameters(self) -> Dict[str, Any]:
        """Get strategy parameters."""
        return self.parameters

    def to_dict(self) -> Dict[str, Any]:
        """Convert backtest result to dictionary."""
        return {
            'metrics': self.metrics.to_dict(),
            'trades': self.trades.to_dict() if not self.trades.empty else {},
            'equity_curve': self.equity_curve.to_dict() if not self.equity_curve.empty else {},
            'positions': self.positions,
            'parameters': self.parameters,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

    def summary(self) -> str:
        """Get a summary of the backtest results."""
        if self.equity_curve.empty:
            return "No backtest data available"

        return f"""
Backtest Results Summary:
========================
Period: {self.start_time} to {self.end_time}
Total Return: {self.metrics.total_return:.2%}
Annual Return: {self.metrics.annual_return:.2%}
Annual Volatility: {self.metrics.annual_volatility:.2%}
Sharpe Ratio: {self.metrics.sharpe_ratio:.2f}
Max Drawdown: {self.metrics.max_drawdown:.2%}
Win Rate: {self.metrics.win_rate:.2%}
Profit Factor: {self.metrics.profit_factor:.2f}
Number of Trades: {self.metrics.num_trades}
Final Value: ${self.equity_curve['total'].iloc[-1]:,.2f}
"""