"""
Strategy definitions for backtesting.

This module provides the base class for implementing trading strategies
and parameter optimization functionality.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Type
import numpy as np
import pandas as pd
from ..interfaces.order_book import Order, OrderType, OrderSide
from ..execution.engine import ExecutionEngine
from .data import HistoricalDataHandler

@dataclass
class ParameterSet:
    """
    Represents a set of parameters for a trading strategy.
    
    This class provides a convenient way to define, validate, and optimize
    strategy parameters.
    """
    params: Dict[str, Union[int, float, str, bool]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        self._validate()
    
    def _validate(self) -> None:
        """Validate the parameter values."""
        # Default implementation does nothing
        # Subclasses should override this to implement parameter validation
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the parameter set to a dictionary."""
        return self.params.copy()
    
    @classmethod
    def from_dict(cls, params: Dict[str, Any]) -> 'ParameterSet':
        """Create a parameter set from a dictionary."""
        return cls(params=params)
    
    def update(self, params: Dict[str, Any]) -> None:
        """Update parameters from a dictionary."""
        self.params.update(params)
        self._validate()
    
    def get_optimization_grid(self) -> Dict[str, List[Any]]:
        """
        Get a grid of parameter values for optimization.
        
        Returns:
            Dictionary mapping parameter names to lists of values to try
        """
        # Default implementation returns an empty grid
        # Subclasses should override this to define parameter search spaces
        return {}


class Strategy(ABC):
    """
    Abstract base class for trading strategies.
    
    This class defines the interface that all trading strategies must implement.
    Subclasses should override the abstract methods to implement specific strategies.
    """
    
    def __init__(
        self,
        data_handler: HistoricalDataHandler,
        execution_engine: ExecutionEngine,
        initial_capital: float = 1_000_000.0,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the strategy.
        
        Args:
            data_handler: Historical data handler
            execution_engine: Execution engine for order management
            initial_capital: Initial capital for the strategy
            parameters: Strategy parameters
            **kwargs: Additional keyword arguments
        """
        self.data_handler = data_handler
        self.execution_engine = execution_engine
        self.initial_capital = initial_capital
        self.parameters = parameters or {}
        self.kwargs = kwargs
        
        # State variables
        self.current_time = None
        self.positions = {}
        self.cash = initial_capital
        self.equity = initial_capital
        self.trades = []
        self.signals = []
        
        # Initialize the strategy
        self.initialize()
    
    def initialize(self) -> None:
        """
        Initialize the strategy.
        
        This method is called once when the strategy is created.
        Override this method to perform any one-time initialization.
        """
        pass
    
    @abstractmethod
    def on_data(self, data: Dict[str, Any]) -> None:
        """
        Process new market data.
        
        This method is called for each new data point. Subclasses must implement
        this method to define the strategy's trading logic.
        
        Args:
            data: Dictionary mapping symbols to their latest market data
        """
        pass
    
    def on_order_filled(self, order: Order) -> None:
        """
        Handle an order fill event.
        
        This method is called whenever an order is filled. Subclasses can
        override this method to implement custom order fill handling.
        
        Args:
            order: The filled order
        """
        pass
    
    def on_order_rejected(self, order: Order, reason: str) -> None:
        """
        Handle an order rejection event.
        
        This method is called whenever an order is rejected. Subclasses can
        override this method to implement custom rejection handling.
        
        Args:
            order: The rejected order
            reason: Reason for rejection
        """
        logger.warning(f"Order rejected: {order.order_id} - {reason}")
    
    def submit_order(
        self,
        symbol: str,
        quantity: float,
        order_type: OrderType = OrderType.MARKET,
        side: OrderSide = OrderSide.BUY,
        price: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Submit a new order.
        
        Args:
            symbol: Symbol to trade
            quantity: Number of shares/contracts to trade
            order_type: Type of order (MARKET, LIMIT, etc.)
            side: BUY or SELL
            price: Limit price (required for LIMIT orders)
            **kwargs: Additional order parameters
            
        Returns:
            Order ID if the order was submitted successfully, None otherwise
        """
        # Create the order
        order = Order(
            symbol=symbol,
            quantity=quantity,
            order_type=order_type,
            side=side,
            price=price,
            **kwargs
        )
        
        # Submit the order through the execution engine
        result = self.execution_engine.submit_order(order)
        
        # Log the order submission
        logger.debug(f"Submitted {side.name} order for {quantity} shares of {symbol} "
                   f"(type: {order_type.name}, price: {price or 'market'})")
        
        return result.order_id if result else None
    
    def get_position(self, symbol: str) -> float:
        """
        Get the current position for a symbol.
        
        Args:
            symbol: Symbol to get position for
            
        Returns:
            Current position (positive for long, negative for short, 0 for flat)
        """
        return self.positions.get(symbol, 0.0)
    
    def get_positions(self) -> Dict[str, float]:
        """
        Get all current positions.
        
        Returns:
            Dictionary mapping symbols to positions
        """
        return self.positions.copy()
    
    def get_cash(self) -> float:
        """
        Get the current cash balance.
        
        Returns:
            Current cash balance
        """
        return self.cash
    
    def get_equity(self) -> float:
        """
        Get the current total equity (cash + positions value).
        
        Returns:
            Current total equity
        """
        return self.equity
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current strategy parameters.
        
        Returns:
            Dictionary of strategy parameters
        """
        return self.parameters.copy()
    
    def set_parameters(self, parameters: Dict[str, Any]) -> None:
        """
        Update strategy parameters.
        
        Args:
            parameters: Dictionary of parameter names and values
        """
        self.parameters.update(parameters)
    
    def get_signals(self) -> pd.DataFrame:
        """
        Get the strategy's trading signals.
        
        Returns:
            DataFrame containing the strategy's signals
        """
        if not self.signals:
            return pd.DataFrame()
        
        return pd.DataFrame(self.signals).set_index('timestamp')
    
    def get_trades(self) -> pd.DataFrame:
        """
        Get the strategy's trade history.
        
        Returns:
            DataFrame containing the strategy's trade history
        """
        if not self.trades:
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def _update_positions(self, order: Order, fill_price: float) -> None:
        """
        Update positions based on a filled order.
        
        Args:
            order: The filled order
            fill_price: Price at which the order was filled
        """
        symbol = order.symbol
        quantity = order.quantity if order.side == OrderSide.BUY else -order.quantity
        
        # Update position
        if symbol not in self.positions:
            self.positions[symbol] = 0.0
        
        self.positions[symbol] += quantity
        
        # Update cash
        self.cash -= quantity * fill_price
        
        # Record the trade
        self.trades.append({
            'timestamp': self.current_time,
            'symbol': symbol,
            'quantity': quantity,
            'price': fill_price,
            'side': order.side.name,
            'order_type': order.order_type.name,
            'value': abs(quantity * fill_price)
        })
        
        # Record the signal
        self.signals.append({
            'timestamp': self.current_time,
            'symbol': symbol,
            'signal': 1 if quantity > 0 else -1,
            'price': fill_price,
            'quantity': abs(quantity)
        })
        
        # Update equity (simplified - in a real implementation, we'd use mark-to-market)
        self.equity = self.cash + sum(
            self.positions[sym] * self._get_current_price(sym)
            for sym in self.positions
        )
    
    def _get_current_price(self, symbol: str) -> float:
        """
        Get the current price for a symbol.
        
        Args:
            symbol: Symbol to get price for
            
        Returns:
            Current price, or 0 if not available
        """
        # This is a simplified implementation
        # In a real implementation, we'd get the current price from the data handler
        return 0.0


class MovingAverageCrossover(Strategy):
    """
    Simple moving average crossover strategy.
    
    This strategy generates buy signals when the fast moving average crosses
    above the slow moving average, and sell signals when it crosses below.
    """
    
    def __init__(
        self,
        data_handler: HistoricalDataHandler,
        execution_engine: ExecutionEngine,
        initial_capital: float = 1_000_000.0,
        fast_window: int = 20,
        slow_window: int = 50,
        **kwargs
    ):
        """
        Initialize the strategy.
        
        Args:
            data_handler: Historical data handler
            execution_engine: Execution engine for order management
            initial_capital: Initial capital for the strategy
            fast_window: Window for fast moving average
            slow_window: Window for slow moving average
            **kwargs: Additional keyword arguments
        """
        parameters = {
            'fast_window': fast_window,
            'slow_window': slow_window,
            **kwargs
        }
        
        super().__init__(
            data_handler=data_handler,
            execution_engine=execution_engine,
            initial_capital=initial_capital,
            parameters=parameters
        )
        
        # State variables
        self.fast_ma = {}
        self.slow_ma = {}
        self.prev_fast_ma = {}
        self.prev_slow_ma = {}
    
    def initialize(self) -> None:
        """Initialize the strategy."""
        # Initialize moving averages for each symbol
        for symbol in self.data_handler.get_symbols():
            self.fast_ma[symbol] = []
            self.slow_ma[symbol] = []
            self.prev_fast_ma[symbol] = None
            self.prev_slow_ma[symbol] = None
    
    def on_data(self, data: Dict[str, Any]) -> None:
        """
        Process new market data.
        
        Args:
            data: Dictionary mapping symbols to their latest market data
        """
        for symbol, bar in data.items():
            if symbol not in self.fast_ma:
                continue
            
            # Update moving averages
            close_price = bar['close']
            
            # Fast MA
            self.fast_ma[symbol].append(close_price)
            if len(self.fast_ma[symbol]) > self.parameters['fast_window']:
                self.fast_ma[symbol].pop(0)
            
            # Slow MA
            self.slow_ma[symbol].append(close_price)
            if len(self.slow_ma[symbol]) > self.parameters['slow_window']:
                self.slow_ma[symbol].pop(0)
            
            # Check if we have enough data
            if (len(self.fast_ma[symbol]) < self.parameters['fast_window'] or
                len(self.slow_ma[symbol]) < self.parameters['slow_window']):
                continue
            
            # Calculate moving averages
            fast_ma = np.mean(self.fast_ma[symbol])
            slow_ma = np.mean(self.slow_ma[symbol])
            
            # Get previous values for crossover detection
            prev_fast = self.prev_fast_ma[symbol] if self.prev_fast_ma[symbol] is not None else fast_ma
            prev_slow = self.prev_slow_ma[symbol] if self.prev_slow_ma[symbol] is not None else slow_ma
            
            # Check for crossover
            if prev_fast <= prev_slow and fast_ma > slow_ma:
                # Bullish crossover - buy signal
                position = self.get_position(symbol)
                if position <= 0:  # Only buy if we're not already long
                    # Calculate position size (simplified)
                    cash = self.get_cash()
                    price = bar['close']
                    quantity = int((cash * 0.1) / price)  # 10% of cash
                    
                    if quantity > 0:
                        self.submit_order(
                            symbol=symbol,
                            quantity=quantity,
                            order_type=OrderType.MARKET,
                            side=OrderSide.BUY
                        )
            
            elif prev_fast >= prev_slow and fast_ma < slow_ma:
                # Bearish crossover - sell signal
                position = self.get_position(symbol)
                if position > 0:  # Only sell if we're long
                    self.submit_order(
                        symbol=symbol,
                        quantity=position,
                        order_type=OrderType.MARKET,
                        side=OrderSide.SELL
                    )
            
            # Update previous values
            self.prev_fast_ma[symbol] = fast_ma
            self.prev_slow_ma[symbol] = slow_ma
