"""
Backtesting engine for evaluating trading strategies.

This module implements a high-performance backtesting engine that can simulate
the execution of trading strategies on historical market data. It handles order
routing, position tracking, and performance calculation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from datetime import datetime, timedelta
import logging

from ..interfaces.order_book import OrderBook, Order, OrderType, OrderSide, OrderStatus
from ..execution.engine import ExecutionEngine
from .data import HistoricalDataHandler
from .strategies import Strategy
from .results import BacktestResult, PerformanceMetrics

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting engine for evaluating trading strategies on historical data.
    
    The engine simulates the execution of trading strategies by processing
    historical market data in a realistic manner, accounting for market impact,
    transaction costs, and order execution logic.
    """
    
    def __init__(
        self,
        data_handler: HistoricalDataHandler,
        initial_cash: float = 1_000_000.0,
        start_date: Optional[Union[str, datetime]] = None,
        end_date: Optional[Union[str, datetime]] = None,
        benchmark: str = 'SPY',
        slippage_model: Optional[Callable] = None,
        commission_model: Optional[Callable] = None,
        random_seed: Optional[int] = None
    ):
        """
        Initialize the backtesting engine.
        
        Args:
            data_handler: Historical data handler instance
            initial_cash: Initial cash balance for the backtest
            start_date: Start date of the backtest (default: earliest available data)
            end_date: End date of the backtest (default: latest available data)
            benchmark: Benchmark symbol for performance comparison
            slippage_model: Function to model slippage (default: volume-weighted)
            commission_model: Function to calculate commissions (default: fixed per-share)
            random_seed: Random seed for reproducibility
        """
        self.data_handler = data_handler
        self.initial_cash = initial_cash
        self.benchmark = benchmark
        self.random_seed = random_seed
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize execution engine with empty order books
        self.order_books = {}  # Will be populated with order books for each symbol
        self.execution_engine = ExecutionEngine(self.order_books)
        
        # Set default slippage and commission models if not provided
        self.slippage_model = slippage_model or self._default_slippage_model
        self.commission_model = commission_model or self._default_commission_model
        
        # Set date range
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None
        
        # Initialize state
        self.current_time = None
        self.positions = {}
        self.cash = initial_cash
        self.orders = []
        self.trades = []
        self.equity_curve = []
        self.strategy = None
        
        # Performance metrics
        self.metrics = {}
    
    def _default_slippage_model(self, order: Order, current_price: float, volume: float) -> float:
        """Default slippage model based on order size and volume."""
        if volume == 0:
            return 0.0
            
        # Calculate market impact based on order size relative to average volume
        impact = 0.1 * (order.quantity / volume)  # 10bps impact per 1% of ADV
        return current_price * impact
    
    def _default_commission_model(self, order: Order) -> float:
        """Default commission model: $0.01 per share with $1 minimum."""
        commission = max(1.0, abs(order.quantity) * 0.01)
        return commission if order.side == OrderSide.SELL else -commission
    
    def add_strategy(self, strategy_class: type, **strategy_params) -> None:
        """
        Add a trading strategy to the backtest.
        
        Args:
            strategy_class: Strategy class (must inherit from Strategy)
            **strategy_params: Parameters to pass to the strategy constructor
        """
        if not issubclass(strategy_class, Strategy):
            raise ValueError("Strategy must be a subclass of Strategy")
            
        self.strategy = strategy_class(
            data_handler=self.data_handler,
            execution_engine=self.execution_engine,
            **strategy_params
        )
    
    def run(self) -> BacktestResult:
        """
        Run the backtest and return the results.
        
        Returns:
            BacktestResult: Object containing backtest results and performance metrics
        """
        if not self.strategy:
            raise ValueError("No strategy added to the backtest")
        
        logger.info(f"Starting backtest from {self.start_date} to {self.end_date}")
        
        # Initialize data handler with date range
        self.data_handler.set_date_range(self.start_date, self.end_date)
        
        # Main backtest loop
        for timestamp, data_slice in self.data_handler:
            self.current_time = timestamp
            
            # Update order books with new market data
            self._update_order_books(data_slice)
            
            # Update strategy with new data
            self.strategy.on_data(data_slice)
            
            # Execute any pending orders
            self._process_orders()
            
            # Update portfolio value
            self._update_equity_curve()
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Generate and return results
        return self._generate_results()
    
    def _update_order_books(self, data_slice: Dict[str, Any]) -> None:
        """Update order books with new market data."""
        for symbol, data in data_slice.items():
            if symbol not in self.order_books:
                self.order_books[symbol] = OrderBook(symbol)
            
            # Update order book with new price and volume
            self.order_books[symbol].update(
                bid_price=data.get('bid', data.get('close')),
                ask_price=data.get('ask', data.get('close')),
                bid_size=data.get('bid_size', 1000),
                ask_size=data.get('ask_size', 1000),
                timestamp=self.current_time
            )
    
    def _process_orders(self) -> None:
        """Process all pending orders through the execution engine."""
        # Process orders through the execution engine
        self.execution_engine.run()
        
        # Get filled orders and update portfolio
        filled_orders = self.execution_engine.get_filled_orders()
        
        for order in filled_orders:
            self._process_fill(order)
    
    def _process_fill(self, order: Order) -> None:
        """Process a filled order and update portfolio state."""
        # Calculate fill price with slippage
        fill_price = order.price
        if order.order_type != OrderType.MARKET:
            # For limit orders, use the limit price as the fill price
            fill_price = order.price
        
        # Calculate commission
        commission = self.commission_model(order)
        
        # Calculate position delta
        position_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
        
        # Update positions
        if order.symbol not in self.positions:
            self.positions[order.symbol] = 0.0
        
        # Update cash and positions
        self.cash -= (fill_price * order.quantity) + commission
        self.positions[order.symbol] += position_delta
        
        # Record the trade
        trade = {
            'timestamp': self.current_time,
            'symbol': order.symbol,
            'quantity': order.quantity,
            'price': fill_price,
            'side': order.side.name,
            'order_type': order.order_type.name,
            'commission': commission,
            'slippage': abs(fill_price - order.price) * order.quantity if order.order_type != OrderType.MARKET else 0.0
        }
        self.trades.append(trade)
        
        logger.debug(f"Filled {order.side.name} order for {order.quantity} shares of {order.symbol} "
                    f"at {fill_price:.2f} (commission: {commission:.2f})")
    
    def _update_equity_curve(self) -> None:
        """Update the equity curve with the current portfolio value."""
        # Calculate current portfolio value
        positions_value = 0.0
        
        for symbol, quantity in self.positions.items():
            if quantity == 0:
                continue
                
            # Get current market price
            if symbol in self.order_books:
                price = self.order_books[symbol].get_mid_price()
                positions_value += price * quantity
        
        # Record equity curve point
        self.equity_curve.append({
            'timestamp': self.current_time,
            'cash': self.cash,
            'positions_value': positions_value,
            'total': self.cash + positions_value,
            'return': (self.cash + positions_value) / self.initial_cash - 1.0
        })
    
    def _calculate_metrics(self) -> None:
        """Calculate performance metrics from the backtest results."""
        if not self.equity_curve:
            return
        
        # Convert to DataFrame for easier calculations
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        
        # Calculate returns
        equity_df['daily_return'] = equity_df['total'].pct_change().fillna(0)
        
        # Basic metrics
        total_return = equity_df['total'].iloc[-1] / equity_df['total'].iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(equity_df)) - 1
        annual_volatility = equity_df['daily_return'].std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * equity_df['daily_return'].mean() / (equity_df['daily_return'].std() + 1e-10)
        
        # Drawdown calculation
        equity_df['cummax'] = equity_df['total'].cummax()
        equity_df['drawdown'] = (equity_df['total'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # Trade statistics
        trades_df = pd.DataFrame(self.trades)
        if not trades_df.empty:
            trades_df['pnl'] = np.where(
                trades_df['side'] == 'BUY',
                -trades_df['quantity'] * trades_df['price'] - trades_df['commission'],
                trades_df['quantity'] * trades_df['price'] - trades_df['commission']
            )
            
            winning_trades = trades_df[trades_df['pnl'] > 0]
            losing_trades = trades_df[trades_df['pnl'] <= 0]
            
            win_rate = len(winning_trades) / len(trades_df) if len(trades_df) > 0 else 0
            avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Store metrics
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(trades_df),
            'final_value': equity_df['total'].iloc[-1],
            'final_cash': self.cash,
            'final_positions_value': equity_df['positions_value'].iloc[-1],
        }
    
    def _generate_results(self) -> 'BacktestResult':
        """Generate a BacktestResult object from the backtest data."""
        # Create performance metrics
        metrics = PerformanceMetrics(
            total_return=self.metrics['total_return'],
            annual_return=self.metrics['annual_return'],
            annual_volatility=self.metrics['annual_volatility'],
            sharpe_ratio=self.metrics['sharpe_ratio'],
            max_drawdown=self.metrics['max_drawdown'],
            win_rate=self.metrics['win_rate'],
            profit_factor=self.metrics['profit_factor'],
            num_trades=self.metrics['num_trades']
        )
        
        # Create trades DataFrame
        trades_df = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        
        # Create equity curve DataFrame
        equity_df = pd.DataFrame(self.equity_curve).set_index('timestamp')
        
        # Create positions snapshot
        positions = self.positions.copy()
        
        # Create and return result object
        return BacktestResult(
            metrics=metrics,
            trades=trades_df,
            equity_curve=equity_df,
            positions=positions,
            parameters=self.strategy.get_parameters() if hasattr(self.strategy, 'get_parameters') else {}
        )
