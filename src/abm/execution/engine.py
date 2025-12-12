"""
Execution Engine for the Agent-Based Modeling system.

This module implements the core execution logic for order routing, smart order routing,
and transaction cost analysis.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pytz

from ..interfaces.order_book import OrderBook, Order, OrderType, OrderSide, OrderStatus
from ..data import MarketDataConnector, DataPipeline, MarketDataValidator
from ..data.config import get_config
from .algorithms.twap import TWAPExecutor
from .algorithms.vwap import VWAPExecutor
from .algorithms.iceberg import IcebergExecutor

logger = logging.getLogger(__name__)

class ExecutionStyle(Enum):
    """Execution styles for order placement."""
    MARKET = auto()
    LIMIT = auto()
    TWAP = auto()       # Time-Weighted Average Price
    VWAP = auto()       # Volume-Weighted Average Price
    POV = auto()        # Percentage of Volume
    ICEBERG = auto()    # Iceberg order
    HIDDEN = auto()     # Hidden order
    PEGGED = auto()     # Pegged order
    
class ExecutionResultStatus(Enum):
    """Status of an execution result."""
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    PENDING = auto()
    
@dataclass
class ExecutionResult:
    """Result of an order execution."""
    order_id: str
    status: ExecutionResultStatus
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_price: float = 0.0
    fees: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution result to a dictionary."""
        return {
            'order_id': self.order_id,
            'status': self.status.name,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'avg_price': self.avg_price,
            'fees': self.fees,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }

@dataclass
class ExecutionReport:
    """Report of an execution with detailed metrics."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = 'GTC'  # Good-Til-Cancelled
    strategy_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(pytz.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(pytz.utc))
    status: ExecutionResultStatus = ExecutionResultStatus.PENDING
    filled_quantity: float = 0.0
    remaining_quantity: float = 0.0
    avg_fill_price: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0
    impact_cost: float = 0.0
    vwap: float = 0.0
    twap: float = 0.0
    parent_order_id: Optional[str] = None
    child_orders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def update(self, fill_quantity: float, fill_price: float, fees: float = 0.0) -> None:
        """Update the execution report with a new fill."""
        if fill_quantity <= 0:
            return
            
        # Calculate new average fill price
        total_value = (self.avg_fill_price * self.filled_quantity) + (fill_price * fill_quantity)
        self.filled_quantity += fill_quantity
        self.remaining_quantity = max(0, self.quantity - self.filled_quantity)
        self.avg_fill_price = total_value / self.filled_quantity if self.filled_quantity > 0 else 0.0
        self.fees += fees
        self.updated_at = datetime.now(pytz.utc)
        
        # Update status
        if self.remaining_quantity <= 0:
            self.status = ExecutionResultStatus.FILLED
        else:
            self.status = ExecutionResultStatus.PARTIALLY_FILLED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the execution report to a dictionary."""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.name,
            'order_type': self.order_type.name,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'time_in_force': self.time_in_force,
            'strategy_id': self.strategy_id,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'status': self.status.name,
            'filled_quantity': self.filled_quantity,
            'remaining_quantity': self.remaining_quantity,
            'avg_fill_price': self.avg_fill_price,
            'fees': self.fees,
            'slippage': self.slippage,
            'impact_cost': self.impact_cost,
            'vwap': self.vwap,
            'twap': self.twap,
            'parent_order_id': self.parent_order_id,
            'child_orders': self.child_orders,
            'metadata': self.metadata
        }

class ExecutionEngine:
    """
    Execution engine for handling order routing, smart order routing,
    and transaction cost analysis.
    """
    
    def __init__(self, 
                 order_books: Dict[str, OrderBook],
                 data_connector: Optional[MarketDataConnector] = None,
                 data_pipeline: Optional[DataPipeline] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the execution engine.
        
        Args:
            order_books: Dictionary of order books by symbol
            data_connector: Market data connector for real-time data
            data_pipeline: Data pipeline for historical data and analytics
            config: Configuration dictionary
        """
        self.order_books = order_books
        self.data_connector = data_connector or MarketDataConnector()
        self.data_pipeline = data_pipeline or DataPipeline()
        self.config = config or {}
        
        # Active and completed orders
        self.active_orders: Dict[str, ExecutionReport] = {}
        self.completed_orders: Dict[str, ExecutionReport] = {}
        self.next_order_id = 1
        
        # Execution metrics
        self.metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
            'total_fees': 0.0,
            'slippage': 0.0,
            'latency': [],  # In seconds
            'algo_orders': {
                'twap': 0,
                'vwap': 0,
                'iceberg': 0
            },
            'risk_checks': {
                'passed': 0,
                'failed': 0,
                'rejections': 0
            }
        }
        
        # Initialize fee and slippage models
        self.fee_model = self._create_fee_model()
        self.slippage_model = self._create_slippage_model()
        self.market_impact_model = self._create_market_impact_model()
        
        # Enhanced risk management
        self.risk_limits = self.config.get('risk_limits', {
            'position_limits': {
                'max_position_size': 1000.0,  # Max position size per symbol
                'max_portfolio_exposure': 0.2,  # 20% of portfolio
                'max_leverage': 10.0,  # 10x leverage
                'max_drawdown': 0.1  # 10% max drawdown
            },
            'order_limits': {
                'max_order_size': 100.0,  # Max size per order
                'max_order_value': 10000.0,  # Max value per order
                'max_orders_per_minute': 100,  # Rate limiting
                'min_order_size': 0.01,  # Min size per order
                'price_band_pct': 0.05  # 5% price band from current price
            },
            'circuit_breakers': {
                'max_daily_loss_pct': 0.05,  # 5% max daily loss
                'max_position_concentration': 0.5,  # 50% max position concentration
                'max_sector_exposure': 0.3  # 30% max sector exposure
            }
        })
        
        self.position_limits = self.risk_limits.get('position_limits', {})
        self.order_limits = self.risk_limits.get('order_limits', {})
        self.circuit_breakers = self.risk_limits.get('circuit_breakers', {})
        
        # Track positions and exposure
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.exposure: Dict[str, float] = {}  # symbol -> notional value
        self.daily_pnl: Dict[str, float] = {}  # symbol -> PnL
        
        # Initialize execution algorithms with proper configuration
        algo_config = self.config.get('algorithms', {})
        self.twap_executor = TWAPExecutor(
            order_books, 
            self.data_connector, 
            algo_config.get('twap', {})
        )
        
        self.vwap_executor = VWAPExecutor(
            order_books, 
            self.data_connector, 
            self.data_pipeline,
            algo_config.get('vwap', {})
        )
        
        self.iceberg_executor = IcebergExecutor(
            order_books, 
            self.data_connector, 
            algo_config.get('iceberg', {})
        )
    
    def _get_average_trade_size(self, symbol: str) -> float:
        """Get the average trade size for a symbol."""
        # This would typically query historical trade data
        # For simplicity, we'll return a fixed value
        return 1.0  # Replace with actual implementation
    
    def _get_30d_volume(self, symbol: str) -> float:
        """Get the 30-day trading volume for a symbol."""
        # This would typically query historical volume data
        # For simplicity, we'll return a fixed value
        return 1000000.0  # $1M default
    
    def _get_average_daily_volume(self, symbol: str, lookback_days: int = 30) -> float:
        """Get the average daily volume for a symbol over the lookback period."""
        # This would typically query historical volume data
        # For simplicity, we'll return a fixed value
        return 10000.0  # 10,000 units per day default
    
    def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit a new order for execution.
        
        Args:
            order: Order to execute
            
        Returns:
            Execution result
        """
        start_time = time.time()
        report = ExecutionReport.from_order(order)
        
        # Check if symbol exists
        if order.symbol not in self.order_books:
            report.status = ExecutionResultStatus.REJECTED
            report.message = f"Symbol {order.symbol} not found"
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
            return report.to_result()
            
        # Check basic order parameters
        if order.quantity <= 0:
            report.status = ExecutionResultStatus.REJECTED
            report.message = "Order quantity must be positive"
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
            return report.to_result()
            
        if order.order_type == OrderType.LIMIT and order.price is None:
            report.status = ExecutionResultStatus.REJECTED
            report.message = "Limit orders require a price"
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
            return report.to_result()
        
        # Check risk limits
        risk_check = self._check_risk_limits(order)
        if not risk_check['allowed']:
            report.status = ExecutionResultStatus.REJECTED
            report.message = f"Risk check failed: {risk_check.get('reason', 'Unknown reason')}"
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
            self.metrics['risk_checks']['failed'] += 1
            return report.to_result()
        
        # Process based on order type
        try:
            if order.order_type == OrderType.MARKET:
                self._process_market_order(report)
            elif order.order_type == OrderType.LIMIT:
                self._process_limit_order(report)
            elif order.order_type == OrderType.STOP:
                self._process_stop_order(report)
            elif order.order_type == OrderType.STOP_LIMIT:
                self._process_stop_limit_order(report)
            elif order.order_type == OrderType.TWAP:
                self._process_twap_order(report)
            elif order.order_type == OrderType.VWAP:
                self._process_vwap_order(report)
            elif order.order_type == OrderType.ICEBERG:
                self._process_iceberg_order(report)
            else:
                report.status = ExecutionResultStatus.REJECTED
                report.message = f"Unsupported order type: {order.order_type}"
                self.metrics['rejected_orders'] += 1
        
        except Exception as e:
            logger.error(f"Error processing order {order.order_id}: {str(e)}", exc_info=True)
            report.status = ExecutionResultStatus.REJECTED
            report.message = f"Error processing order: {str(e)}"
            self.metrics['rejected_orders'] += 1
        
        # Update metrics
        latency = time.time() - start_time
        self.metrics['latency'].append(latency)
        self.metrics['total_orders'] += 1
        
        if report.status == ExecutionResultStatus.FILLED or report.status == ExecutionResultStatus.PARTIALLY_FILLED:
            self.metrics['filled_orders'] += 1
            self.metrics['total_volume'] += report.filled_quantity
            
            # Calculate fees and slippage
            if report.avg_price is not None:
                fees = self.fee_model(report.symbol, report.filled_quantity, report.avg_price, report.side)
                self.metrics['total_fees'] += fees
                
                # Estimate slippage
                if report.side == OrderSide.BUY:
                    slippage = (report.avg_price - report.price) * report.filled_quantity if report.price else 0
                else:
                    slippage = (report.price - report.avg_price) * report.filled_quantity if report.price else 0
                self.metrics['slippage'] += slippage
                
                # Update position and exposure
                self._update_position_and_exposure(report)
        
        # Store the report
        if report.status in [ExecutionResultStatus.FILLED, ExecutionResultStatus.PARTIALLY_FILLED, 
                           ExecutionResultStatus.CANCELLED, ExecutionResultStatus.REJECTED]:
            self.completed_orders[order.order_id] = report
            if order.order_id in self.active_orders:
                del self.active_orders[order.order_id]
        else:
            self.active_orders[order.order_id] = report
        
        return report.to_result()
                
    def _process_market_order(self, report: ExecutionReport) -> None:
        """Process a market order."""
        symbol = report.symbol
        quantity = report.remaining_quantity
        side = report.side
        
        if symbol not in self.order_books:
            self._reject_order(report, f"No order book for symbol: {symbol}")
            return
            
        order_book = self.order_books[symbol]
        
        try:
            # Estimate impact and slippage
            impact_pct, participation, _ = self.market_impact_model(symbol, quantity, side)
            slippage = self.slippage_model(symbol, quantity, side)
            
            # Get current mid price
            best_bid, best_ask = order_book.get_best_bid_ask()
            if best_bid is None or best_ask is None:
                self._reject_order(report, "No prices available in the order book")
                return
                
            mid_price = (best_bid + best_ask) / 2
            
            # Calculate expected price with impact and slippage
            if side == OrderSide.BUY:
                # For buys, we pay more than the best ask
                expected_price = best_ask * (1 + impact_pct + slippage)
            else:  # SELL
                # For sells, we receive less than the best bid
                expected_price = best_bid * (1 - impact_pct - slippage)
            
            # Simulate execution (in a real system, this would interact with the exchange)
            filled_qty = min(quantity, quantity * 0.95)  # Simulate partial fill
            filled_price = expected_price
            
            # Calculate fees
            fees = self.fee_model(symbol, filled_qty, filled_price, side)
            
            # Update execution report
            report.update(filled_qty, filled_price, fees)
            report.slippage = (filled_price - mid_price) / mid_price if side == OrderSide.BUY else \
                             (mid_price - filled_price) / mid_price
            report.impact_cost = impact_pct
            report.vwap = filled_price  # For market orders, VWAP = fill price
            report.twap = filled_price  # For market orders, TWAP = fill price
            
            # Update metrics
            self.metrics['total_volume'] += filled_qty * filled_price
            self.metrics['total_fees'] += fees
            
            # Log execution
            logger.info(f"Executed market {side.name} order {report.order_id}: "
                       f"{filled_qty} @ {filled_price:.8f} {symbol.split('/')[1]}")
            
            # If not fully filled, cancel the remaining
            if report.remaining_quantity > 0:
                self.cancel_order(report.order_id, "Partial fill")
            
        except Exception as e:
            logger.error(f"Error processing market order {report.order_id}: {str(e)}", exc_info=True)
            self._reject_order(report, f"Error: {str(e)}")
    
    def _process_limit_order(self, report: ExecutionReport) -> None:
        """Process a limit order."""
        # In a real implementation, this would add the order to the order book
        # and monitor for fills. For simulation, we'll assume immediate fill if possible.
        
        symbol = report.symbol
        quantity = report.remaining_quantity
        side = report.side
        price = report.price
        
        if price is None:
            self._reject_order(report, "Limit orders require a price")
            return
            
        if symbol not in self.order_books:
            self._reject_order(report, f"No order book for symbol: {symbol}")
            return
            
        order_book = self.order_books[symbol]
        
        try:
            # Check if the order can be filled immediately
            best_bid, best_ask = order_book.get_best_bid_ask()
            
            if best_bid is None or best_ask is None:
                # No prices available, add to order book
                self._add_to_order_book(report)
                return
                
            mid_price = (best_bid + best_ask) / 2
            
            # Check if the order can be filled immediately
            if (side == OrderSide.BUY and price >= best_ask) or \
               (side == OrderSide.SELL and price <= best_bid):
                # Immediate fill is possible
                filled_price = best_ask if side == OrderSide.BUY else best_bid
                filled_qty = min(quantity, quantity * 0.9)  # Simulate partial fill
                
                # Calculate fees
                fees = self.fee_model(symbol, filled_qty, filled_price, side)
                
                # Update execution report
                report.update(filled_qty, filled_price, fees)
                report.slippage = (filled_price - price) / price if side == OrderSide.BUY else \
                                 (price - filled_price) / price
                
                # For limit orders, impact is typically lower
                report.impact_cost = 0.0  # No impact for limit orders that are filled at or better than requested
                
                # Update metrics
                self.metrics['total_volume'] += filled_qty * filled_price
                self.metrics['total_fees'] += fees
                
                # Log execution
                logger.info(f"Filled limit {side.name} order {report.order_id}: "
                          f"{filled_qty} @ {filled_price:.8f} {symbol.split('/')[1]}")
                
                # If not fully filled, add the remaining to the order book
                if report.remaining_quantity > 0:
                    self._add_to_order_book(report)
            else:
                # Cannot be filled immediately, add to order book
                self._add_to_order_book(report)
                
        except Exception as e:
            logger.error(f"Error processing limit order {report.order_id}: {str(e)}", exc_info=True)
            self._reject_order(report, f"Error: {str(e)}")
    
    def _process_stop_order(self, report: ExecutionReport) -> None:
        """Process a stop order (stop-loss or take-profit)."""
        # In a real implementation, this would monitor the market price and trigger
        # a market order when the stop price is reached. For simulation, we'll assume
        # the stop is triggered immediately if the condition is met.
        
        symbol = report.symbol
        stop_price = report.stop_price
        
        if stop_price is None:
            self._reject_order(report, "Stop orders require a stop price")
            return
            
        if symbol not in self.order_books:
            self._reject_order(report, f"No order book for symbol: {symbol}")
            return
            
        order_book = self.order_books[symbol]
        
        try:
            # Get current market price
            best_bid, best_ask = order_book.get_best_bid_ask()
            
            if best_bid is None or best_ask is None:
                self._reject_order(report, "No prices available in the order book")
                return
                
            current_price = best_ask if report.side == OrderSide.BUY else best_bid
            
            # Check if stop condition is met
            stop_triggered = False
            if report.order_type == OrderType.STOP_LOSS:
                # For buy stop-loss, trigger when price >= stop_price
                # For sell stop-loss, trigger when price <= stop_price
                stop_triggered = (report.side == OrderSide.BUY and current_price >= stop_price) or \
                               (report.side == OrderSide.SELL and current_price <= stop_price)
            elif report.order_type == OrderType.TAKE_PROFIT:
                # For buy take-profit, trigger when price <= stop_price
                # For sell take-profit, trigger when price >= stop_price
                stop_triggered = (report.side == OrderSide.BUY and current_price <= stop_price) or \
                               (report.side == OrderSide.SELL and current_price >= stop_price)
            
            if stop_triggered:
                # Convert to market order and execute
                logger.info(f"Stop order {report.order_id} triggered at price {current_price}")
                self._process_market_order(report)
            else:
                # Add to active orders for monitoring
                logger.info(f"Stop order {report.order_id} not triggered (current: {current_price}, stop: {stop_price})")
                
        except Exception as e:
            logger.error(f"Error processing stop order {report.order_id}: {str(e)}", exc_info=True)
            self._reject_order(report, f"Error: {str(e)}")
    
    def _process_stop_limit_order(self, report: ExecutionReport) -> None:
        """Process a stop-limit order (stop-loss-limit or take-profit-limit)."""
        # Similar to stop orders, but triggers a limit order instead of a market order
        
        symbol = report.symbol
        stop_price = report.stop_price
        limit_price = report.price
        
        if stop_price is None or limit_price is None:
            self._reject_order(report, "Stop-limit orders require both stop and limit prices")
            return
            
        if symbol not in self.order_books:
            self._reject_order(report, f"No order book for symbol: {symbol}")
            return
            
        order_book = self.order_books[symbol]
        
        try:
            # Get current market price
            best_bid, best_ask = order_book.get_best_bid_ask()
            
            if best_bid is None or best_ask is None:
                self._reject_order(report, "No prices available in the order book")
                return
                
            current_price = best_ask if report.side == OrderSide.BUY else best_bid
            
            # Check if stop condition is met
            stop_triggered = False
            if report.order_type == OrderType.STOP_LOSS_LIMIT:
                # For buy stop-limit, trigger when price >= stop_price
                # For sell stop-limit, trigger when price <= stop_price
                stop_triggered = (report.side == OrderSide.BUY and current_price >= stop_price) or \
                               (report.side == OrderSide.SELL and current_price <= stop_price)
            elif report.order_type == OrderType.TAKE_PROFIT_LIMIT:
                # For buy take-profit-limit, trigger when price <= stop_price
                # For sell take-profit-limit, trigger when price >= stop_price
                stop_triggered = (report.side == OrderSide.BUY and current_price <= stop_price) or \
                               (report.side == OrderSide.SELL and current_price >= stop_price)
            
            if stop_triggered:
                # Convert to limit order and execute
                logger.info(f"Stop-limit order {report.order_id} triggered at price {current_price}")
                
                # Update order type to LIMIT and process
                report.order_type = OrderType.LIMIT
                self._process_limit_order(report)
            else:
                # Add to active orders for monitoring
                logger.info(f"Stop-limit order {report.order_id} not triggered (current: {current_price}, stop: {stop_price})")
                
        except Exception as e:
            logger.error(f"Error processing stop-limit order {report.order_id}: {str(e)}", exc_info=True)
            self._reject_order(report, f"Error: {str(e)}")
    
    def _process_twap_order(self, report: ExecutionReport) -> None:
        """
        Process a TWAP (Time-Weighted Average Price) order.
        
        Args:
            report: Execution report for the TWAP order
            
        The method creates a TWAP order with the specified parameters, validates the duration and interval,
        and updates the execution report with the order status. It also handles any errors that
        may occur during order creation.
        """
        try:
            order = report.order
            
            # Extract TWAP parameters from order metadata or use defaults
            twap_params = order.metadata.get('twap_params', {})
            duration_seconds = int(twap_params.get('duration_seconds', 300))  # 5 minutes default
            slice_interval = int(twap_params.get('slice_interval', 30))  # 30 seconds between slices
            
            # Validate parameters
            if duration_seconds <= 0:
                raise ValueError(f"Duration must be positive, got {duration_seconds} seconds")
            if slice_interval <= 0:
                raise ValueError(f"Slice interval must be positive, got {slice_interval} seconds")
            if slice_interval > duration_seconds:
                raise ValueError(f"Slice interval ({slice_interval}s) cannot be larger than duration ({duration_seconds}s)")
                
            logger.info(f"Creating TWAP order for {order.symbol} with {order.quantity} units "
                       f"over {duration_seconds} seconds (slice interval: {slice_interval}s)")
            
            # Create TWAP order
            order_id = self.twap_executor.create_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                duration_seconds=duration_seconds,
                slice_interval=slice_interval,
                order_type=OrderType.LIMIT if order.order_type == OrderType.TWAP else order.order_type,
                price=order.price,
                **twap_params
            )
            
            # Update report
            report.status = ExecutionResultStatus.PENDING
            report.metadata['twap_order_id'] = order_id
            report.message = f"TWAP order created: {order_id}"
            
            # Update metrics
            self.metrics['algo_orders']['twap'] += 1
            logger.info(f"Successfully created TWAP order {order_id} for parent order {order.order_id}")
            
        except ValueError as ve:
            error_msg = f"Invalid TWAP order parameters: {str(ve)}"
            logger.error(error_msg)
            report.status = ExecutionResultStatus.REJECTED
            report.message = error_msg
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
            
        except Exception as e:
            error_msg = f"Error creating TWAP order: {str(e)}"
            logger.error(error_msg, exc_info=True)
            report.status = ExecutionResultStatus.REJECTED
            report.message = error_msg
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
    
    def _process_vwap_order(self, report: ExecutionReport) -> None:
        """
        Process a VWAP (Volume-Weighted Average Price) order.
        
        Args:
            report: Execution report for the VWAP order
            
        The method creates a VWAP order with the specified parameters, validates the duration,
        and updates the execution report with the order status. It also handles any errors that
        may occur during order creation.
        """
        try:
            order = report.order
            
            # Extract VWAP parameters from order metadata or use defaults
            vwap_params = order.metadata.get('vwap_params', {})
            duration_seconds = int(vwap_params.get('duration_seconds', 300))  # 5 minutes default
            
            # Validate parameters
            if duration_seconds <= 0:
                raise ValueError(f"Duration must be positive, got {duration_seconds} seconds")
                
            # Get average daily volume for validation
            avg_daily_volume = self._get_average_daily_volume(order.symbol)
            if avg_daily_volume <= 0:
                logger.warning(f"Average daily volume for {order.symbol} is {avg_daily_volume}, which may affect VWAP execution")
                
            logger.info(f"Creating VWAP order for {order.symbol} with {order.quantity} units "
                       f"over {duration_seconds} seconds (avg daily volume: {avg_daily_volume:.2f})")
            
            # Create VWAP order
            order_id = self.vwap_executor.create_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                duration_seconds=duration_seconds,
                order_type=OrderType.LIMIT if order.order_type == OrderType.VWAP else order.order_type,
                price=order.price,
                **vwap_params
            )
            
            # Update report
            report.status = ExecutionResultStatus.PENDING
            report.metadata['vwap_order_id'] = order_id
            report.message = f"VWAP order created: {order_id}"
            
            # Update metrics
            self.metrics['algo_orders']['vwap'] += 1
            logger.info(f"Successfully created VWAP order {order_id} for parent order {order.order_id}")
            
        except ValueError as ve:
            error_msg = f"Invalid VWAP order parameters: {str(ve)}"
            logger.error(error_msg)
            report.status = ExecutionResultStatus.REJECTED
            report.message = error_msg
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
            
        except Exception as e:
            error_msg = f"Error creating VWAP order: {str(e)}"
            logger.error(error_msg, exc_info=True)
            report.status = ExecutionResultStatus.REJECTED
            report.message = error_msg
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
    
    def _process_iceberg_order(self, report: ExecutionReport) -> None:
        """
        Process an Iceberg order.
        
        Args:
            report: Execution report for the Iceberg order
            
        The method creates an Iceberg order with the specified parameters, validates the tip size,
        and updates the execution report with the order status. It also handles any errors that
        may occur during order creation.
        """
        try:
            order = report.order
            
            # Extract Iceberg parameters from order metadata or use defaults
            iceberg_params = order.metadata.get('iceberg_params', {})
            
            # Calculate tip size - default to 10% of order quantity or 10 units, whichever is smaller
            default_tip_size = min(order.quantity * 0.1, 10.0)
            tip_size = float(iceberg_params.get('tip_size', default_tip_size))
            
            # Validate tip size
            if tip_size <= 0:
                raise ValueError(f"Tip size must be positive, got {tip_size}")
            if tip_size > order.quantity:
                raise ValueError(f"Tip size ({tip_size}) cannot be larger than order quantity ({order.quantity})")
            
            logger.info(f"Creating Iceberg order for {order.symbol} with tip size {tip_size} "
                       f"(total quantity: {order.quantity})")
            
            # Create Iceberg order
            order_id = self.iceberg_executor.create_order(
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                tip_size=tip_size,
                order_type=OrderType.LIMIT if order.order_type == OrderType.ICEBERG else order.order_type,
                price=order.price,
                **iceberg_params
            )
            
            # Update report
            report.status = ExecutionResultStatus.PENDING
            report.metadata['iceberg_order_id'] = order_id
            report.message = f"Iceberg order created: {order_id}"
            
            # Update metrics
            self.metrics['algo_orders']['iceberg'] += 1
            logger.info(f"Successfully created Iceberg order {order_id} for parent order {order.order_id}")
            
        except ValueError as ve:
            error_msg = f"Invalid Iceberg order parameters: {str(ve)}"
            logger.error(error_msg)
            report.status = ExecutionResultStatus.REJECTED
            report.message = error_msg
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
            
        except Exception as e:
            error_msg = f"Error creating Iceberg order: {str(e)}"
            logger.error(error_msg, exc_info=True)
            report.status = ExecutionResultStatus.REJECTED
            report.message = error_msg
            self.metrics['rejected_orders'] += 1
            self.metrics['risk_checks']['rejections'] += 1
    
    def run(self) -> None:
        """
        Run the execution engine (to be called in a loop).
        
        This method processes all active orders, including algorithmic orders (TWAP, VWAP, Iceberg).
        """
        # Process TWAP orders
        for order_id, order in list(self.twap_executor.active_orders.items()):
            result = self.twap_executor.execute_next_slice(order_id)
            if result:
                # Update the parent order status if needed
                parent_order_id = result.get('parent_order_id')
                if parent_order_id in self.active_orders:
                    parent_report = self.active_orders[parent_order_id]
                    parent_report.filled_quantity += result.get('filled_quantity', 0)
                    parent_report.remaining_quantity = result.get('remaining_quantity', parent_report.order.quantity)
                    
                    # If order is complete, update status
                    if parent_report.remaining_quantity <= 0:
                        parent_report.status = ExecutionResultStatus.FILLED
                        parent_report.message = "TWAP order completed"
                        self.completed_orders[parent_order_id] = parent_report
                        del self.active_orders[parent_order_id]
        
        # Process VWAP orders
        for order_id, order in list(self.vwap_executor.active_orders.items()):
            result = self.vwap_executor.execute_next_slice(order_id)
            if result:
                # Update the parent order status if needed
                parent_order_id = result.get('parent_order_id')
                if parent_order_id in self.active_orders:
                    parent_report = self.active_orders[parent_order_id]
                    parent_report.filled_quantity += result.get('filled_quantity', 0)
                    parent_report.remaining_quantity = result.get('remaining_quantity', parent_report.order.quantity)
                    
                    # Update average price
                    if 'avg_price' in result:
                        total_value = parent_report.avg_price * (parent_report.filled_quantity - result.get('filled_quantity', 0)) + \
                                    result.get('avg_price', 0) * result.get('filled_quantity', 0)
                        parent_report.avg_price = total_value / parent_report.filled_quantity if parent_report.filled_quantity > 0 else 0
                    
                    # If order is complete, update status
                    if parent_report.remaining_quantity <= 0:
                        parent_report.status = ExecutionResultStatus.FILLED
                        parent_report.message = "VWAP order completed"
                        self.completed_orders[parent_order_id] = parent_report
                        del self.active_orders[parent_order_id]
        
        # Process Iceberg orders
        for order_id, order in list(self.iceberg_executor.active_orders.items()):
            result = self.iceberg_executor.execute_next_slice(order_id)
            if result:
                # Update the parent order status if needed
                parent_order_id = result.get('parent_order_id')
                if parent_order_id in self.active_orders:
                    parent_report = self.active_orders[parent_order_id]
                    parent_report.filled_quantity = result.get('filled_quantity', 0)
                    parent_report.remaining_quantity = result.get('remaining_quantity', parent_report.order.quantity)
                    
                    # Update average price
                    if 'avg_price' in result:
                        parent_report.avg_price = result['avg_price']
                    
                    # Update status based on Iceberg order status
                    if result.get('status') == 'FILLED':
                        parent_report.status = ExecutionResultStatus.FILLED
                        parent_report.message = "Iceberg order completed"
                        self.completed_orders[parent_order_id] = parent_report
                        del self.active_orders[parent_order_id]
                    elif result.get('status') == 'PARTIALLY_FILLED':
                        parent_report.status = ExecutionResultStatus.PARTIALLY_FILLED
        
        # Process other active orders (market, limit, etc.)
        for order_id, report in list(self.active_orders.items()):
            if report.status not in [ExecutionResultStatus.FILLED, ExecutionResultStatus.CANCELLED, 
                                   ExecutionResultStatus.REJECTED]:
                # Check for order timeouts
                if hasattr(report, 'created_at') and (datetime.now(pytz.utc) - report.created_at).total_seconds() > 86400:  # 24h timeout
                    self.cancel_order(order_id, "Order timed out")
                # Add other order processing logic here
    
    def cancel_order(self, order_id: str, reason: str = "User requested") -> bool:
        """
        Cancel an active order.
        
        Args:
            order_id: ID of the order to cancel
            reason: Reason for cancellation
            
        Returns:
            True if order was cancelled, False otherwise
        """
        if order_id in self.active_orders:
            report = self.active_orders[order_id]
            order = report.order
            
            # Cancel based on order type
            if order.order_type == OrderType.TWAP and 'twap_order_id' in report.metadata:
                self.twap_executor.cancel_order(report.metadata['twap_order_id'])
            elif order.order_type == OrderType.VWAP and 'vwap_order_id' in report.metadata:
                self.vwap_executor.cancel_order(report.metadata['vwap_order_id'])
            elif order.order_type == OrderType.ICEBERG and 'iceberg_order_id' in report.metadata:
                self.iceberg_executor.cancel_order(report.metadata['iceberg_order_id'])
            
            # Update order status
            report.status = ExecutionResultStatus.CANCELLED
            report.message = f"Order cancelled: {reason}"
            report.updated_at = datetime.now(pytz.utc)
            
            # Move to completed orders
            self.completed_orders[order_id] = report
            del self.active_orders[order_id]
            
            # Update metrics
            self.metrics['cancelled_orders'] += 1
            
            return True
            
        return False
    
    def _check_risk_limits(self, order: Order) -> Dict[str, Any]:
        """
        Check if an order violates any risk limits with enhanced validations.
        
        Args:
            order: Order to check
            
        Returns:
            Dictionary with 'allowed' flag and 'reason' if not allowed
        """
        # Track passed checks for metrics
        risk_checks_passed = 0
        
        # 1. Check order size limits
        max_order_size = self.order_limits.get('max_order_size', float('inf'))
        min_order_size = self.order_limits.get('min_order_size', 0.0)
        
        if order.quantity > max_order_size:
            return {
                'allowed': False,
                'reason': f"Order size {order.quantity} exceeds maximum of {max_order_size}",
                'check': 'max_order_size'
            }
            
        if order.quantity < min_order_size:
            return {
                'allowed': False,
                'reason': f"Order size {order.quantity} is below minimum of {min_order_size}",
                'check': 'min_order_size'
            }
        
        # 2. Check order value limits
        if order.price is not None:
            order_value = order.quantity * order.price
            max_order_value = self.order_limits.get('max_order_value', float('inf'))
            
            if order_value > max_order_value:
                return {
                    'allowed': False,
                    'reason': f"Order value {order_value:.2f} exceeds maximum of {max_order_value:.2f}",
                    'check': 'max_order_value'
                }
        
        # 3. Check position limits
        current_position = self._get_position(order.symbol)
        position_delta = order.quantity if order.side == OrderSide.BUY else -order.quantity
        new_position = current_position + position_delta
        
        max_position_size = self.position_limits.get('max_position_size', float('inf'))
        if abs(new_position) > max_position_size:
            return {
                'allowed': False,
                'reason': f"New position {new_position:.4f} would exceed maximum of {max_position_size}",
                'check': 'max_position_size'
            }
        
        # 4. Check portfolio concentration
        if order.price is not None and 'portfolio_value' in self.metrics:
            position_value = abs(new_position) * order.price
            portfolio_value = self.metrics['portfolio_value']
            max_concentration = self.circuit_breakers.get('max_position_concentration', 1.0)
            
            if portfolio_value > 0 and (position_value / portfolio_value) > max_concentration:
                return {
                    'allowed': False,
                    'reason': f"Position would exceed maximum concentration of {max_concentration*100:.1f}% of portfolio",
                    'check': 'position_concentration'
                }
        
        # 5. Check daily loss limits
        if 'daily_pnl' in self.metrics and 'initial_portfolio_value' in self.metrics:
            daily_pnl = self.metrics['daily_pnl'].get('total', 0)
            initial_value = self.metrics['initial_portfolio_value']
            max_daily_loss = self.circuit_breakers.get('max_daily_loss_pct', 0.05)  # 5% default
            
            if initial_value > 0 and (-daily_pnl / initial_value) > max_daily_loss:
                return {
                    'allowed': False,
                    'reason': f"Order rejected: Daily loss limit of {max_daily_loss*100:.1f}% reached",
                    'check': 'daily_loss_limit'
                }
        
        # 6. Check rate limits
        current_minute = int(time.time() / 60)
        if not hasattr(self, '_last_order_minute'):
            self._last_order_minute = current_minute
            self._orders_this_minute = 0
        
        if current_minute != self._last_order_minute:
            self._last_order_minute = current_minute
            self._orders_this_minute = 0
        
        max_orders_per_minute = self.order_limits.get('max_orders_per_minute', float('inf'))
        self._orders_this_minute += 1
        
        if self._orders_this_minute > max_orders_per_minute:
            return {
                'allowed': False,
                'reason': f"Rate limit exceeded: {self._orders_this_minute} orders this minute (max: {max_orders_per_minute})",
                'check': 'rate_limit'
            }
        
        # 7. Check price bands (for limit orders)
        if order.order_type == OrderType.LIMIT and order.price is not None:
            order_book = self.order_books.get(order.symbol)
            if order_book:
                best_bid, best_ask = order_book.get_best_bid_ask()
                if best_bid is not None and best_ask is not None:
                    mid_price = (best_bid + best_ask) / 2
                    price_band_pct = self.order_limits.get('price_band_pct', 0.05)  # 5% default
                    
                    if order.side == OrderSide.BUY and order.price > mid_price * (1 + price_band_pct):
                        return {
                            'allowed': False,
                            'reason': f"Buy limit price {order.price:.4f} is more than {price_band_pct*100:.1f}% above mid price {mid_price:.4f}",
                            'check': 'price_band'
                        }
                    elif order.side == OrderSide.SELL and order.price < mid_price * (1 - price_band_pct):
                        return {
                            'allowed': False,
                            'reason': f"Sell limit price {order.price:.4f} is more than {price_band_pct*100:.1f}% below mid price {mid_price:.4f}",
                            'check': 'price_band'
                        }
        
        # 8. Check leverage limits (if applicable)
        if hasattr(self, 'calculate_leverage') and hasattr(self, 'max_leverage'):
            current_leverage = self.calculate_leverage()
            max_leverage = self.position_limits.get('max_leverage', 10.0)  # 10x default
            
            # Estimate new leverage (simplified)
            if order.price is not None and 'portfolio_value' in self.metrics:
                position_value = abs(new_position) * order.price
                portfolio_value = self.metrics['portfolio_value']
                
                if portfolio_value > 0:
                    new_leverage = position_value / portfolio_value
                    if new_leverage > max_leverage:
                        return {
                            'allowed': False,
                            'reason': f"Order would exceed maximum leverage of {max_leverage:.1f}x (new: {new_leverage:.1f}x)",
                            'check': 'leverage_limit'
                        }
        
        # All checks passed
        self.metrics['risk_checks']['passed'] += 1
        return {'allowed': True, 'check': 'all_passed'}
    
    def _get_position(self, symbol: str) -> float:
        """
        Get the current position for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Current position quantity (positive for long, negative for short)
        """
        return self.positions.get(symbol, 0.0)
    
    def _update_position_and_exposure(self, report: ExecutionReport) -> None:
        """
        Update position and exposure after an order is filled.
        
        Args:
            report: Execution report of the filled order
        """
        if report.avg_price is None:
            return
            
        symbol = report.symbol
        quantity = report.filled_quantity
        price = report.avg_price
        
        # Update position
        position_delta = quantity if report.side == OrderSide.BUY else -quantity
        self.positions[symbol] = self.positions.get(symbol, 0.0) + position_delta
        
        # Update exposure
        position_value = abs(self.positions[symbol]) * price
        self.exposure[symbol] = position_value
        
        # Update daily P&L (simplified)
        if 'daily_pnl' not in self.metrics:
            self.metrics['daily_pnl'] = {}
        
        # This is a simplified P&L calculation
        # In a real system, you'd track cost basis and calculate P&L more accurately
        pnl_delta = -quantity * price  # Negative for buys, positive for sells
        self.metrics['daily_pnl'][symbol] = self.metrics['daily_pnl'].get(symbol, 0.0) + pnl_delta
        self.metrics['daily_pnl']['total'] = self.metrics['daily_pnl'].get('total', 0.0) + pnl_delta  # Replace with actual implementation
    
    def get_order_status(self, order_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of an order.
        
        Args:
            order_id: ID of the order
            
        Returns:
            Order status as a dictionary, or None if not found
        """
        # Check active orders first
        if order_id in self.active_orders:
            return self.active_orders[order_id].to_dict()
            
        # Check completed orders
        if order_id in self.completed_orders:
            return self.completed_orders[order_id].to_dict()
            
        return None
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all open orders.
        
        Args:
            symbol: Optional symbol to filter by
            
        Returns:
            List of open orders
        """
        orders = []
        
        for order_id, report in self.active_orders.items():
            if symbol is None or report.symbol == symbol:
                orders.append(report.to_dict())
                
        return orders
    
    def get_order_history(self, 
                        symbol: Optional[str] = None,
                        start_time: Optional[datetime] = None,
                        end_time: Optional[datetime] = None,
                        limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get order history with optional filters.
        
        Args:
            symbol: Optional symbol to filter by
            start_time: Optional start time filter
            end_time: Optional end time filter
            limit: Maximum number of orders to return
            
        Returns:
            List of historical orders
        """
        orders = []
        
        # Convert completed_orders to a list and sort by update time (newest first)
        sorted_orders = sorted(
            self.completed_orders.values(),
            key=lambda x: x.updated_at,
            reverse=True
        )
        
        for report in sorted_orders:
            # Apply filters
            if symbol is not None and report.symbol != symbol:
                continue
                
            if start_time is not None and report.updated_at < start_time:
                continue
                
            if end_time is not None and report.updated_at > end_time:
                continue
                
            orders.append(report.to_dict())
            
            # Apply limit
            if len(orders) >= limit:
                break
                
        return orders
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get execution metrics.
        
        Returns:
            Dictionary of execution metrics
        """
        # Calculate additional metrics
        elapsed_hours = (datetime.now(pytz.utc) - self.metrics['start_time']).total_seconds() / 3600
        
        metrics = self.metrics.copy()
        metrics.update({
            'elapsed_hours': elapsed_hours,
            'orders_per_hour': metrics['total_orders'] / max(1, elapsed_hours),
            'avg_order_value': metrics['total_volume'] / max(1, metrics['filled_orders']),
            'last_updated': datetime.now(pytz.utc).isoformat()
        })
        
        return metrics
    
    def shutdown(self) -> None:
        """Shut down the execution engine."""
        self._shutdown = True
        
        # Cancel all active orders
        for order_id in list(self.active_orders.keys()):
            self.cancel_order(order_id, "System shutdown")
        
        logger.info("Execution engine shutdown complete")


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    from ..interfaces.order_book import OrderBook
    
    # Create a sample order book
    order_book = OrderBook(symbol="BTC/USDT")
    
    # Add some liquidity
    order_book.add_order(Order(
        order_id="ob_1",
        symbol="BTC/USDT",
        side=OrderSide.SELL,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=50000.0,
        timestamp=datetime.now(pytz.utc)
    ))
    
    order_book.add_order(Order(
        order_id="ob_2",
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        order_type=OrderType.LIMIT,
        quantity=1.0,
        price=49000.0,
        timestamp=datetime.now(pytz.utc)
    ))
    
    # Create execution engine
    engine = ExecutionEngine(order_books={"BTC/USDT": order_book}, config={
        'slippage_model': 'fixed',
        'slippage_bps': 5.0,  # 5 bps slippage
        'fee_type': 'percentage',
        'maker_fee_bps': 2.0,  # 2 bps
        'taker_fee_bps': 5.0,  # 5 bps
        'impact_model': 'linear',
        'impact_coeff': 0.1  # 10% impact for 100% of ADV
    })
    
    # Submit a market buy order
    order_id = engine.submit_order(
        symbol="BTC/USDT",
        side=OrderSide.BUY,
        quantity=0.5,
        order_type=OrderType.MARKET,
        strategy_id="test_strategy"
    )
    
    # Get order status
    status = engine.get_order_status(order_id)
    print(f"Order status: {status}")
    
    # Get metrics
    metrics = engine.get_metrics()
    print(f"Execution metrics: {metrics}")
    
    # Shutdown
    engine.shutdown()
