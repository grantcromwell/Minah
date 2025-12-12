"""
Optimized Execution Engine for high-frequency trading.

This module implements performance-optimized versions of the core execution components.
"""
from __future__ import annotations

import time
import logging
from typing import Dict, List, Optional, Tuple, Any, Deque, Set
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque, defaultdict
import numpy as np
from datetime import datetime, timedelta
import pytz
from concurrent.futures import ThreadPoolExecutor
from threading import RLock

# Local imports
from ..interfaces.order_book import OrderBook, Order, OrderType, OrderSide, OrderStatus
from ..data import MarketDataConnector, DataPipeline

# Type aliases
OrderID = str
Symbol = str

# Configure logging
logger = logging.getLogger(__name__)

class ExecutionStyle(Enum):
    """Optimized execution styles enum."""
    MARKET = auto()
    LIMIT = auto()
    TWAP = auto()
    VWAP = auto()
    POV = auto()
    ICEBERG = auto()
    HIDDEN = auto()
    PEGGED = auto()

class ExecutionResultStatus(Enum):
    """Optimized execution result status enum."""
    FILLED = auto()
    PARTIALLY_FILLED = auto()
    REJECTED = auto()
    CANCELLED = auto()
    EXPIRED = auto()
    PENDING = auto()

# Object pool for ExecutionResult
class ExecutionResultPool:
    """Object pool for ExecutionResult instances."""
    def __init__(self, initial_size: int = 1000):
        self._pool = deque(maxlen=initial_size * 2)
        self._lock = RLock()
        self._active = set()
        
        # Pre-fill the pool
        for _ in range(initial_size):
            self._pool.append(self._create_result())
    
    def _create_result(self) -> ExecutionResult:
        """Create a new ExecutionResult instance."""
        return ExecutionResult(
            order_id="",
            status=ExecutionResultStatus.PENDING,
            filled_quantity=0.0,
            remaining_quantity=0.0,
            avg_price=0.0,
            fees=0.0,
            timestamp=datetime.now(pytz.utc),
            metadata={}
        )
    
    def acquire(self, order_id: str) -> ExecutionResult:
        """Acquire an ExecutionResult from the pool."""
        with self._lock:
            if self._pool:
                result = self._pool.popleft()
            else:
                result = self._create_result()
            
            # Reset and update the result
            result.order_id = order_id
            result.status = ExecutionResultStatus.PENDING
            result.filled_quantity = 0.0
            result.remaining_quantity = 0.0
            result.avg_price = 0.0
            result.fees = 0.0
            result.timestamp = datetime.now(pytz.utc)
            result.metadata.clear()
            
            self._active.add(id(result))
            return result
    
    def release(self, result: ExecutionResult) -> None:
        """Release an ExecutionResult back to the pool."""
        with self._lock:
            if id(result) in self._active:
                self._active.remove(id(result))
                self._pool.append(result)

# Global object pool
result_pool = ExecutionResultPool(initial_size=1000)

@dataclass(slots=True)
class ExecutionResult:
    """Optimized execution result with slots for memory efficiency."""
    __slots__ = [
        'order_id', 'status', 'filled_quantity', 'remaining_quantity',
        'avg_price', 'fees', 'timestamp', 'metadata'
    ]
    
    order_id: str
    status: ExecutionResultStatus
    filled_quantity: float
    remaining_quantity: float
    avg_price: float
    fees: float
    timestamp: datetime
    metadata: Dict[str, Any]
    
    def release(self) -> None:
        """Release this result back to the pool."""
        result_pool.release(self)

class OptimizedExecutionEngine:
    """
    Optimized execution engine with performance improvements for high-frequency trading.
    """
    
    def __init__(self, 
                 order_books: Dict[Symbol, OrderBook],
                 data_connector: Optional[MarketDataConnector] = None,
                 data_pipeline: Optional[DataPipeline] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the optimized execution engine.
        
        Args:
            order_books: Dictionary of order books by symbol
            data_connector: Market data connector
            data_pipeline: Data pipeline for historical data
            config: Configuration dictionary
        """
        self.order_books = order_books
        self.data_connector = data_connector or MarketDataConnector()
        self.data_pipeline = data_pipeline or DataPipeline()
        self.config = config or {}
        
        # Thread-safe collections
        self.active_orders: Dict[OrderID, Dict] = {}
        self.completed_orders: Dict[OrderID, Dict] = {}
        self.positions: Dict[Symbol, float] = defaultdict(float)
        self._order_counter = 1
        self._lock = RLock()
        
        # Thread pool for concurrent order processing
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.config.get('max_workers', 4),
            thread_name_prefix='exec_engine_worker'
        )
        
        # Initialize metrics
        self.metrics = {
            'total_orders': 0,
            'filled_orders': 0,
            'cancelled_orders': 0,
            'rejected_orders': 0,
            'total_volume': 0.0,
            'total_fees': 0.0,
            'slippage': 0.0,
            'latency': deque(maxlen=1000),  # Circular buffer for recent latencies
            'algo_orders': defaultdict(int),
            'risk_checks': {
                'passed': 0,
                'failed': 0,
                'rejections': 0
            }
        }
    
    def submit_order(self, order: Order) -> ExecutionResult:
        """
        Submit a new order for execution with optimized performance.
        
        Args:
            order: Order to execute
            
        Returns:
            Execution result
        """
        start_time = time.perf_counter()
        result = result_pool.acquire(order.order_id)
        
        try:
            # Basic validation
            if order.quantity <= 0:
                result.status = ExecutionResultStatus.REJECTED
                result.metadata['reason'] = 'Invalid quantity'
                self.metrics['rejected_orders'] += 1
                return result
                
            # Check if symbol exists
            if order.symbol not in self.order_books:
                result.status = ExecutionResultStatus.REJECTED
                result.metadata['reason'] = f'Symbol {order.symbol} not found'
                self.metrics['rejected_orders'] += 1
                self.metrics['risk_checks']['rejections'] += 1
                return result
            
            # Process the order based on type
            if order.order_type == OrderType.MARKET:
                self._process_market_order(order, result)
            elif order.order_type == OrderType.LIMIT:
                self._process_limit_order(order, result)
            else:
                result.status = ExecutionResultStatus.REJECTED
                result.metadata['reason'] = f'Unsupported order type: {order.order_type}'
                self.metrics['rejected_orders'] += 1
                
        except Exception as e:
            logger.exception(f"Error processing order {order.order_id}")
            result.status = ExecutionResultStatus.REJECTED
            result.metadata['error'] = str(e)
            self.metrics['rejected_orders'] += 1
            
        finally:
            # Update metrics
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.metrics['latency'].append(latency_ms)
            self.metrics['total_orders'] += 1
            
            if result.status == ExecutionResultStatus.FILLED:
                self.metrics['filled_orders'] += 1
                self.metrics['total_volume'] += result.filled_quantity
            
            return result
    
    def _process_market_order(self, order: Order, result: ExecutionResult) -> None:
        """Process a market order."""
        order_book = self.order_books[order.symbol]
        
        # Get best available price
        if order.side == OrderSide.BUY:
            best_price = order_book.get_ask()
            if best_price is None:
                result.status = ExecutionResultStatus.REJECTED
                result.metadata['reason'] = 'No ask price available'
                return
        else:  # SELL
            best_price = order_book.get_bid()
            if best_price is None:
                result.status = ExecutionResultStatus.REJECTED
                result.metadata['reason'] = 'No bid price available'
                return
        
        # Execute at best available price
        result.avg_price = best_price
        result.filled_quantity = order.quantity
        result.remaining_quantity = 0.0
        result.status = ExecutionResultStatus.FILLED
        
        # Update position
        with self._lock:
            self.positions[order.symbol] += (
                order.quantity if order.side == OrderSide.BUY else -order.quantity
            )
    
    def _process_limit_order(self, order: Order, result: ExecutionResult) -> None:
        """Process a limit order."""
        if order.price is None:
            result.status = ExecutionResultStatus.REJECTED
            result.metadata['reason'] = 'Limit price required for limit orders'
            return
            
        order_book = self.order_books[order.symbol]
        
        # Check if the order can be filled immediately
        if order.side == OrderSide.BUY and order.price >= order_book.get_ask():
            # Marketable limit order (buy at or above ask)
            fill_price = order_book.get_ask()
            if fill_price is None:
                result.status = ExecutionResultStatus.REJECTED
                result.metadata['reason'] = 'No ask price available'
                return
                
            result.avg_price = min(order.price, fill_price)
            result.filled_quantity = order.quantity
            result.remaining_quantity = 0.0
            result.status = ExecutionResultStatus.FILLED
            
        elif order.side == OrderSide.SELL and order.price <= order_book.get_bid():
            # Marketable limit order (sell at or below bid)
            fill_price = order_book.get_bid()
            if fill_price is None:
                result.status = ExecutionResultStatus.REJECTED
                result.metadata['reason'] = 'No bid price available'
                return
                
            result.avg_price = max(order.price, fill_price)
            result.filled_quantity = order.quantity
            result.remaining_quantity = 0.0
            result.status = ExecutionResultStatus.FILLED
            
        else:
            # Resting limit order - add to order book
            result.status = ExecutionResultStatus.PENDING
            result.remaining_quantity = order.quantity
            
            with self._lock:
                self.active_orders[order.order_id] = {
                    'order': order,
                    'created_at': datetime.now(pytz.utc),
                    'status': 'PENDING'
                }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current execution metrics."""
        with self._lock:
            return {
                **self.metrics,
                'active_orders': len(self.active_orders),
                'completed_orders': len(self.completed_orders),
                'positions': dict(self.positions),
                'avg_latency_ms': (
                    sum(self.metrics['latency']) / len(self.metrics['latency']) 
                    if self.metrics['latency'] else 0
                )
            }
    
    def shutdown(self) -> None:
        """Shut down the execution engine and release resources."""
        self.thread_pool.shutdown(wait=True)
        logger.info("Execution engine shutdown complete")
