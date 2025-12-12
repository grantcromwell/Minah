"""
TWAP (Time-Weighted Average Price) execution algorithm.
"""
from __future__ import annotations

import time
import random
import logging
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pytz

from ....interfaces.order_book import Order, OrderType, OrderSide
from ....data import MarketDataConnector

logger = logging.getLogger(__name__)

@dataclass
class TWAPOrder:
    """TWAP order configuration."""
    symbol: str
    side: OrderSide
    total_quantity: float
    duration_seconds: float
    start_time: datetime
    end_time: datetime
    order_type: OrderType = OrderType.LIMIT
    price_band_pct: float = 0.01  # 1% price band around market price
    min_slice_size: float = 0.01  # Minimum order size
    max_slice_size: float = 100.0  # Maximum order size per slice
    
    def __post_init__(self):
        """Validate and initialize order parameters."""
        if self.duration_seconds <= 0:
            raise ValueError("Duration must be positive")
        if self.total_quantity <= 0:
            raise ValueError("Total quantity must be positive")
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
        
        self.remaining_quantity = self.total_quantity
        self.slices: List[Tuple[float, float]] = []  # (time_offset, quantity) pairs
        self._generate_slices()
    
    def _generate_slices(self) -> None:
        """Generate order slices based on TWAP parameters."""
        # Calculate number of slices based on duration and minimum slice size
        min_slice_duration = max(1.0, 60.0)  # At least 1 second per slice, 1 minute default
        max_slices = int(self.duration_seconds / min_slice_duration)
        
        # Calculate slice quantity
        slice_quantity = max(
            self.min_slice_size,
            min(
                self.max_slice_size,
                self.total_quantity / max(1, max_slices)
            )
        )
        
        # Generate slices
        remaining = self.total_quantity
        time_step = self.duration_seconds / max(1, (self.total_quantity / slice_quantity))
        
        current_time = 0.0
        while remaining > 0 and current_time <= self.duration_seconds:
            # Calculate quantity for this slice
            qty = min(slice_quantity, remaining)
            if qty < self.min_slice_size and qty < remaining:
                # Don't create very small slices unless it's the last one
                current_time += time_step
                continue
                
            self.slices.append((current_time, qty))
            remaining -= qty
            current_time += time_step
        
        # Adjust last slice to account for any remaining quantity due to rounding
        if remaining > 0:
            if self.slices:
                # Add to last slice
                last_time, last_qty = self.slices[-1]
                self.slices[-1] = (last_time, last_qty + remaining)
            else:
                # Single slice case
                self.slices.append((0, remaining))
    
    def get_next_slice(self, current_time: datetime) -> Optional[Tuple[float, float]]:
        """
        Get the next order slice to execute.
        
        Args:
            current_time: Current time
            
        Returns:
            Tuple of (time_offset, quantity) or None if no more slices
        """
        if not self.slices:
            return None
            
        elapsed = (current_time - self.start_time).total_seconds()
        
        # Find the next slice that should be executed
        for i, (time_offset, qty) in enumerate(self.slices):
            if time_offset <= elapsed and qty > 0:
                # Mark this slice as executed
                self.slices[i] = (time_offset, 0)  # Mark as executed
                self.remaining_quantity -= qty
                return time_offset, qty
                
        return None

class TWAPExecutor:
    """
    TWAP (Time-Weighted Average Price) execution algorithm.
    
    This algorithm slices a large order into smaller chunks and executes them
    at regular intervals to minimize market impact.
    """
    
    def __init__(self, 
                 order_books: Dict[str, Any],
                 data_connector: Optional[MarketDataConnector] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the TWAP executor.
        
        Args:
            order_books: Dictionary of order books by symbol
            data_connector: Market data connector
            config: Configuration dictionary
        """
        self.order_books = order_books
        self.data_connector = data_connector or MarketDataConnector()
        self.config = config or {}
        self.active_orders: Dict[str, TWAPOrder] = {}
        self.completed_orders: Dict[str, Dict] = {}
        self.next_order_id = 1
        
        # Default configuration
        self.default_slice_duration = self.config.get('default_slice_duration', 60.0)  # seconds
        self.max_slice_size = self.config.get('max_slice_size', 100.0)
        self.min_slice_size = self.config.get('min_slice_size', 0.01)
        self.price_band_pct = self.config.get('price_band_pct', 0.01)  # 1%
    
    def create_order(self,
                    symbol: str,
                    side: OrderSide,
                    quantity: float,
                    duration_seconds: Optional[float] = None,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    order_type: OrderType = OrderType.LIMIT,
                    **kwargs) -> str:
        """
        Create a new TWAP order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Total quantity to execute
            duration_seconds: Duration of the TWAP in seconds
            start_time: Optional start time (default: now)
            end_time: Optional end time (if provided, overrides duration_seconds)
            order_type: Type of order (LIMIT/MARKET)
            **kwargs: Additional order parameters
            
        Returns:
            Order ID
        """
        # Set start and end times
        now = datetime.now(pytz.utc)
        start_time = start_time or now
        
        if end_time is not None:
            if end_time <= start_time:
                raise ValueError("End time must be after start time")
            duration_seconds = (end_time - start_time).total_seconds()
        else:
            duration_seconds = duration_seconds or self.default_slice_duration
            end_time = start_time + timedelta(seconds=duration_seconds)
        
        # Create order
        order_id = f"twap_{self.next_order_id}"
        self.next_order_id += 1
        
        order = TWAPOrder(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            duration_seconds=duration_seconds,
            start_time=start_time,
            end_time=end_time,
            order_type=order_type,
            price_band_pct=self.price_band_pct,
            min_slice_size=self.min_slice_size,
            max_slice_size=self.max_slice_size,
            **kwargs
        )
        
        self.active_orders[order_id] = order
        self.completed_orders[order_id] = {
            'status': 'PENDING',
            'created_at': now,
            'updated_at': now,
            'filled_quantity': 0.0,
            'remaining_quantity': quantity,
            'slices': []
        }
        
        return order_id
    
    def execute_next_slice(self, order_id: str) -> Optional[Dict]:
        """
        Execute the next slice of a TWAP order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order execution details or None if no slice to execute
        """
        if order_id not in self.active_orders:
            return None
            
        order = self.active_orders[order_id]
        now = datetime.now(pytz.utc)
        
        # Check if order is still active
        if now > order.end_time:
            self._complete_order(order_id, 'EXPIRED')
            return None
            
        if now < order.start_time:
            return None
            
        # Get next slice to execute
        slice_info = order.get_next_slice(now)
        if not slice_info:
            if order.remaining_quantity <= 0:
                self._complete_order(order_id, 'FILLED')
            return None
            
        time_offset, quantity = slice_info
        
        # Get current market price
        symbol = order.symbol
        if symbol not in self.order_books:
            logger.warning(f"No order book for symbol: {symbol}")
            return None
            
        order_book = self.order_books[symbol]
        best_bid, best_ask = order_book.get_best_bid_ask()
        
        if best_bid is None or best_ask is None:
            logger.warning(f"No prices available for {symbol}")
            return None
            
        # Calculate limit price based on order type
        if order.order_type == OrderType.MARKET:
            price = best_ask if order.side == OrderSide.BUY else best_bid
        else:  # LIMIT order
            mid_price = (best_bid + best_ask) / 2
            if order.side == OrderSide.BUY:
                price = min(best_ask, mid_price * (1 + order.price_band_pct))
            else:  # SELL
                price = max(best_bid, mid_price * (1 - order.price_band_pct))
        
        # Create and execute the order
        slice_order = Order(
            order_id=f"{order_id}_slice_{len(self.completed_orders[order_id]['slices'])}",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=quantity,
            price=price if order.order_type == OrderType.LIMIT else None,
            timestamp=now,
            metadata={
                'parent_order_id': order_id,
                'twap_slice': True,
                'slice_number': len(self.completed_orders[order_id]['slices']) + 1,
                'total_slices': len(order.slices)
            }
        )
        
        # In a real implementation, this would submit the order to the exchange
        # For now, we'll just log it
        logger.info(f"Executing TWAP slice: {slice_order}")
        
        # Update order status
        self.completed_orders[order_id]['slices'].append({
            'order_id': slice_order.order_id,
            'quantity': quantity,
            'price': price,
            'timestamp': now.isoformat(),
            'status': 'EXECUTED'
        })
        
        self.completed_orders[order_id]['updated_at'] = now
        self.completed_orders[order_id]['filled_quantity'] += quantity
        self.completed_orders[order_id]['remaining_quantity'] = order.remaining_quantity
        
        return {
            'order_id': slice_order.order_id,
            'parent_order_id': order_id,
            'symbol': order.symbol,
            'side': order.side.name,
            'quantity': quantity,
            'price': price,
            'timestamp': now.isoformat(),
            'remaining_quantity': order.remaining_quantity,
            'slice_number': len(self.completed_orders[order_id]['slices']),
            'total_slices': len(order.slices) + len(self.completed_orders[order_id]['slices'])
        }
    
    def _complete_order(self, order_id: str, status: str) -> None:
        """Mark an order as completed."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            self.completed_orders[order_id].update({
                'status': status,
                'updated_at': datetime.now(pytz.utc),
                'end_time': datetime.now(pytz.utc)
            })
            del self.active_orders[order_id]
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get the status of a TWAP order."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            return {
                'order_id': order_id,
                'symbol': order.symbol,
                'side': order.side.name,
                'total_quantity': order.total_quantity,
                'filled_quantity': self.completed_orders[order_id]['filled_quantity'],
                'remaining_quantity': order.remaining_quantity,
                'status': 'ACTIVE',
                'start_time': order.start_time.isoformat(),
                'end_time': order.end_time.isoformat(),
                'slices_completed': len(self.completed_orders[order_id]['slices']),
                'total_slices': len(order.slices) + len(self.completed_orders[order_id]['slices'])
            }
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
        return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a TWAP order."""
        if order_id in self.active_orders:
            self._complete_order(order_id, 'CANCELLED')
            return True
        return False
    
    def run(self) -> None:
        """Run the TWAP executor (to be called in a loop)."""
        now = datetime.now(pytz.utc)
        
        # Process all active orders
        for order_id in list(self.active_orders.keys()):
            self.execute_next_slice(order_id)
            
            # Clean up completed orders
            if order_id in self.active_orders and self.active_orders[order_id].remaining_quantity <= 0:
                self._complete_order(order_id, 'FILLED')
