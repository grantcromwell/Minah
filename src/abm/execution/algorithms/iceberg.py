"""
Iceberg Order execution algorithm.

This module implements the Iceberg order type which only shows a small portion
(the "tip") of the total order quantity at any given time.
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

from ....interfaces.order_book import Order, OrderType, OrderSide, OrderStatus
from ....data import MarketDataConnector

logger = logging.getLogger(__name__)

@dataclass
class IcebergOrder:
    """Iceberg order configuration."""
    symbol: str
    side: OrderSide
    total_quantity: float
    tip_size: float  # Maximum visible quantity at any time
    order_type: OrderType = OrderType.LIMIT
    price: Optional[float] = None  # Fixed price for limit orders
    time_in_force: str = 'GTC'  # Good-Til-Cancelled
    max_slippage_pct: float = 0.01  # 1% maximum slippage
    min_quantity: float = 0.01  # Minimum order size
    max_attempts: int = 3  # Maximum number of placement attempts per slice
    
    def __post_init__(self):
        """Validate and initialize order parameters."""
        if self.total_quantity <= 0:
            raise ValueError("Total quantity must be positive")
        if self.tip_size <= 0 or self.tip_size > self.total_quantity:
            raise ValueError("Tip size must be positive and not exceed total quantity")
        if self.order_type == OrderType.LIMIT and self.price is None:
            raise ValueError("Limit orders require a price")
        
        self.remaining_quantity = self.total_quantity
        self.display_quantity = 0.0  # Currently displayed quantity
        self.executed_quantity = 0.0
        self.attempts = 0
        self.status: OrderStatus = OrderStatus.NEW
        self.created_at = datetime.now(pytz.utc)
        self.updated_at = self.created_at
        self.child_orders: List[Dict] = []
    
    def get_next_slice(self) -> Optional[float]:
        """
        Get the next slice quantity to display.
        
        Returns:
            Quantity to display next, or None if order is complete
        """
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
            return None
            
        # If we still have displayed quantity, return None (wait for it to be filled)
        if self.display_quantity > 0:
            return None
            
        # Calculate next slice size
        slice_size = min(self.tip_size, self.remaining_quantity)
        self.display_quantity = slice_size
        self.attempts = 0
        self.updated_at = datetime.now(pytz.utc)
        
        return slice_size
    
    def on_fill(self, filled_quantity: float, avg_price: float) -> None:
        """
        Update order state when a fill occurs.
        
        Args:
            filled_quantity: Quantity that was filled
            avg_price: Average fill price
        """
        if filled_quantity <= 0 or filled_quantity > self.display_quantity:
            raise ValueError("Invalid filled quantity")
            
        self.display_quantity -= filled_quantity
        self.remaining_quantity -= filled_quantity
        self.executed_quantity += filled_quantity
        self.updated_at = datetime.now(pytz.utc)
        
        # Update status
        if self.remaining_quantity <= 0:
            self.status = OrderStatus.FILLED
        elif self.status == OrderStatus.NEW:
            self.status = OrderStatus.PARTIALLY_FILLED
    
    def on_reject(self) -> None:
        """Handle order rejection."""
        self.attempts += 1
        self.updated_at = datetime.now(pytz.utc)
        
        if self.attempts >= self.max_attempts:
            # Too many rejections, cancel the order
            self.status = OrderStatus.REJECTED
            self.display_quantity = 0.0
    
    def cancel(self) -> None:
        """Cancel the order."""
        if self.status not in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self.status = OrderStatus.CANCELLED
            self.display_quantity = 0.0
            self.updated_at = datetime.now(pytz.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'side': self.side.name,
            'total_quantity': self.total_quantity,
            'remaining_quantity': self.remaining_quantity,
            'executed_quantity': self.executed_quantity,
            'display_quantity': self.display_quantity,
            'tip_size': self.tip_size,
            'order_type': self.order_type.name,
            'price': self.price,
            'status': self.status.name,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'child_orders': [str(o) for o in self.child_orders]
        }

class IcebergExecutor:
    """
    Iceberg order executor.
    
    This algorithm manages iceberg orders by only showing a small portion
    (the "tip") of the total order quantity at any given time.
    """
    
    def __init__(self, 
                 order_books: Dict[str, Any],
                 data_connector: Optional[MarketDataConnector] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Iceberg executor.
        
        Args:
            order_books: Dictionary of order books by symbol
            data_connector: Market data connector
            config: Configuration dictionary
        """
        self.order_books = order_books
        self.data_connector = data_connector or MarketDataConnector()
        self.config = config or {}
        self.active_orders: Dict[str, IcebergOrder] = {}
        self.completed_orders: Dict[str, Dict] = {}
        self.next_order_id = 1
        
        # Default configuration
        self.default_tip_pct = self.config.get('default_tip_pct', 0.1)  # 10% tip by default
        self.min_tip_size = self.config.get('min_tip_size', 0.01)
        self.max_slippage_pct = self.config.get('max_slippage_pct', 0.01)  # 1%
        self.reprice_interval = self.config.get('reprice_interval', 5.0)  # seconds
        self.max_attempts = self.config.get('max_attempts', 3)
    
    def create_order(self,
                    symbol: str,
                    side: OrderSide,
                    quantity: float,
                    tip_size: Optional[float] = None,
                    tip_percentage: Optional[float] = None,
                    order_type: OrderType = OrderType.LIMIT,
                    price: Optional[float] = None,
                    time_in_force: str = 'GTC',
                    **kwargs) -> str:
        """
        Create a new Iceberg order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Total quantity to execute
            tip_size: Fixed tip size (overrides tip_percentage if both provided)
            tip_percentage: Tip size as percentage of total quantity
            order_type: Type of order (LIMIT/MARKET)
            price: Limit price (required for LIMIT orders)
            time_in_force: Order time in force
            **kwargs: Additional order parameters
            
        Returns:
            Order ID
        """
        # Calculate tip size
        if tip_size is not None:
            # Use fixed tip size
            tip_size = max(self.min_tip_size, min(tip_size, quantity))
        elif tip_percentage is not None:
            # Calculate tip size as percentage of total quantity
            tip_size = max(self.min_tip_size, quantity * min(1.0, max(0.0, tip_percentage)))
        else:
            # Use default tip percentage
            tip_size = max(self.min_tip_size, quantity * self.default_tip_pct)
        
        # For market orders, we need to get the current market price
        if order_type == OrderType.MARKET and price is None:
            if symbol in self.order_books:
                order_book = self.order_books[symbol]
                best_bid, best_ask = order_book.get_best_bid_ask()
                price = best_ask if side == OrderSide.BUY else best_bid
                
                if price is None:
                    raise ValueError("Could not determine market price")
        
        # Create order
        order_id = f"iceberg_{self.next_order_id}"
        self.next_order_id += 1
        
        order = IcebergOrder(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            tip_size=tip_size,
            order_type=order_type,
            price=price,
            time_in_force=time_in_force,
            max_slippage_pct=self.max_slippage_pct,
            min_quantity=self.min_tip_size,
            max_attempts=self.max_attempts,
            **kwargs
        )
        
        self.active_orders[order_id] = order
        self.completed_orders[order_id] = {
            'status': 'PENDING',
            'created_at': datetime.now(pytz.utc),
            'updated_at': datetime.now(pytz.utc),
            'filled_quantity': 0.0,
            'remaining_quantity': quantity,
            'child_orders': []
        }
        
        return order_id
    
    def _get_limit_price(self, order: IcebergOrder) -> float:
        """
        Get the limit price for an order, considering slippage.
        
        Args:
            order: Iceberg order
            
        Returns:
            Limit price with slippage
        """
        if order.order_type == OrderType.MARKET:
            return order.price or 0.0  # Should have been set at creation
            
        # For limit orders, apply slippage if needed
        if order.side == OrderSide.BUY:
            return order.price * (1 + order.max_slippage_pct)
        else:  # SELL
            return order.price * (1 - order.max_slippage_pct)
    
    def execute_next_slice(self, order_id: str) -> Optional[Dict]:
        """
        Execute the next slice of an Iceberg order.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order execution details or None if no action taken
        """
        if order_id not in self.active_orders:
            return None
            
        order = self.active_orders[order_id]
        now = datetime.now(pytz.utc)
        
        # Check if we need to place a new slice
        if order.display_quantity <= 0:
            # Get next slice
            slice_qty = order.get_next_slice()
            if slice_qty is None:  # Order is complete
                self._complete_order(order_id, 'FILLED')
                return None
        else:
            # Check if we need to reprice the existing slice
            last_update = (now - order.updated_at).total_seconds()
            if last_update < self.reprice_interval:
                return None  # Too soon to reprice
                
            # Cancel the current slice to replace it
            self._cancel_current_slice(order_id)
            slice_qty = order.display_quantity  # Keep the same quantity
        
        # Place the order
        if order.order_type == OrderType.MARKET:
            # For market orders, we need to get the current price
            if order.symbol not in self.order_books:
                logger.warning(f"No order book for symbol: {order.symbol}")
                return None
                
            order_book = self.order_books[order.symbol]
            best_bid, best_ask = order_book.get_best_bid_ask()
            
            if best_bid is None or best_ask is None:
                logger.warning(f"No prices available for {order.symbol}")
                return None
                
            price = best_ask if order.side == OrderSide.BUY else best_bid
        else:
            # For limit orders, use the order price with slippage
            price = self._get_limit_price(order)
        
        # Create and execute the order
        child_order_id = f"{order_id}_child_{len(order.child_orders) + 1}"
        child_order = {
            'order_id': child_order_id,
            'symbol': order.symbol,
            'side': order.side,
            'order_type': order.order_type,
            'quantity': slice_qty,
            'price': price,
            'timestamp': now,
            'status': 'NEW',
            'attempts': order.attempts + 1
        }
        
        # In a real implementation, this would submit the order to the exchange
        # For now, we'll simulate execution with some random fill
        fill_qty = slice_qty * random.uniform(0.8, 1.0)  # 80-100% fill
        fill_price = price * random.uniform(0.999, 1.001)  # Small price variation
        
        # Update order status
        child_order.update({
            'filled_quantity': fill_qty,
            'avg_price': fill_price,
            'status': 'FILLED',
            'updated_at': now
        })
        
        # Update the iceberg order
        order.on_fill(fill_qty, fill_price)
        order.child_orders.append(child_order)
        
        # Update completed orders
        self.completed_orders[order_id].update({
            'updated_at': now,
            'filled_quantity': order.executed_quantity,
            'remaining_quantity': order.remaining_quantity,
            'status': order.status.name,
            'child_orders': order.child_orders
        })
        
        # If order is complete, move to completed orders
        if order.status in [OrderStatus.FILLED, OrderStatus.CANCELLED, OrderStatus.REJECTED]:
            self._complete_order(order_id, order.status.name)
        
        return {
            'order_id': child_order_id,
            'parent_order_id': order_id,
            'symbol': order.symbol,
            'side': order.side.name,
            'quantity': slice_qty,
            'filled_quantity': fill_qty,
            'price': price,
            'avg_price': fill_price,
            'status': 'FILLED',
            'timestamp': now.isoformat(),
            'remaining_quantity': order.remaining_quantity,
            'order_type': order.order_type.name
        }
    
    def _cancel_current_slice(self, order_id: str) -> None:
        """Cancel the current slice of an Iceberg order."""
        if order_id not in self.active_orders:
            return
            
        order = self.active_orders[order_id]
        
        if order.display_quantity > 0:
            # In a real implementation, this would cancel the active child order
            logger.info(f"Cancelling slice for order {order_id} (qty: {order.display_quantity})")
            order.display_quantity = 0.0
            order.updated_at = datetime.now(pytz.utc)
    
    def _complete_order(self, order_id: str, status: str) -> None:
        """Mark an order as completed."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            
            # Update status
            self.completed_orders[order_id].update({
                'status': status,
                'updated_at': datetime.now(pytz.utc),
                'end_time': datetime.now(pytz.utc),
                'filled_quantity': order.executed_quantity,
                'remaining_quantity': order.remaining_quantity
            })
            
            # Calculate execution metrics
            filled_value = sum(o['filled_quantity'] * o['avg_price'] 
                             for o in order.child_orders 
                             if 'filled_quantity' in o and 'avg_price' in o)
            
            if order.executed_quantity > 0:
                vwap = filled_value / order.executed_quantity
                self.completed_orders[order_id]['vwap'] = vwap
            
            # Move to completed orders
            del self.active_orders[order_id]
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get the status of an Iceberg order."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            
            # Calculate VWAP of filled orders
            filled_value = sum(o.get('filled_quantity', 0) * o.get('avg_price', 0) 
                             for o in order.child_orders 
                             if 'filled_quantity' in o and 'avg_price' in o)
            
            vwap = filled_value / order.executed_quantity if order.executed_quantity > 0 else 0.0
            
            return {
                'order_id': order_id,
                'symbol': order.symbol,
                'side': order.side.name,
                'total_quantity': order.total_quantity,
                'filled_quantity': order.executed_quantity,
                'remaining_quantity': order.remaining_quantity,
                'display_quantity': order.display_quantity,
                'tip_size': order.tip_size,
                'order_type': order.order_type.name,
                'price': order.price,
                'vwap': vwap,
                'status': order.status.name,
                'created_at': order.created_at.isoformat(),
                'updated_at': order.updated_at.isoformat(),
                'child_orders_count': len(order.child_orders)
            }
            
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
            
        return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an Iceberg order."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            
            # Cancel any active slice
            self._cancel_current_slice(order_id)
            
            # Mark order as cancelled
            order.cancel()
            self._complete_order(order_id, 'CANCELLED')
            
            return True
            
        return False
    
    def run(self) -> None:
        """Run the Iceberg executor (to be called in a loop)."""
        # Process all active orders
        for order_id in list(self.active_orders.keys()):
            self.execute_next_slice(order_id)
