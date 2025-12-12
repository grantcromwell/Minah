"""
VWAP (Volume-Weighted Average Price) execution algorithm.
"""
from __future__ import annotations

import time
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import pytz

from ....interfaces.order_book import Order, OrderType, OrderSide
from ....data import MarketDataConnector, DataPipeline

logger = logging.getLogger(__name__)

@dataclass
class VWAPOrder:
    """VWAP order configuration."""
    symbol: str
    side: OrderSide
    total_quantity: float
    start_time: datetime
    end_time: datetime
    order_type: OrderType = OrderType.LIMIT
    price_band_pct: float = 0.01  # 1% price band around market price
    min_slice_size: float = 0.01  # Minimum order size
    max_slice_size: float = 100.0  # Maximum order size per slice
    volume_profile: Optional[Dict[float, float]] = None  # Time of day -> volume percentage
    
    def __post_init__(self):
        """Validate and initialize order parameters."""
        if self.total_quantity <= 0:
            raise ValueError("Total quantity must be positive")
        if self.start_time >= self.end_time:
            raise ValueError("Start time must be before end time")
            
        self.duration_seconds = (self.end_time - self.start_time).total_seconds()
        if self.duration_seconds <= 0:
            raise ValueError("Order duration must be positive")
            
        self.remaining_quantity = self.total_quantity
        self.slices: List[Tuple[float, float]] = []  # (time_offset, quantity) pairs
        self._generate_volume_profile()
        self._generate_slices()
    
    def _generate_volume_profile(self) -> None:
        """Generate volume profile if not provided."""
        if self.volume_profile is not None:
            return
            
        # Default volume profile (U-shaped: higher volume at open and close)
        # This is a simple approximation - in practice, you'd use historical data
        self.volume_profile = {}
        total_hours = self.duration_seconds / 3600
        
        # Generate a U-shaped volume profile
        for i in range(int(self.duration_seconds // 60)):  # Minute intervals
            t = i / (self.duration_seconds / 60)  # 0 to 1
            # U-shape: higher at start and end, lower in the middle
            volume_pct = 0.5 * (1 - np.cos(2 * np.pi * t)) + 0.5  # 0 to 1
            self.volume_profile[i] = volume_pct
        
        # Normalize to sum to 1
        total = sum(self.volume_profile.values())
        if total > 0:
            self.volume_profile = {k: v/total for k, v in self.volume_profile.items()}
    
    def _generate_slices(self) -> None:
        """Generate order slices based on VWAP parameters."""
        if not self.volume_profile:
            self._generate_volume_profile()
            
        # Sort time intervals
        intervals = sorted(self.volume_profile.items(), key=lambda x: x[0])
        
        # Calculate quantities for each interval
        for interval, volume_pct in intervals:
            qty = self.total_quantity * volume_pct
            if qty < self.min_slice_size and qty < self.total_quantity:
                # Skip very small slices unless it's the only one
                continue
                
            time_offset = interval * 60  # Convert minutes to seconds
            self.slices.append((time_offset, qty))
        
        # Sort by time offset
        self.slices.sort(key=lambda x: x[0])
        
        # Ensure we don't exceed max slice size
        adjusted_slices = []
        for time_offset, qty in self.slices:
            if qty > self.max_slice_size:
                # Split into multiple slices
                num_slices = int(np.ceil(qty / self.max_slice_size))
                slice_qty = qty / num_slices
                for i in range(num_slices):
                    adjusted_slices.append((time_offset + i * 0.1, slice_qty))  # Small time offset between slices
            else:
                adjusted_slices.append((time_offset, qty))
                
        self.slices = adjusted_slices
    
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

class VWAPExecutor:
    """
    VWAP (Volume-Weighted Average Price) execution algorithm.
    
    This algorithm slices a large order based on historical volume profile
    to minimize market impact by executing more when volume is typically higher.
    """
    
    def __init__(self, 
                 order_books: Dict[str, Any],
                 data_connector: Optional[MarketDataConnector] = None,
                 data_pipeline: Optional[DataPipeline] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the VWAP executor.
        
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
        self.active_orders: Dict[str, VWAPOrder] = {}
        self.completed_orders: Dict[str, Dict] = {}
        self.next_order_id = 1
        
        # Default configuration
        self.default_duration = self.config.get('default_duration', 3600)  # 1 hour
        self.max_slice_size = self.config.get('max_slice_size', 100.0)
        self.min_slice_size = self.config.get('min_slice_size', 0.01)
        self.price_band_pct = self.config.get('price_band_pct', 0.01)  # 1%
        self.historical_days = self.config.get('historical_days', 30)  # Days of history for volume profile
    
    def create_order(self,
                    symbol: str,
                    side: OrderSide,
                    quantity: float,
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    duration_seconds: Optional[float] = None,
                    order_type: OrderType = OrderType.LIMIT,
                    volume_profile: Optional[Dict[float, float]] = None,
                    **kwargs) -> str:
        """
        Create a new VWAP order.
        
        Args:
            symbol: Trading symbol
            side: Order side (BUY/SELL)
            quantity: Total quantity to execute
            start_time: Optional start time (default: now)
            end_time: Optional end time (if provided, overrides duration_seconds)
            duration_seconds: Duration of the VWAP in seconds (if end_time not provided)
            order_type: Type of order (LIMIT/MARKET)
            volume_profile: Optional volume profile (time of day -> volume percentage)
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
            duration_seconds = duration_seconds or self.default_duration
            end_time = start_time + timedelta(seconds=duration_seconds)
        
        # If no volume profile provided, generate one from historical data
        if volume_profile is None:
            volume_profile = self._generate_historical_volume_profile(
                symbol, 
                start_time, 
                end_time
            )
        
        # Create order
        order_id = f"vwap_{self.next_order_id}"
        self.next_order_id += 1
        
        order = VWAPOrder(
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            start_time=start_time,
            end_time=end_time,
            order_type=order_type,
            price_band_pct=self.price_band_pct,
            min_slice_size=self.min_slice_size,
            max_slice_size=self.max_slice_size,
            volume_profile=volume_profile,
            **kwargs
        )
        
        self.active_orders[order_id] = order
        self.completed_orders[order_id] = {
            'status': 'PENDING',
            'created_at': now,
            'updated_at': now,
            'filled_quantity': 0.0,
            'remaining_quantity': quantity,
            'slices': [],
            'volume_profile': volume_profile
        }
        
        return order_id
    
    def _generate_historical_volume_profile(self, 
                                          symbol: str,
                                          start_time: datetime,
                                          end_time: datetime) -> Dict[float, float]:
        """
        Generate volume profile from historical data.
        
        Args:
            symbol: Trading symbol
            start_time: Start time of the order
            end_time: End time of the order
            
        Returns:
            Dictionary of time offset in minutes -> volume percentage
        """
        # In a real implementation, this would query historical data
        # For now, we'll return a default U-shaped profile
        duration_minutes = int((end_time - start_time).total_seconds() / 60)
        volume_profile = {}
        
        # Generate a U-shaped volume profile (higher at open and close)
        for i in range(duration_minutes):
            t = i / duration_minutes  # 0 to 1
            # U-shape: higher at start and end, lower in the middle
            volume_pct = 0.5 * (1 - np.cos(2 * np.pi * t)) + 0.5  # 0 to 1
            volume_profile[i] = volume_pct
        
        # Normalize to sum to 1
        total = sum(volume_profile.values())
        if total > 0:
            volume_profile = {k: v/total for k, v in volume_profile.items()}
            
        return volume_profile
    
    def execute_next_slice(self, order_id: str) -> Optional[Dict]:
        """
        Execute the next slice of a VWAP order.
        
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
                'vwap_slice': True,
                'slice_number': len(self.completed_orders[order_id]['slices']) + 1,
                'total_slices': len(order.slices)
            }
        )
        
        # In a real implementation, this would submit the order to the exchange
        # For now, we'll just log it
        logger.info(f"Executing VWAP slice: {slice_order}")
        
        # Update order status
        self.completed_orders[order_id]['slices'].append({
            'order_id': slice_order.order_id,
            'quantity': quantity,
            'price': price,
            'timestamp': now.isoformat(),
            'status': 'EXECUTED',
            'time_offset_minutes': time_offset / 60  # Convert to minutes
        })
        
        self.completed_orders[order_id]['updated_at'] = now
        self.completed_orders[order_id]['filled_quantity'] += quantity
        self.completed_orders[order_id]['remaining_quantity'] = order.remaining_quantity
        
        # Calculate VWAP so far
        filled_qty = self.completed_orders[order_id]['filled_quantity']
        if filled_qty > 0:
            vwap = sum(s['quantity'] * s['price'] for s in self.completed_orders[order_id]['slices']) / filled_qty
            self.completed_orders[order_id]['current_vwap'] = vwap
        
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
            'total_slices': len(order.slices) + len(self.completed_orders[order_id]['slices']),
            'time_offset_minutes': time_offset / 60  # Convert to minutes
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
            
            # Calculate final VWAP
            filled_qty = self.completed_orders[order_id]['filled_quantity']
            if filled_qty > 0:
                vwap = sum(s['quantity'] * s['price'] 
                          for s in self.completed_orders[order_id]['slices']) / filled_qty
                self.completed_orders[order_id]['final_vwap'] = vwap
            
            del self.active_orders[order_id]
    
    def get_order_status(self, order_id: str) -> Optional[Dict]:
        """Get the status of a VWAP order."""
        if order_id in self.active_orders:
            order = self.active_orders[order_id]
            status = {
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
            
            # Add VWAP if available
            if 'current_vwap' in self.completed_orders[order_id]:
                status['current_vwap'] = self.completed_orders[order_id]['current_vwap']
                
            return status
            
        elif order_id in self.completed_orders:
            return self.completed_orders[order_id]
            
        return None
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a VWAP order."""
        if order_id in self.active_orders:
            self._complete_order(order_id, 'CANCELLED')
            return True
        return False
    
    def run(self) -> None:
        """Run the VWAP executor (to be called in a loop)."""
        now = datetime.now(pytz.utc)
        
        # Process all active orders
        for order_id in list(self.active_orders.keys()):
            self.execute_next_slice(order_id)
            
            # Clean up completed orders
            if order_id in self.active_orders and self.active_orders[order_id].remaining_quantity <= 0:
                self._complete_order(order_id, 'FILLED')
