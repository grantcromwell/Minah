from typing import Dict, List, Optional, Tuple, DefaultDict, Any
from dataclasses import dataclass, field
from enum import Enum
import heapq
from collections import defaultdict
import numpy as np
import logging

logger = logging.getLogger(__name__)

class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"



class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

@dataclass(order=True)
class Order:
    """Represents a single order in the order book."""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    timestamp: Optional[float] = None
    agent_id: str = field(default="unknown", compare=False)

    def __post_init__(self):
        # Set timestamp if not provided
        if self.timestamp is None:
            import time
            self.timestamp = time.time()

        # For buy orders, we want highest price first (min-heap with negative price)
        # For sell orders, we want lowest price first (min-heap with positive price)
        if self.price is not None:
            if self.side == OrderSide.BUY:
                self.price = -abs(self.price)
            else:
                self.price = abs(self.price)

class OrderBook:
    """Order book implementation for a single trading pair."""
    
    def __init__(self, symbol: str, tick_size: float = 0.01, lot_size: float = 0.01):
        """
        Initialize the order book.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            tick_size: Minimum price movement
            lot_size: Minimum order size
        """
        self.symbol = symbol
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.buy_orders: List[Order] = []  # Max-heap for buy orders (highest bid first)
        self.sell_orders: List[Order] = []  # Min-heap for sell orders (lowest ask first)
        self.orders: Dict[str, Order] = {}  # All active orders by order_id
        self.trade_history: List[Dict] = []
        self.order_sequence = 0
        
        # Market data
        self.last_price: Optional[float] = None
        self.volume_24h: float = 0.0
        self.high_price: Optional[float] = None
        self.low_price: Optional[float] = None
        self.vwap: float = 0.0
        self.vwap_volume: float = 0.0
    
    def process_order(self, order_data: Dict) -> List[Dict]:
        """
        Process a new order.
        
        Args:
            order_data: Dictionary containing order details
                - side: 'BUY' or 'SELL'
                - price: Order price (None for market orders)
                - quantity: Order quantity
                - order_id: Optional order ID
                - agent_id: ID of the agent placing the order
                - timestamp: Optional timestamp (defaults to current time)
                
        Returns:
            List of trades that occurred as a result of this order
        """
        # Validate order data
        side = OrderSide(order_data.get('side').upper())
        price = order_data.get('price')
        quantity = float(order_data.get('quantity', 0))
        order_id = order_data.get('order_id', f"order_{self.order_sequence}")
        agent_id = order_data.get('agent_id', 'unknown')
        timestamp = order_data.get('timestamp', self._current_time())
        
        # Validate quantity
        if quantity <= 0 or quantity < self.lot_size:
            logger.warning(f"Invalid quantity {quantity} for {order_id}")
            return []
        
        # Round price to nearest tick
        if price is not None:
            price = round(price / self.tick_size) * self.tick_size
            price = max(0, price)  # Ensure non-negative price
        
        # Create order
        order = Order(
            price=price if price is not None else (0.0 if side == OrderSide.BUY else float('inf')),
            quantity=quantity,
            order_id=order_id,
            agent_id=agent_id,
            timestamp=timestamp,
            side=side
        )
        
        # Process order
        trades = []
        if side == OrderSide.BUY:
            trades = self._match_buy_order(order)
        else:
            trades = self._match_sell_order(order)
        
        # Add to order book if not fully filled
        if order.quantity > 0 and price is not None:
            self._add_to_order_book(order)
        
        self.orders[order_id] = order
        self.order_sequence += 1
        
        return trades
    
    def _match_buy_order(self, order: Order) -> List[Dict]:
        """Match a buy order against the sell side of the book."""
        trades = []
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0 and self.sell_orders:
            best_ask = self.sell_orders[0]
            
            # Check if we can match (buy price >= ask price)
            if order.price < best_ask.price and order.price != 0:  # 0 indicates market order
                break
            
            # Calculate trade quantity
            trade_qty = min(remaining_quantity, best_ask.quantity)
            trade_price = best_ask.price  # Price of the order in the book
            
            # Execute trade
            self._execute_trade(
                buy_order=order,
                sell_order=best_ask,
                price=trade_price,
                quantity=trade_qty,
                trades=trades
            )
            
            # Update remaining quantity
            remaining_quantity -= trade_qty
            
            # Remove filled orders from the book
            if best_ask.quantity <= 0:
                heapq.heappop(self.sell_orders)
                del self.orders[best_ask.order_id]
            
            # Stop if this was a market order and we're done
            if order.price == 0 and remaining_quantity <= 0:
                break
        
        order.quantity = remaining_quantity
        return trades
    
    def _match_sell_order(self, order: Order) -> List[Dict]:
        """Match a sell order against the buy side of the book."""
        trades = []
        remaining_quantity = order.quantity
        
        while remaining_quantity > 0 and self.buy_orders:
            best_bid = self.buy_orders[0]
            
            # Check if we can match (sell price <= bid price)
            if order.price > -best_bid.price and order.price != float('inf'):  # inf indicates market order
                break
            
            # Calculate trade quantity
            trade_qty = min(remaining_quantity, best_bid.quantity)
            trade_price = -best_bid.price  # Convert back to positive price
            
            # Execute trade
            self._execute_trade(
                buy_order=best_bid,
                sell_order=order,
                price=trade_price,
                quantity=trade_qty,
                trades=trades
            )
            
            # Update remaining quantity
            remaining_quantity -= trade_qty
            
            # Remove filled orders from the book
            if best_bid.quantity <= 0:
                heapq.heappop(self.buy_orders)
                del self.orders[best_bid.order_id]
            
            # Stop if this was a market order and we're done
            if order.price == float('inf') and remaining_quantity <= 0:
                break
        
        order.quantity = remaining_quantity
        return trades
    
    def _execute_trade(
        self,
        buy_order: Order,
        sell_order: Order,
        price: float,
        quantity: float,
        trades: List[Dict]
    ) -> None:
        """Execute a trade between a buy and sell order."""
        # Update order quantities
        buy_order.quantity -= quantity
        sell_order.quantity -= quantity
        
        # Record trade
        trade = {
            'symbol': self.symbol,
            'price': price,
            'quantity': quantity,
            'buy_order_id': buy_order.order_id,
            'sell_order_id': sell_order.order_id,
            'buyer_id': buy_order.agent_id,
            'seller_id': sell_order.agent_id,
            'timestamp': self._current_time(),
            'trade_id': f"trade_{len(self.trade_history) + 1}"
        }
        
        # Update market data
        self._update_market_data(price, quantity)
        
        # Add to trade history
        self.trade_history.append(trade)
        trades.append(trade)
    
    def _update_market_data(self, price: float, quantity: float) -> None:
        """Update market data after a trade."""
        # Update last price
        self.last_price = price
        
        # Update high/low prices
        if self.high_price is None or price > self.high_price:
            self.high_price = price
        if self.low_price is None or price < self.low_price:
            self.low_price = price
        
        # Update VWAP (Volume Weighted Average Price)
        self.vwap = ((self.vwap * self.vwap_volume) + (price * quantity)) / (self.vwap_volume + quantity)
        self.vwap_volume += quantity
        
        # Update 24h volume (simplified)
        self.volume_24h += quantity
    
    def _add_to_order_book(self, order: Order) -> None:
        """Add an order to the appropriate side of the book."""
        if order.side == OrderSide.BUY:
            heapq.heappush(self.buy_orders, order)
        else:
            heapq.heappush(self.sell_orders, order)
    
    def get_best_bid(self) -> Optional[float]:
        """Get the current best bid price."""
        if not self.buy_orders:
            return None
        return -self.buy_orders[0].price if self.buy_orders[0].quantity > 0 else None
    
    def get_best_ask(self) -> Optional[float]:
        """Get the current best ask price."""
        if not self.sell_orders:
            return None
        return self.sell_orders[0].price if self.sell_orders[0].quantity > 0 else None
    
    def get_mid_price(self) -> Optional[float]:
        """Get the current mid price (average of best bid and ask)."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if bid is None or ask is None:
            return self.last_price
            
        return (bid + ask) / 2
    
    def get_spread(self) -> float:
        """Get the current bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        
        if bid is None or ask is None:
            return float('inf')
            
        return ask - bid
    
    def get_order_imbalance(self) -> float:
        """Calculate the order book imbalance."""
        total_bid_volume = sum(order.quantity for order in self.buy_orders)
        total_ask_volume = sum(order.quantity for order in self.sell_orders)
        
        if total_bid_volume + total_ask_volume == 0:
            return 0.0
            
        return (total_bid_volume - total_ask_volume) / (total_bid_volume + total_ask_volume)
    
    def get_market_depth(self, levels: int = 10) -> Dict[str, List[Tuple[float, float]]]:
        """Get the current market depth up to the specified number of levels."""
        def get_top_orders(orders, n, invert=False):
            # Get top n orders by price (handling the heap structure)
            top_orders = []
            temp_heap = []
            
            while orders and len(top_orders) < n:
                order = heapq.heappop(orders)
                if order.quantity > 0:  # Only include orders with remaining quantity
                    price = -order.price if invert else order.price
                    top_orders.append((price, order.quantity))
                temp_heap.append(order)
            
            # Restore the heap
            for order in temp_heap:
                heapq.heappush(orders, order)
                
            return top_orders
        
        bids = get_top_orders(self.buy_orders.copy(), levels, invert=True)
        asks = get_top_orders(self.sell_orders.copy(), levels, invert=False)
        
        return {
            'bids': bids,
            'asks': asks
        }
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an existing order."""
        if order_id not in self.orders:
            return False
            
        order = self.orders[order_id]
        order.quantity = 0  # Mark as cancelled
        
        # The order will be removed from the book during the next matching cycle
        return True
    
    def reset(self) -> None:
        """Reset the order book to its initial state."""
        self.buy_orders = []
        self.sell_orders = []
        self.orders = {}
        self.trade_history = []
        self.order_sequence = 0
        self.last_price = None
        self.volume_24h = 0.0
        self.high_price = None
        self.low_price = None
        self.vwap = 0.0
        self.vwap_volume = 0.0
    
    def _current_time(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
