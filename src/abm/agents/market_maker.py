from typing import Dict, List, Any, Optional
import numpy as np
import logging
from .base_agent import BaseAgent, AgentType, AgentState
from ..interfaces.order_book import OrderSide

logger = logging.getLogger(__name__)

class MarketMakerAgent(BaseAgent):
    """Market maker agent that provides liquidity by placing limit orders around the mid price."""
    
    def __init__(
        self,
        symbol: str,
        spread_target: float = 0.001,  # 10bps target spread
        order_size: float = 1.0,
        inventory_target: float = 0.0,
        max_position: float = 100.0,
        **kwargs
    ):
        """
        Initialize the market maker agent.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            spread_target: Target spread as a fraction of mid price
            order_size: Size of each order in base currency
            inventory_target: Target inventory position
            max_position: Maximum absolute position size
        """
        super().__init__(agent_type=AgentType.MARKET_MAKER, **kwargs)
        self.symbol = symbol
        self.spread_target = spread_target
        self.order_size = order_size
        self.inventory_target = inventory_target
        self.max_position = max_position
        self.active_orders = set()
        
    def decide_actions(self, market_data: Dict[str, Any], timestamp: int) -> List[Dict]:
        """
        Determine actions based on current market conditions.
        """
        if self.symbol not in market_data.get('prices', {}):
            return []
            
        mid_price = market_data['prices'][self.symbol]
        if mid_price <= 0:
            return []
            
        # Calculate inventory adjustment factor
        current_position = self.state.positions.get(self.symbol, 0.0)
        inventory_imbalance = (current_position - self.inventory_target) / self.max_position
        
        # Adjust spread based on inventory
        spread_adjustment = 1.0 + abs(inventory_imbalance) * 2.0  # Widen spread as inventory deviates from target
        half_spread = (self.spread_target * spread_adjustment) / 2.0
        
        # Calculate bid and ask prices
        bid_price = mid_price * (1.0 - half_spread)
        ask_price = mid_price * (1.0 + half_spread)
        
        # Adjust prices based on inventory
        if inventory_imbalance > 0:
            # Reduce bid price to reduce long position
            bid_price *= (1.0 - 0.2 * inventory_imbalance)
            # Increase ask price to reduce long position
            ask_price *= (1.0 + 0.1 * inventory_imbalance)
        elif inventory_imbalance < 0:
            # Increase ask price to reduce short position
            ask_price *= (1.0 + 0.2 * abs(inventory_imbalance))
            # Reduce bid price to reduce short position
            bid_price *= (1.0 - 0.1 * abs(inventory_imbalance))
        
        # Calculate order sizes (adjust based on inventory)
        bid_size = self.order_size * (1.0 - 0.5 * inventory_imbalance)
        ask_size = self.order_size * (1.0 + 0.5 * inventory_imbalance)
        
        # Ensure we don't exceed position limits
        if current_position + ask_size > self.max_position:
            ask_size = max(0, self.max_position - current_position)
        if current_position - bid_size < -self.max_position:
            bid_size = max(0, current_position + self.max_position)
        
        # Create orders
        orders = []
        
        # Only place orders if we have enough cash/position
        if bid_size > 0 and (current_position - bid_size) >= -self.max_position:
            orders.append({
                'symbol': self.symbol,
                'side': 'BUY',
                'price': bid_price,
                'quantity': bid_size,
                'order_id': f"{self.agent_id}_bid_{timestamp}",
                'agent_id': self.agent_id,
                'timestamp': timestamp
            })
        
        if ask_size > 0 and (current_position + ask_size) <= self.max_position:
            orders.append({
                'symbol': self.symbol,
                'side': 'SELL',
                'price': ask_price,
                'quantity': ask_size,
                'order_id': f"{self.agent_id}_ask_{timestamp}",
                'agent_id': self.agent_id,
                'timestamp': timestamp
            })
        
        # Cancel any existing orders
        self._cancel_all_orders()
        self.active_orders = {order['order_id'] for order in orders}
        
        return orders
    
    def _cancel_all_orders(self) -> None:
        """Cancel all active orders."""
        # In a real implementation, we would send cancel requests for each order
        self.active_orders.clear()
    
    def update_state(self, market_data: Dict[str, Any], timestamp: int) -> None:
        """Update agent state based on market data and trades."""
        super().update_state(market_data, timestamp)
        
        # Update metrics
        current_position = self.state.positions.get(self.symbol, 0.0)
        self.metrics.update({
            'position': current_position,
            'inventory_imbalance': (current_position - self.inventory_target) / self.max_position,
            'timestamp': timestamp
        })
