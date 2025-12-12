from typing import Dict, List, Any, Optional, Deque
from collections import deque
import numpy as np
import logging
from .base_agent import BaseAgent, AgentType, AgentState
from ..interfaces.order_book import OrderSide

logger = logging.getLogger(__name__)

class TrendFollowingAgent(BaseAgent):
    """Trend-following agent that identifies and follows market trends."""
    
    def __init__(
        self,
        symbol: str,
        lookback_period: int = 20,
        entry_threshold: float = 0.001,  # 10bps trend confirmation
        exit_threshold: float = 0.0005,  # 5bps trend reversal
        position_size: float = 5.0,
        max_position: float = 50.0,
        stop_loss_pct: float = 0.01,  # 1% stop loss
        take_profit_pct: float = 0.02,  # 2% take profit
        **kwargs
    ):
        """
        Initialize the trend-following agent.
        
        Args:
            symbol: Trading symbol (e.g., 'BTC-USD')
            lookback_period: Number of periods to consider for trend calculation
            entry_threshold: Minimum trend strength to enter a position
            exit_threshold: Trend reversal threshold to exit a position
            position_size: Base position size in base currency
            max_position: Maximum absolute position size
            stop_loss_pct: Stop loss as percentage of entry price
            take_profit_pct: Take profit as percentage of entry price
        """
        super().__init__(agent_type=AgentType.TREND_FOLLOWER, **kwargs)
        self.symbol = symbol
        self.lookback_period = lookback_period
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.position_size = position_size
        self.max_position = max_position
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
        # Track price history
        self.price_history: Deque[float] = deque(maxlen=lookback_period + 1)
        self.position = 0.0
        self.entry_price = 0.0
        self.current_pnl = 0.0
        
    def decide_actions(self, market_data: Dict[str, Any], timestamp: int) -> List[Dict]:
        """
        Determine actions based on trend analysis.
        """
        if self.symbol not in market_data.get('prices', {}):
            return []
            
        current_price = market_data['prices'][self.symbol]
        if current_price <= 0:
            return []
            
        # Update price history
        self.price_history.append(current_price)
        
        # Need at least lookback_period + 1 points to calculate trends
        if len(self.price_history) <= self.lookback_period:
            return []
            
        # Calculate trend using linear regression
        x = np.arange(len(self.price_history))
        y = np.array(self.price_history)
        z = np.polyfit(x, y, 1)
        trend_slope = z[0] / current_price  # Normalize by price
        
        # Get current position
        current_position = self.state.positions.get(self.symbol, 0.0)
        
        # Check for stop loss or take profit
        if current_position != 0:
            price_change = (current_price - self.entry_price) / self.entry_price
            
            # Check stop loss
            if (current_position > 0 and price_change < -self.stop_loss_pct) or \
               (current_position < 0 and price_change > self.stop_loss_pct):
                # Exit position due to stop loss
                return self._create_market_order(-current_position, 'STOP_LOSS')
                
            # Check take profit
            if (current_position > 0 and price_change > self.take_profit_pct) or \
               (current_position < 0 and price_change < -self.take_profit_pct):
                # Exit position due to take profit
                return self._create_market_order(-current_position, 'TAKE_PROFIT')
        
        # Determine position sizing based on trend strength
        position_size = min(
            self.position_size * (1.0 + abs(trend_slope) * 100),  # Scale with trend strength
            self.max_position - abs(current_position)  # Respect position limits
        )
        
        # Generate trading signals
        if trend_slope > self.entry_threshold and current_position <= 0:
            # Uptrend detected and we're not already long
            if current_position < 0:
                # Close short position first
                return self._create_market_order(-current_position, 'CLOSE_SHORT')
            else:
                # Enter long position
                return self._create_market_order(position_size, 'LONG')
                
        elif trend_slope < -self.entry_threshold and current_position >= 0:
            # Downtrend detected and we're not already short
            if current_position > 0:
                # Close long position first
                return self._create_market_order(-current_position, 'CLOSE_LONG')
            else:
                # Enter short position
                return self._create_market_order(-position_size, 'SHORT')
                
        elif abs(trend_slope) < self.exit_threshold and current_position != 0:
            # Trend is weakening, exit position
            return self._create_market_order(-current_position, 'EXIT_TREND_END')
            
        return []
    
    def _create_market_order(self, quantity: float, order_type: str) -> List[Dict]:
        """Create a market order."""
        if abs(quantity) < 1e-8:  # Avoid floating point precision issues
            return []
            
        side = 'BUY' if quantity > 0 else 'SELL'
        order = {
            'symbol': self.symbol,
            'side': side,
            'price': None,  # Market order
            'quantity': abs(quantity),
            'order_id': f"{self.agent_id}_{order_type}_{int(time.time() * 1000)}",
            'agent_id': self.agent_id,
            'timestamp': int(time.time() * 1000)
        }
        
        # Update position tracking
        self.position += quantity
        if abs(quantity) > 1e-8:  # Only update entry price for non-zero quantities
            self.entry_price = (self.entry_price * (self.position - quantity) + 
                              current_price * quantity) / self.position
        
        return [order]
    
    def update_state(self, market_data: Dict[str, Any], timestamp: int) -> None:
        """Update agent state based on market data and trades."""
        super().update_state(market_data, timestamp)
        
        # Update current position and PnL
        current_position = self.state.positions.get(self.symbol, 0.0)
        current_price = market_data['prices'].get(self.symbol, 0)
        
        if current_position != 0 and self.entry_price != 0:
            self.current_pnl = (current_price - self.entry_price) / self.entry_price * 100  # in %
            if current_position < 0:  # Short position
                self.current_pnl *= -1
        else:
            self.current_pnl = 0.0
        
        # Update metrics
        self.metrics.update({
            'position': current_position,
            'entry_price': self.entry_price,
            'current_pnl_pct': self.current_pnl,
            'timestamp': timestamp
        })
