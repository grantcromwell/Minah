from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass, field
import uuid
from enum import Enum

class AgentType(Enum):
    MARKET_MAKER = "market_maker"
    TREND_FOLLOWER = "trend_follower"
    MEAN_REVERTER = "mean_reverter"
    ARBITRAGEUR = "arbitrageur"
    LIQUIDITY_CONSUMER = "liquidity_consumer"

@dataclass
class AgentState:
    cash: float = 0.0
    positions: Dict[str, float] = field(default_factory=dict)
    pnl: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

class BaseAgent(ABC):
    """Base class for all agents in the ABM system."""
    
    def __init__(self, 
                 agent_id: str = None, 
                 agent_type: AgentType = None,
                 initial_cash: float = 1_000_000.0,
                 risk_aversion: float = 0.5,
                 **kwargs):
        """
        Initialize the base agent.
        
        Args:
            agent_id: Unique identifier for the agent
            agent_type: Type of the agent from AgentType enum
            initial_cash: Initial cash balance
            risk_aversion: Risk aversion coefficient (0-1)
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_type = agent_type
        self.state = AgentState(cash=initial_cash)
        self.risk_aversion = max(0.0, min(1.0, risk_aversion))
        self.history = []
        self.metrics = {}
        
    @abstractmethod
    def decide_actions(self, market_data: Dict[str, Any], timestamp: int) -> List[Dict]:
        """
        Determine actions to take based on current market state.
        
        Args:
            market_data: Current market data snapshot
            timestamp: Current simulation timestamp
            
        Returns:
            List of order actions (dicts with order details)
        """
        pass
    
    def update_state(self, market_data: Dict[str, Any], timestamp: int) -> None:
        """
        Update agent's internal state based on market data.
        
        Args:
            market_data: Current market data snapshot
            timestamp: Current simulation timestamp
        """
        # Update positions and PnL based on latest market data
        self._update_pnl(market_data)
        self._record_metrics()
        
    def _update_pnl(self, market_data: Dict[str, Any]) -> None:
        """Update PnL based on current positions and market data."""
        if not hasattr(self, 'state'):
            self.state = AgentState()
            
        # Mark positions to market
        mtm_value = 0.0
        for symbol, position in self.state.positions.items():
            if symbol in market_data.get('prices', {}):
                mtm_value += position * market_data['prices'][symbol]
                
        # Update PnL (unrealized + realized)
        self.state.pnl = (self.state.cash + mtm_value) - 1_000_000.0  # Assuming initial value
    
    def _record_metrics(self) -> None:
        """Record current metrics for performance analysis."""
        self.metrics = {
            'pnl': self.state.pnl,
            'cash': self.state.cash,
            'positions': self.state.positions.copy(),
            'timestamp': len(self.history)
        }
        self.history.append(self.metrics)
    
    def get_state_summary(self) -> Dict:
        """Get a summary of the agent's current state."""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type.value if self.agent_type else None,
            'cash': self.state.cash,
            'positions': self.state.positions,
            'pnl': self.state.pnl,
            'metrics': self.metrics
        }
