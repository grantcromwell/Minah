""
Trading Environment for Reinforcement Learning

This module implements a trading environment compatible with OpenAI Gym's interface.
"""

import gym
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from collections import deque

logger = logging.getLogger(__name__)

class ActionType(Enum):
    """Types of actions an agent can take."""
    HOLD = 0
    BUY = 1
    SELL = 2
    CLOSE_POSITION = 3

@dataclass
class Position:
    """Represents an open position in the market."""
    entry_price: float
    size: float
    entry_time: int
    position_type: str  # 'LONG' or 'SHORT'
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    @property
    def is_long(self) -> bool:
        return self.position_type == 'LONG'
    
    @property
    def is_short(self) -> bool:
        return self.position_type == 'SHORT'

@dataclass
class TradingState:
    """Current state of the trading environment."""
    current_step: int = 0
    current_price: float = 0.0
    portfolio_value: float = 0.0
    cash: float = 0.0
    positions: List[Position] = field(default_factory=list)
    total_fees: float = 0.0
    n_trades: int = 0
    
    @property
    def total_position_size(self) -> float:
        return sum(pos.size for pos in self.positions)
    
    @property
    def has_position(self) -> bool:
        return len(self.positions) > 0
    
    @property
    def current_position_type(self) -> Optional[str]:
        return self.positions[0].position_type if self.positions else None

class TradingEnvironment(gym.Env):
    """
    A trading environment for reinforcement learning.
    
    This environment simulates trading with features like position management,
    transaction costs, and realistic order execution.
    """
    
    metadata = {'render.modes': ['human', 'rgb_array']}
    
    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 10000.0,
        commission: float = 0.001,  # 0.1% commission per trade
        slippage: float = 0.0005,   # 0.05% slippage
        max_position_size: float = 1.0,  # Max position size as fraction of portfolio
        max_trade_size: float = 0.1,     # Max trade size as fraction of portfolio
        window_size: int = 10,      # Number of past time steps to include in state
        reward_type: str = 'sharpe',  # 'pnl', 'sharpe', 'sortino', 'calmar'
        render_mode: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize the trading environment.
        
        Args:
            data: DataFrame with OHLCV data and features
            initial_balance: Initial account balance
            commission: Trading commission as a fraction of trade value
            slippage: Slippage as a fraction of trade value
            max_position_size: Maximum position size as a fraction of portfolio
            max_trade_size: Maximum trade size as a fraction of portfolio
            window_size: Number of past time steps to include in state
            reward_type: Type of reward function to use
            render_mode: Rendering mode ('human' or 'rgb_array')
        """
        super().__init__()
        
        self.data = data
        self.initial_balance = initial_balance
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.max_trade_size = max_trade_size
        self.window_size = window_size
        self.reward_type = reward_type
        self.render_mode = render_mode
        
        # Extract price and feature columns
        self.price_columns = ['open', 'high', 'low', 'close', 'volume']
        self.feature_columns = [col for col in data.columns if col not in self.price_columns]
        
        # Define action and observation spaces
        self.action_space = gym.spaces.Discrete(len(ActionType))
        
        # Observation space: OHLCV + features + portfolio state
        obs_shape = (len(self.price_columns) + len(self.feature_columns) + 3,)  # +3 for position, cash, portfolio_value
        self.observation_space = gym.spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=obs_shape,
            dtype=np.float32
        )
        
        # Initialize state
        self.state = None
        self.current_step = None
        self.done = None
        self.episode_returns = []
        self.trade_history = []
        self.portfolio_history = []
        
        # For rendering
        self.fig = None
        self.ax = None
    
    def reset(self) -> np.ndarray:
        """Reset the environment to initial state."""
        self.current_step = self.window_size  # Start after the window
        self.done = False
        self.episode_returns = []
        self.trade_history = []
        self.portfolio_history = []
        
        # Initialize state
        self.state = TradingState(
            current_step=self.current_step,
            current_price=self._get_current_price(),
            portfolio_value=self.initial_balance,
            cash=self.initial_balance,
            positions=[],
            total_fees=0.0,
            n_trades=0
        )
        
        # Record initial portfolio value
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': self.state.portfolio_value,
            'cash': self.state.cash,
            'position_value': 0.0,
            'price': self.state.current_price
        })
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """
        Execute one time step within the environment.
        
        Args:
            action: The action to take (from ActionType enum)
            
        Returns:
            observation: The new observation
            reward: The reward for the action
            done: Whether the episode is complete
            info: Additional information
        """
        if self.done:
            return self._get_observation(), 0.0, True, {}
        
        # Get current state
        current_price = self._get_current_price()
        prev_portfolio_value = self._get_portfolio_value(current_price)
        
        # Execute the action
        action_type = ActionType(action)
        self._execute_action(action_type, current_price)
        
        # Update to next time step
        self.current_step += 1
        self.state.current_step = self.current_step
        self.state.current_price = self._get_current_price()
        
        # Calculate new portfolio value
        new_portfolio_value = self._get_portfolio_value(self.state.current_price)
        
        # Calculate step return
        step_return = (new_portfolio_value / prev_portfolio_value) - 1.0
        self.episode_returns.append(step_return)
        
        # Calculate reward
        reward = self._calculate_reward(step_return)
        
        # Check if episode is done
        self.done = self.current_step >= len(self.data) - 1
        
        # Record portfolio history
        self.portfolio_history.append({
            'step': self.current_step,
            'portfolio_value': new_portfolio_value,
            'cash': self.state.cash,
            'position_value': self._get_position_value(self.state.current_price),
            'price': self.state.current_price
        })
        
        # Prepare info dictionary
        info = {
            'portfolio_value': new_portfolio_value,
            'return': (new_portfolio_value / self.initial_balance) - 1.0,
            'n_trades': self.state.n_trades,
            'total_fees': self.state.total_fees,
            'current_position': self.state.current_position_type,
            'position_size': self.state.total_position_size
        }
        
        return self._get_observation(), reward, self.done, info
    
    def _execute_action(self, action: ActionType, price: float) -> None:
        """Execute a trading action."""
        if action == ActionType.HOLD:
            return
        
        position_value = self._get_position_value(price)
        
        if action == ActionType.BUY and not self.state.has_position:
            # Calculate position size based on available cash and max trade size
            max_affordable = (self.state.cash * self.max_trade_size) / price
            position_size = min(max_affordable, (self.state.portfolio_value * self.max_position_size) / price)
            
            if position_size * price < 1.0:  # Minimum trade size
                return
                
            # Calculate fees and slippage
            trade_value = position_size * price
            fees = trade_value * self.commission
            slippage_cost = trade_value * self.slippage
            total_cost = trade_value + fees + slippage_cost
            
            # Update state
            if total_cost <= self.state.cash:
                self.state.cash -= total_cost
                self.state.positions.append(Position(
                    entry_price=price,
                    size=position_size,
                    entry_time=self.current_step,
                    position_type='LONG'
                ))
                self.state.total_fees += fees + slippage_cost
                self.state.n_trades += 1
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'BUY',
                    'price': price,
                    'size': position_size,
                    'fees': fees,
                    'slippage': slippage_cost
                })
        
        elif action == ActionType.SELL and not self.state.has_position:
            # For short selling, we need to borrow the asset
            position_size = (self.state.portfolio_value * self.max_trade_size) / price
            
            if position_size * price < 1.0:  # Minimum trade size
                return
                
            # Calculate fees and slippage
            trade_value = position_size * price
            fees = trade_value * self.commission
            slippage_cost = trade_value * self.slippage
            
            # Update state
            self.state.cash += (trade_value - fees - slippage_cost)
            self.state.positions.append(Position(
                entry_price=price,
                size=position_size,
                entry_time=self.current_step,
                position_type='SHORT'
            ))
            self.state.total_fees += fees + slippage_cost
            self.state.n_trades += 1
            
            # Record trade
            self.trade_history.append({
                'step': self.current_step,
                'action': 'SELL_SHORT',
                'price': price,
                'size': position_size,
                'fees': fees,
                'slippage': slippage_cost
            })
        
        elif action == ActionType.CLOSE_POSITION and self.state.has_position:
            # Close all positions
            for position in self.state.positions:
                if position.is_long:
                    # Long position: sell to close
                    trade_value = position.size * price
                    fees = trade_value * self.commission
                    slippage_cost = trade_value * self.slippage
                    self.state.cash += (trade_value - fees - slippage_cost)
                else:
                    # Short position: buy to cover
                    trade_value = position.size * position.entry_price  # Original value
                    current_value = position.size * price
                    fees = current_value * self.commission
                    slippage_cost = current_value * self.slippage
                    self.state.cash += (trade_value - (current_value + fees + slippage_cost))
                
                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'action': 'CLOSE_' + position.position_type,
                    'price': price,
                    'size': position.size,
                    'fees': fees,
                    'slippage': slippage_cost,
                    'pnl': self._calculate_position_pnl(position, price)
                })
                
                # Update metrics
                self.state.total_fees += fees + slippage_cost
                self.state.n_trades += 1
            
            # Clear positions
            self.state.positions = []
    
    def _calculate_reward(self, step_return: float) -> float:
        """Calculate the reward for the current step."""
        if self.reward_type == 'pnl':
            # Simple PnL-based reward
            return step_return
            
        elif self.reward_type == 'sharpe':
            # Sharpe ratio (annualized)
            if len(self.episode_returns) < 2:
                return 0.0
                
            returns = np.array(self.episode_returns)
            sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9)
            return sharpe
            
        elif self.reward_type == 'sortino':
            # Sortino ratio (annualized)
            if len(self.episode_returns) < 2:
                return 0.0
                
            returns = np.array(self.episode_returns)
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0:
                sortino = np.sqrt(252) * np.mean(returns) / 1e-9
            else:
                sortino = np.sqrt(252) * np.mean(returns) / (np.std(downside_returns) + 1e-9)
            return sortino
            
        elif self.reward_type == 'calmar':
            # Calmar ratio (simplified)
            if len(self.portfolio_history) < 2:
                return 0.0
                
            portfolio_values = np.array([h['portfolio_value'] for h in self.portfolio_history])
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
            if max_drawdown == 0:
                return 0.0
                
            total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
            calmar = total_return / max_drawdown
            return calmar
            
        else:
            raise ValueError(f"Unknown reward type: {self.reward_type}")
    
    def _calculate_max_drawdown(self, values: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        peak = values[0]
        max_drawdown = 0.0
        
        for value in values[1:]:
            if value > peak:
                peak = value
            else:
                drawdown = (peak - value) / peak
                max_drawdown = max(max_drawdown, drawdown)
                
        return max_drawdown
    
    def _get_current_price(self) -> float:
        """Get the current price (close price at current step)."""
        return float(self.data.iloc[self.current_step]['close'])
    
    def _get_position_value(self, price: float) -> float:
        """Calculate the current value of all positions."""
        if not self.state.positions:
            return 0.0
            
        total_value = 0.0
        for position in self.state.positions:
            if position.is_long:
                total_value += position.size * price
            else:  # Short position
                total_value += position.size * (2 * position.entry_price - price)
                
        return total_value
    
    def _calculate_position_pnl(self, position: Position, current_price: float) -> float:
        """Calculate PnL for a position."""
        if position.is_long:
            return (current_price - position.entry_price) * position.size
        else:  # Short position
            return (position.entry_price - current_price) * position.size
    
    def _get_portfolio_value(self, price: float) -> float:
        """Calculate total portfolio value (cash + positions)."""
        return self.state.cash + self._get_position_value(price)
    
    def _get_observation(self) -> np.ndarray:
        """Get the current observation (state)."""
        # Get price and feature data for the current window
        start_idx = max(0, self.current_step - self.window_size + 1)
        end_idx = self.current_step + 1
        
        # Get price data (OHLCV)
        price_data = self.data[self.price_columns].iloc[start_idx:end_idx].values
        
        # Get feature data
        if self.feature_columns:
            feature_data = self.data[self.feature_columns].iloc[start_idx:end_idx].values
            obs_data = np.hstack([price_data, feature_data])
        else:
            obs_data = price_data
        
        # Pad with zeros if window is not full
        if obs_data.shape[0] < self.window_size:
            padding = np.zeros((self.window_size - obs_data.shape[0], obs_data.shape[1]))
            obs_data = np.vstack([padding, obs_data])
        
        # Add portfolio state (position, cash, portfolio_value)
        position_state = np.array([
            self.state.total_position_size / self.max_position_size if self.state.has_position else 0.0,
            self.state.cash / self.initial_balance,
            self.state.portfolio_value / self.initial_balance
        ])
        
        # Flatten and concatenate
        obs = np.concatenate([obs_data.flatten(), position_state])
        
        return obs.astype(np.float32)
    
    def render(self, mode: str = 'human') -> None:
        """Render the environment."""
        if mode == 'human':
            import matplotlib.pyplot as plt
            from matplotlib import gridspec
            
            if self.fig is None:
                plt.ion()
                self.fig = plt.figure(figsize=(12, 8))
                gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
                
                # Price and portfolio value
                self.ax1 = plt.subplot(gs[0])
                self.ax2 = self.ax1.twinx()
                
                # Actions
                self.ax3 = plt.subplot(gs[1])
                
                plt.tight_layout()
            
            # Update price and portfolio value
            self.ax1.clear()
            self.ax2.clear()
            self.ax3.clear()
            
            # Plot price
            price_data = self.data['close'].iloc[:self.current_step+1]
            self.ax1.plot(price_data.index, price_data, 'b-', label='Price')
            
            # Plot portfolio value (on secondary y-axis)
            portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
            steps = [h['step'] for h in self.portfolio_history]
            self.ax2.plot(steps, portfolio_values, 'g-', label='Portfolio Value')
            
            # Plot buy/sell signals
            buys = [t for t in self.trade_history if 'BUY' in t['action']]
            sells = [t for t in self.trade_history if 'SELL' in t['action']]
            
            if buys:
                buy_steps = [t['step'] for t in buys]
                buy_prices = [self.data['close'].iloc[s] for s in buy_steps]
                self.ax1.scatter(buy_steps, buy_prices, color='g', marker='^', label='Buy', s=100)
                
            if sells:
                sell_steps = [t['step'] for t in sells]
                sell_prices = [self.data['close'].iloc[s] for s in sell_steps]
                self.ax1.scatter(sell_steps, sell_prices, color='r', marker='v', label='Sell', s=100)
            
            # Plot actions
            actions = [0] * len(price_data)
            for trade in self.trade_history:
                if 'BUY' in trade['action']:
                    actions[trade['step']] = 1  # Buy
                elif 'SELL' in trade['action']:
                    actions[trade['step']] = -1  # Sell
            
            self.ax3.bar(range(len(actions)), actions, color='b', alpha=0.3)
            
            # Set labels and title
            self.ax1.set_title('Trading Environment')
            self.ax1.set_ylabel('Price')
            self.ax2.set_ylabel('Portfolio Value')
            self.ax3.set_xlabel('Step')
            self.ax3.set_ylabel('Action')
            self.ax3.set_yticks([-1, 0, 1], ['Sell', 'Hold', 'Buy'])
            
            # Add legend
            lines1, labels1 = self.ax1.get_legend_handles_labels()
            lines2, labels2 = self.ax2.get_legend_handles_labels()
            self.ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.pause(0.01)
            
        elif mode == 'rgb_array':
            # Return RGB array for video recording
            if self.fig is not None:
                self.fig.canvas.draw()
                img = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
                img = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
                return img
            return None
    
    def close(self) -> None:
        """Close the environment."""
        if self.fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self.fig)
            self.fig = None
    
    def get_metrics(self) -> Dict[str, float]:
        """Calculate performance metrics for the episode."""
        if not self.portfolio_history:
            return {}
        
        # Calculate returns
        portfolio_values = np.array([h['portfolio_value'] for h in self.portfolio_history])
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] / portfolio_values[0]) - 1.0
        annualized_return = (1 + total_return) ** (252 / len(portfolio_values)) - 1
        
        # Volatility and risk metrics
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)
            sharpe = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-9)
            
            # Sortino ratio
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0:
                downside_volatility = np.std(downside_returns) * np.sqrt(252)
                sortino = np.sqrt(252) * np.mean(returns) / (downside_volatility + 1e-9)
            else:
                sortino = np.inf
            
            # Max drawdown
            max_drawdown = self._calculate_max_drawdown(portfolio_values)
        else:
            volatility = 0.0
            sharpe = 0.0
            sortino = 0.0
            max_drawdown = 0.0
        
        # Trade metrics
        n_trades = len(self.trade_history) // 2  # Count round trips
        win_rate = 0.0
        avg_win = 0.0
        avg_loss = 0.0
        profit_factor = 0.0
        
        if n_trades > 0:
            pnls = [t.get('pnl', 0) for t in self.trade_history if 'pnl' in t]
            winning_trades = [p for p in pnls if p > 0]
            losing_trades = [p for p in pnls if p < 0]
            
            win_rate = len(winning_trades) / n_trades if n_trades > 0 else 0.0
            avg_win = np.mean(winning_trades) if winning_trades else 0.0
            avg_loss = np.mean(losing_trades) if losing_trades else 0.0
            
            total_win = np.sum(winning_trades)
            total_loss = abs(np.sum(losing_trades))
            profit_factor = total_win / (total_loss + 1e-9) if total_loss > 0 else np.inf
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_fees': self.state.total_fees if hasattr(self.state, 'total_fees') else 0.0
        }
