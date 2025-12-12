"""
Machine Learning Backtester

This module provides a backtesting framework specifically designed for machine learning-based
trading strategies, with support for walk-forward optimization, cross-validation, and
performance analysis.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import warnings

from ..ml.ensemble import EnsembleModel
from ..ml.regime_detection import MarketRegimeDetector
from .engine import BacktestEngine
from ..interfaces.order_book import Order, OrderType, OrderSide

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=FutureWarning)

@dataclass
class BacktestResult:
    """Container for backtest results and metrics."""
    returns: pd.Series
    trades: pd.DataFrame
    positions: pd.DataFrame
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    feature_importance: Optional[pd.DataFrame] = None
    regime_history: Optional[pd.DataFrame] = None
    model_weights: Optional[pd.DataFrame] = None
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        return pd.DataFrame({
            'return': self.returns,
            'cumulative_return': (1 + self.returns).cumprod() - 1
        })
    
    def plot_equity_curve(self, save_path: Optional[str] = None) -> None:
        """Plot the equity curve."""
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        (1 + self.returns).cumprod().plot()
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Cumulative Return')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()

class MLBacktester:
    """
    Backtesting engine for machine learning-based trading strategies.
    
    This class extends the basic backtesting functionality with ML-specific features:
    - Walk-forward optimization
    - Cross-validation
    - Feature importance analysis
    - Model performance tracking
    - Regime-aware backtesting
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        model: Any,
        feature_columns: List[str],
        target_column: str,
        initial_capital: float = 1_000_000.0,
        commission: float = 0.001,
        slippage: float = 0.0005,
        position_size: float = 0.1,
        max_position_size: float = 0.5,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        regime_aware: bool = True,
        regime_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize the ML backtester.
        
        Args:
            data: DataFrame containing the time series data
            model: ML model or ensemble to use for predictions
            feature_columns: List of column names to use as features
            target_column: Name of the target column
            initial_capital: Initial capital for the backtest
            commission: Commission per trade (as a fraction of trade value)
            slippage: Slippage per trade (as a fraction of trade value)
            position_size: Fraction of capital to risk per trade
            max_position_size: Maximum position size as a fraction of capital
            stop_loss: Stop loss level (as a fraction of entry price)
            take_profit: Take profit level (as a fraction of entry price)
            regime_aware: Whether to use regime-aware backtesting
            regime_params: Parameters for regime detection
            **kwargs: Additional keyword arguments
        """
        self.data = data.copy()
        self.model = model
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.regime_aware = regime_aware
        self.regime_params = regime_params or {}
        self.kwargs = kwargs
        
        # Initialize state
        self.current_idx = 0
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity = [initial_capital]
        self.returns = []
        self.signals = []
        self.regime_history = []
        self.model_weights = []
        
        # Initialize regime detector if needed
        self.regime_detector = None
        if self.regime_aware:
            self.regime_detector = MarketRegimeDetector(
                n_regimes=self.regime_params.get('n_regimes', 3),
                lookback_window=self.regime_params.get('lookback_window', 252),
                method=self.regime_params.get('method', 'hmm')
            )
    
    def prepare_features(self, idx: int) -> Tuple[np.ndarray, float]:
        """
        Prepare features and target for prediction at the given index.
        
        Args:
            idx: Current index in the data
            
        Returns:
            Tuple of (features, target)
        """
        if idx < 1:
            return np.array([]), 0.0
        
        # Get features
        features = self.data[self.feature_columns].iloc[idx-1].values.reshape(1, -1)
        
        # Get target (next period's return)
        target = self.data[self.target_column].iloc[idx]
        
        return features, target
    
    def generate_signals(self, idx: int, features: np.ndarray) -> float:
        """
        Generate trading signals using the ML model.
        
        Args:
            idx: Current index in the data
            features: Input features for prediction
            
        Returns:
            Signal value (-1 to 1)
        """
        if features.size == 0:
            return 0.0
        
        # Get prediction from the model
        try:
            if hasattr(self.model, 'predict_proba'):
                # For classification models
                proba = self.model.predict_proba(features)[0]
                # Use the probability of the positive class as the signal
                signal = 2 * proba[1] - 1  # Scale to [-1, 1]
            else:
                # For regression models
                signal = float(self.model.predict(features)[0])
                # Clip to [-1, 1] range
                signal = max(-1.0, min(1.0, signal))
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return 0.0
    
    def execute_trades(self, idx: int, signal: float) -> None:
        """
        Execute trades based on the signal.
        
        Args:
            idx: Current index in the data
            signal: Trading signal (-1 to 1)
        """
        if idx >= len(self.data):
            return
        
        current_price = self.data['close'].iloc[idx]
        current_time = self.data.index[idx]
        
        # Calculate position size based on signal and available capital
        position_value = self.cash * self.position_size * abs(signal)
        position_size = position_value / current_price
        
        # Apply position limits
        max_position = (self.cash * self.max_position_size) / current_price
        position_size = max(-max_position, min(max_position, position_size))
        
        # Round to whole shares/contracts
        position_size = int(position_size)
        
        if position_size == 0:
            return
        
        # Determine order type and price
        order_type = OrderType.MARKET
        price = current_price * (1 + np.sign(position_size) * self.slippage)
        
        # Create and submit order
        order = Order(
            symbol='asset',  # Assuming single asset for simplicity
            quantity=abs(position_size),
            order_type=order_type,
            side=OrderSide.BUY if position_size > 0 else OrderSide.SELL,
            price=price
        )
        
        # Calculate commission
        commission = abs(position_size * price * self.commission)
        
        # Update positions and cash
        if 'asset' not in self.positions:
            self.positions['asset'] = 0.0
        
        self.positions['asset'] += position_size
        self.cash -= position_size * price + commission
        
        # Record the trade
        self.trades.append({
            'timestamp': current_time,
            'price': price,
            'quantity': position_size,
            'value': abs(position_size * price),
            'commission': commission,
            'side': 'buy' if position_size > 0 else 'sell',
            'signal': signal
        })
        
        logger.debug(f"Executed trade: {position_size} shares at {price:.2f} (signal: {signal:.2f})")
    
    def update_equity(self, idx: int) -> None:
        """
        Update the equity curve.
        
        Args:
            idx: Current index in the data
        """
        if idx >= len(self.data):
            return
        
        # Calculate current position value
        position_value = 0.0
        for symbol, quantity in self.positions.items():
            if quantity != 0:
                position_value += quantity * self.data['close'].iloc[idx]
        
        # Update equity
        self.equity.append(self.cash + position_value)
        
        # Calculate return
        if len(self.equity) > 1:
            ret = (self.equity[-1] / self.equity[-2]) - 1
            self.returns.append(ret)
    
    def run_backtest(self) -> BacktestResult:
        """
        Run the backtest.
        
        Returns:
            BacktestResult containing the backtest results
        """
        logger.info("Starting backtest...")
        
        # Initialize progress bar
        pbar = tqdm(total=len(self.data), desc="Backtesting")
        
        # Main backtest loop
        for idx in range(1, len(self.data)):
            try:
                # Update progress bar
                pbar.update(1)
                
                # Update regime detection
                if self.regime_aware and self.regime_detector is not None:
                    try:
                        # Update regime detector with latest data
                        if idx > self.regime_detector.lookback_window:
                            window_data = self.data.iloc[idx - self.regime_detector.lookback_window:idx]
                            self.regime_detector.fit(window_data)
                            
                            # Get current regime
                            current_regime = self.regime_detector.current_regime
                            regime_probs = self.regime_detector.regime_probs
                            
                            # Record regime
                            self.regime_history.append({
                                'timestamp': self.data.index[idx],
                                'regime': current_regime,
                                'probabilities': regime_probs.tolist() if hasattr(regime_probs, 'tolist') else regime_probs
                            })
                    except Exception as e:
                        logger.error(f"Error updating regime detector: {str(e)}")
                
                # Prepare features and target
                features, target = self.prepare_features(idx)
                
                # Skip if we don't have enough data
                if features.size == 0:
                    continue
                
                # Update model if online learning is supported
                if hasattr(self.model, 'partial_fit'):
                    try:
                        # Use previous data point as training example
                        X_train = self.data[self.feature_columns].iloc[idx-1].values.reshape(1, -1)
                        y_train = np.array([self.data[self.target_column].iloc[idx]])
                        
                        # Update model
                        self.model.partial_fit(X_train, y_train)
                    except Exception as e:
                        logger.error(f"Error updating model: {str(e)}")
                
                # Generate trading signal
                signal = self.generate_signals(idx, features)
                
                # Record signal
                self.signals.append({
                    'timestamp': self.data.index[idx],
                    'signal': signal,
                    'price': self.data['close'].iloc[idx]
                })
                
                # Execute trades
                self.execute_trades(idx, signal)
                
                # Update equity
                self.update_equity(idx)
                
                # Record model weights if available
                if hasattr(self.model, 'get_model_weights'):
                    try:
                        weights = self.model.get_model_weights()
                        self.model_weights.append({
                            'timestamp': self.data.index[idx],
                            **weights
                        })
                    except Exception as e:
                        logger.error(f"Error getting model weights: {str(e)}")
                
            except Exception as e:
                logger.error(f"Error at index {idx}: {str(e)}")
                continue
        
        # Close progress bar
        pbar.close()
        
        # Calculate performance metrics
        returns = pd.Series(self.returns, index=self.data.index[2:len(self.returns)+2])
        trades = pd.DataFrame(self.trades) if self.trades else pd.DataFrame()
        positions = pd.Series(
            [p.get('asset', 0) for p in self.positions],
            index=self.data.index[1:len(self.positions)+1]
        )
        
        # Calculate performance metrics
        metrics = self.calculate_metrics(returns, trades)
        
        # Prepare additional results
        regime_history = pd.DataFrame(self.regime_history) if self.regime_history else None
        model_weights = pd.DataFrame(self.model_weights) if self.model_weights else None
        
        return BacktestResult(
            returns=returns,
            trades=trades,
            positions=positions,
            metrics=metrics,
            parameters={
                'initial_capital': self.initial_capital,
                'commission': self.commission,
                'slippage': self.slippage,
                'position_size': self.position_size,
                'max_position_size': self.max_position_size,
                'stop_loss': self.stop_loss,
                'take_profit': self.take_profit,
                'regime_aware': self.regime_aware
            },
            regime_history=regime_history,
            model_weights=model_weights
        )
    
    def calculate_metrics(
        self,
        returns: pd.Series,
        trades: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Args:
            returns: Series of returns
            trades: DataFrame of trades
            
        Returns:
            Dictionary of performance metrics
        """
        if len(returns) < 2:
            return {}
        
        # Basic metrics
        total_return = (1 + returns).prod() - 1
        annual_return = (1 + returns).prod() ** (252 / len(returns)) - 1
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = np.sqrt(252) * returns.mean() / (returns.std() + 1e-10)
        
        # Maximum drawdown
        cum_returns = (1 + returns).cumprod()
        rolling_max = cum_returns.cummax()
        drawdowns = (cum_returns - rolling_max) / (rolling_max + 1e-10)
        max_drawdown = drawdowns.min()
        
        # Trade metrics
        if not trades.empty:
            # Calculate trade returns
            trades['return'] = 0.0
            for i in range(1, len(trades)):
                if trades['side'].iloc[i-1] == 'buy':
                    entry_price = trades['price'].iloc[i-1]
                    exit_price = trades['price'].iloc[i]
                    trades.loc[trades.index[i], 'return'] = (exit_price / entry_price) - 1
            
            # Filter out non-trade returns (signals without execution)
            trade_returns = trades[trades['side'] == 'sell']['return']
            
            if len(trade_returns) > 0:
                win_rate = (trade_returns > 0).mean()
                avg_win = trade_returns[trade_returns > 0].mean()
                avg_loss = trade_returns[trade_returns <= 0].mean()
                profit_factor = -avg_win * (trade_returns > 0).sum() / \
                              (avg_loss * (trade_returns <= 0).sum() + 1e-10)
            else:
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0
                profit_factor = 0.0
        else:
            win_rate = 0.0
            avg_win = 0.0
            avg_loss = 0.0
            profit_factor = 0.0
        
        # Sortino ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = np.sqrt(252) * returns.mean() / (downside_std + 1e-10)
        
        # Calmar ratio (return over max drawdown)
        calmar_ratio = annual_return / (abs(max_drawdown) + 1e-10)
        
        # Omega ratio
        threshold = 0.0
        excess = returns - threshold
        omega_ratio = excess[excess > 0].sum() / (abs(excess[excess < 0].sum()) + 1e-10)
        
        return {
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'annual_volatility': float(annual_volatility),
            'sharpe_ratio': float(sharpe_ratio),
            'sortino_ratio': float(sortino_ratio),
            'calmar_ratio': float(calmar_ratio),
            'omega_ratio': float(omega_ratio),
            'max_drawdown': float(max_drawdown),
            'win_rate': float(win_rate),
            'avg_win': float(avg_win),
            'avg_loss': float(avg_loss),
            'profit_factor': float(profit_factor),
            'num_trades': len(trades) // 2 if not trades.empty else 0,
            'final_equity': float(self.equity[-1]) if self.equity else float(self.initial_capital)
        }
    
    def run_walk_forward(
        self,
        train_size: int = 252,
        test_size: int = 63,
        step_size: int = 21,
        min_train_size: int = 100,
        refit: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run walk-forward backtesting.
        
        Args:
            train_size: Number of periods for training
            test_size: Number of periods for testing
            step_size: Number of periods to move forward after each iteration
            min_train_size: Minimum number of periods required for training
            refit: Whether to refit the model for each training window
            **kwargs: Additional arguments to pass to the backtest
            
        Returns:
            Dictionary containing walk-forward results
        """
        results = []
        
        # Initialize progress bar
        total_windows = (len(self.data) - train_size - test_size) // step_size + 1
        pbar = tqdm(total=total_windows, desc="Walk-forward backtesting")
        
        # Walk-forward loop
        for i in range(0, len(self.data) - train_size - test_size + 1, step_size):
            # Update progress bar
            pbar.update(1)
            
            # Get train and test data
            train_data = self.data.iloc[i:i+train_size].copy()
            test_data = self.data.iloc[i+train_size:i+train_size+test_size].copy()
            
            # Skip if we don't have enough data
            if len(train_data) < min_train_size or len(test_data) < test_size // 2:
                continue
            
            # Reset model for this fold
            if hasattr(self.model, 'reset'):
                self.model.reset()
            
            # Fit model on training data
            if refit:
                try:
                    X_train = train_data[self.feature_columns].values
                    y_train = train_data[self.target_column].values
                    
                    if hasattr(self.model, 'fit'):
                        self.model.fit(X_train, y_train)
                except Exception as e:
                    logger.error(f"Error fitting model for window {i}: {str(e)}")
                    continue
            
            # Run backtest on test data
            try:
                # Create a new backtester for this fold
                backtester = MLBacktester(
                    data=test_data,
                    model=self.model,
                    feature_columns=self.feature_columns,
                    target_column=self.target_column,
                    initial_capital=self.initial_capital,
                    commission=self.commission,
                    slippage=self.slippage,
                    position_size=self.position_size,
                    max_position_size=self.max_position_size,
                    stop_loss=self.stop_loss,
                    take_profit=self.take_profit,
                    regime_aware=self.regime_aware,
                    regime_params=self.regime_params,
                    **self.kwargs
                )
                
                # Run backtest
                result = backtester.run_backtest()
                
                # Store results
                results.append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'metrics': result.metrics,
                    'returns': result.returns,
                    'trades': result.trades
                })
                
            except Exception as e:
                logger.error(f"Error running backtest for window {i}: {str(e)}")
                continue
        
        # Close progress bar
        pbar.close()
        
        # Aggregate results
        if not results:
            return {}
        
        # Calculate average metrics across all folds
        avg_metrics = {}
        for metric in results[0]['metrics']:
            values = [r['metrics'][metric] for r in results if metric in r['metrics']]
            if values:
                avg_metrics[f'avg_{metric}'] = np.mean(values)
                avg_metrics[f'std_{metric}'] = np.std(values)
        
        # Combine all returns and trades
        all_returns = pd.concat([r['returns'] for r in results if not r['returns'].empty])
        all_trades = pd.concat([r['trades'] for r in results if not r['trades'].empty])
        
        # Calculate overall metrics
        overall_metrics = self.calculate_metrics(all_returns, all_trades)
        
        return {
            'window_results': results,
            'average_metrics': avg_metrics,
            'overall_metrics': overall_metrics,
            'all_returns': all_returns,
            'all_trades': all_trades
        }
