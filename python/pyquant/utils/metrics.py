import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, auc
)
import torch
from scipy import stats
import json
import os
from pathlib import Path

# Set plotting style
plt.style.use('seaborn')
sns.set_palette('viridis')

def calculate_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate simple returns from price series."""
    return np.diff(prices) / prices[:-1]

def calculate_log_returns(prices: np.ndarray) -> np.ndarray:
    """Calculate log returns from price series."""
    return np.diff(np.log(prices))

def calculate_sharpe_ratio(returns: np.ndarray, risk_free_rate: float = 0.0, 
                          periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sharpe ratio of a returns stream.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        float: Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # Handle case where all returns are the same
    if np.std(excess_returns) == 0:
        return 0.0
    
    sharpe = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(excess_returns)
    return float(sharpe)

def calculate_sortino_ratio(returns: np.ndarray, risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate the annualized Sortino ratio of a returns stream.
    
    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate (default: 0.0)
        periods_per_year: Number of periods per year (default: 252 trading days)
        
    Returns:
        float: Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = np.minimum(0, excess_returns)
    
    # Handle case where all returns are non-negative
    if np.std(downside_returns) == 0:
        return 0.0
    
    sortino = np.sqrt(periods_per_year) * np.mean(excess_returns) / np.std(downside_returns)
    return float(sortino)

def max_drawdown(returns: np.ndarray) -> float:
    """
    Calculate the maximum drawdown of a return series.
    
    Args:
        returns: Array of returns
        
    Returns:
        float: Maximum drawdown as a negative number (e.g., -0.15 for 15% drawdown)
    """
    if len(returns) == 0:
        return 0.0
    
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    
    if len(drawdowns) == 0:
        return 0.0
    
    return float(np.min(drawdowns))  # Most negative value is the max drawdown

def calmar_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """
    Calculate the Calmar ratio (annualized return over max drawdown).
    
    Args:
        returns: Array of returns
        periods_per_year: Number of periods per year
        
    Returns:
        float: Calmar ratio
    """
    if len(returns) == 0:
        return 0.0
    
    annual_return = np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1
    mdd = abs(max_drawdown(returns))
    
    if mdd == 0:
        return 0.0
    
    return float(annual_return / mdd)

def profit_factor(returns: np.ndarray) -> float:
    """
    Calculate the profit factor (gross profits / gross losses).
    
    Args:
        returns: Array of returns
        
    Returns:
        float: Profit factor (values > 1 indicate profitable strategy)
    """
    if len(returns) == 0:
        return 0.0
    
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    
    if losses == 0:
        return float('inf')
    
    return float(gains / losses)

def win_rate(returns: np.ndarray) -> float:
    """
    Calculate the win rate (percentage of positive returns).
    
    Args:
        returns: Array of returns
        
    Returns:
        float: Win rate between 0 and 1
    """
    if len(returns) == 0:
        return 0.0
    
    return float(np.mean(returns > 0))

def average_win_loss_ratio(returns: np.ndarray) -> float:
    """
    Calculate the average win/loss ratio.
    
    Args:
        returns: Array of returns
        
    Returns:
        float: Ratio of average win to average loss
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    
    if avg_loss == 0:
        return float('inf')
    
    return float(avg_win / avg_loss)

def value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate the Value at Risk (VaR) at a given confidence level.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        float: Value at Risk (negative value indicating loss)
    """
    if len(returns) == 0:
        return 0.0
    
    return float(np.percentile(returns, (1 - confidence_level) * 100))

def conditional_value_at_risk(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate the Conditional Value at Risk (CVaR) at a given confidence level.
    
    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)
        
    Returns:
        float: Conditional Value at Risk (negative value indicating loss)
    """
    if len(returns) == 0:
        return 0.0
    
    var = value_at_risk(returns, confidence_level)
    cvar = returns[returns <= var].mean()
    
    return float(cvar if not np.isnan(cvar) else 0.0)

def information_ratio(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the Information Ratio.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        float: Information ratio
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    excess_returns = returns - benchmark_returns
    
    if np.std(excess_returns) == 0:
        return 0.0
    
    return float(np.mean(excess_returns) / np.std(excess_returns))

def beta(returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Calculate the beta of a strategy relative to a benchmark.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        
    Returns:
        float: Beta coefficient
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    cov = np.cov(returns, benchmark_returns)[0, 1]
    var = np.var(benchmark_returns)
    
    if var == 0:
        return 0.0
    
    return float(cov / var)

def alpha(returns: np.ndarray, benchmark_returns: np.ndarray, 
         risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the alpha of a strategy.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        float: Alpha (annualized excess return)
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    beta_val = beta(returns, benchmark_returns)
    avg_return = np.mean(returns) * periods_per_year
    avg_benchmark = np.mean(benchmark_returns) * periods_per_year
    
    return float(avg_return - (risk_free_rate + beta_val * (avg_benchmark - risk_free_rate)))

def treynor_ratio(returns: np.ndarray, benchmark_returns: np.ndarray,
                 risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
    """
    Calculate the Treynor ratio.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Array of benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        float: Treynor ratio
    """
    if len(returns) != len(benchmark_returns) or len(returns) < 2:
        return 0.0
    
    beta_val = beta(returns, benchmark_returns)
    
    if beta_val == 0:
        return 0.0
    
    excess_return = np.mean(returns) * periods_per_year - risk_free_rate
    return float(excess_return / beta_val)

def compute_all_metrics(returns: np.ndarray, 
                       benchmark_returns: Optional[np.ndarray] = None,
                       risk_free_rate: float = 0.0,
                       periods_per_year: int = 252) -> Dict[str, float]:
    """
    Compute all available performance metrics.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Optional array of benchmark returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        
    Returns:
        dict: Dictionary of metric names and values
    """
    metrics = {
        'total_return': float(np.prod(1 + returns) - 1),
        'annualized_return': float(np.prod(1 + returns) ** (periods_per_year / len(returns)) - 1),
        'annualized_volatility': float(np.std(returns) * np.sqrt(periods_per_year)),
        'sharpe_ratio': calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year),
        'sortino_ratio': calculate_sortino_ratio(returns, risk_free_rate, periods_per_year),
        'max_drawdown': max_drawdown(returns),
        'calmar_ratio': calmar_ratio(returns, periods_per_year),
        'profit_factor': profit_factor(returns),
        'win_rate': win_rate(returns),
        'avg_win_loss_ratio': average_win_loss_ratio(returns),
        'var_95': value_at_risk(returns, 0.95),
        'cvar_95': conditional_value_at_risk(returns, 0.95),
    }
    
    if benchmark_returns is not None and len(returns) == len(benchmark_returns):
        metrics.update({
            'alpha': alpha(returns, benchmark_returns, risk_free_rate, periods_per_year),
            'beta': beta(returns, benchmark_returns),
            'information_ratio': information_ratio(returns, benchmark_returns),
            'treynor_ratio': treynor_ratio(returns, benchmark_returns, risk_free_rate, periods_per_year),
            'r_squared': float(np.corrcoef(returns, benchmark_returns)[0, 1] ** 2),
        })
    
    return metrics

def plot_equity_curve(returns: np.ndarray, 
                     benchmark_returns: Optional[np.ndarray] = None,
                     title: str = 'Equity Curve',
                     save_path: Optional[str] = None) -> None:
    """
    Plot the equity curve of a strategy.
    
    Args:
        returns: Array of strategy returns
        benchmark_returns: Optional array of benchmark returns
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    plt.figure(figsize=(12, 6))
    
    # Calculate cumulative returns
    equity = np.cumprod(1 + returns) - 1
    plt.plot(equity, label='Strategy', linewidth=2)
    
    if benchmark_returns is not None and len(returns) == len(benchmark_returns):
        benchmark_equity = np.cumprod(1 + benchmark_returns) - 1
        plt.plot(benchmark_equity, label='Benchmark', linewidth=2, alpha=0.7)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Period', fontsize=12)
    plt.ylabel('Cumulative Return', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_drawdown(returns: np.ndarray, 
                 title: str = 'Drawdown',
                 save_path: Optional[str] = None) -> None:
    """
    Plot the drawdown of a strategy.
    
    Args:
        returns: Array of strategy returns
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    cum_returns = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns - peak) / peak
    
    plt.figure(figsize=(12, 4))
    plt.fill_between(range(len(drawdowns)), drawdowns * 100, 0, 
                    color='red', alpha=0.3)
    plt.title(title, fontsize=14)
    plt.xlabel('Period', fontsize=12)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

def plot_rolling_sharpe(returns: np.ndarray, 
                       window: int = 63,  # 3 months of daily data
                       risk_free_rate: float = 0.0,
                       periods_per_year: int = 252,
                       title: str = 'Rolling Sharpe Ratio',
                       save_path: Optional[str] = None) -> None:
    """
    Plot the rolling Sharpe ratio of a strategy.
    
    Args:
        returns: Array of strategy returns
        window: Rolling window size
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year
        title: Plot title
        save_path: Path to save the plot (optional)
    """
    if len(returns) < window:
        return
    
    rolling_sharpe = returns.rolling(window).apply(
        lambda x: calculate_sharpe_ratio(x, risk_free_rate, periods_per_year)
    )
    
    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe, label=f'Rolling {window}-period Sharpe Ratio', linewidth=2)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1)
    plt.title(title, fontsize=14)
    plt.xlabel('Period', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    plt.show()

class ModelEvaluator:
    """Class for evaluating trading models."""
    
    def __init__(self, model, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the model evaluator.
        
        Args:
            model: Trained PyTorch model
            device: Device to use for evaluation ('cuda' or 'cpu')
        """
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
    
    def evaluate(
        self,
        data_loader,
        criterion=None,
        return_predictions: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: DataLoader for the evaluation dataset
            criterion: Loss function (optional)
            return_predictions: Whether to return model predictions
            
        Returns:
            dict: Dictionary of evaluation metrics
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in data_loader:
                if len(batch) == 2:
                    inputs, targets = batch
                    policy_targets, value_targets = None, None
                elif len(batch) == 3:
                    inputs, policy_targets, value_targets = batch
                    policy_targets = policy_targets.to(self.device)
                    value_targets = value_targets.to(self.device)
                else:
                    raise ValueError("Batch must contain 2 or 3 elements")
                
                inputs = inputs.to(self.device)
                
                # Forward pass
                policy_preds, value_preds = self.model(inputs)
                
                # Compute loss if criterion is provided
                if criterion is not None and policy_targets is not None and value_targets is not None:
                    loss = criterion(policy_preds, value_preds, policy_targets, value_targets)
                    total_loss += loss.item() * inputs.size(0)
                
                # Store predictions and targets
                if return_predictions:
                    all_preds.append((policy_preds.cpu().numpy(), value_preds.cpu().numpy()))
                    if policy_targets is not None and value_targets is not None:
                        all_targets.append((policy_targets.cpu().numpy(), value_targets.cpu().numpy()))
        
        # Calculate metrics
        metrics = {}
        
        if criterion is not None and len(data_loader.dataset) > 0:
            metrics['loss'] = total_loss / len(data_loader.dataset)
        
        if return_predictions:
            return metrics, all_preds, all_targets
        
        return metrics
    
    def predict(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate predictions for a batch of inputs.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            tuple: (policy_predictions, value_predictions)
        """
        self.model.eval()
        
        with torch.no_grad():
            inputs = inputs.to(self.device)
            policy_preds, value_preds = self.model(inputs)
            
        return policy_preds.cpu(), value_preds.cpu()
    
    def save_metrics(self, metrics: Dict[str, float], filepath: str) -> None:
        """
        Save evaluation metrics to a JSON file.
        
        Args:
            metrics: Dictionary of metrics
            filepath: Path to save the metrics
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Convert numpy types to Python native types for JSON serialization
        metrics_serializable = {}
        for k, v in metrics.items():
            if isinstance(v, (np.integer, np.floating)):
                metrics_serializable[k] = float(v)
            elif isinstance(v, np.ndarray):
                metrics_serializable[k] = v.tolist()
            else:
                metrics_serializable[k] = v
        
        with open(filepath, 'w') as f:
            json.dump(metrics_serializable, f, indent=2)
    
    @staticmethod
    def load_metrics(filepath: str) -> Dict[str, float]:
        """
        Load evaluation metrics from a JSON file.
        
        Args:
            filepath: Path to the metrics file
            
        Returns:
            dict: Dictionary of metrics
        """
        with open(filepath, 'r') as f:
            metrics = json.load(f)
        
        return metrics
