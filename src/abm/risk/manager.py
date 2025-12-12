""
Risk Management Module for the Agent-Based Modeling system.

This module implements risk management functionality including position limits,
value-at-risk (VaR) calculations, and circuit breakers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import IsolationForest
from scipy.stats import norm

from ..data import MarketDataConnector, DataPipeline
from ..data.utils import calculate_value_at_risk, calculate_expected_shortfall
from ..data.config import get_config

logger = logging.getLogger(__name__)

class RiskLimitType(Enum):
    """Types of risk limits."""
    POSITION_SIZE = auto()
    DAILY_LOSS = auto()
    VAR = auto()
    DRAWDOWN = auto()
    LEVERAGE = auto()
    CONCENTRATION = auto()
    
class RiskViolationSeverity(Enum):
    """Severity levels for risk violations."""
    INFO = auto()
    WARNING = auto()
    CRITICAL = auto()
    
@dataclass
class RiskLimit:
    """Risk limit definition."""
    limit_type: RiskLimitType
    value: float
    symbol: Optional[str] = None  # None for portfolio-wide limits
    severity: RiskViolationSeverity = RiskViolationSeverity.WARNING
    action: Optional[str] = None  # Optional action to take when limit is breached
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'limit_type': self.limit_type.name,
            'value': self.value,
            'symbol': self.symbol,
            'severity': self.severity.name,
            'action': self.action
        }

@dataclass
class RiskViolation:
    """Risk limit violation."""
    limit: RiskLimit
    current_value: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(pytz.utc))
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'limit': self.limit.to_dict(),
            'current_value': self.current_value,
            'timestamp': self.timestamp.isoformat(),
            'message': self.message,
            'severity': self.limit.severity.name
        }

class RiskManager:
    """
    Risk management system for monitoring and enforcing risk limits.
    """
    
    def __init__(self, 
                 portfolio: Optional[Dict[str, float]] = None,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the risk manager.
        
        Args:
            portfolio: Initial portfolio positions {symbol: quantity}
            config: Configuration dictionary
        """
        self.portfolio = portfolio or {}
        self.config = config or get_config().get('risk', {})
        self.limits: List[RiskLimit] = []
        self.violations: List[RiskViolation] = []
        self.historical_positions: List[Dict[str, Any]] = []
        self.anomaly_detector = None
        self._initialize_anomaly_detector()
        self._setup_default_limits()
        
        # Initialize metrics
        self.metrics = {
            'total_checks': 0,
            'total_violations': 0,
            'critical_violations': 0,
            'warning_violations': 0,
            'info_violations': 0,
            'last_check': None,
            'start_time': datetime.now(pytz.utc)
        }
    
    def _initialize_anomaly_detector(self) -> None:
        """Initialize the anomaly detection model."""
        contamination = float(self.config.get('anomaly_contamination', 0.05))
        self.anomaly_detector = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=42
        )
    
    def _setup_default_limits(self) -> None:
        """Set up default risk limits from config."""
        # Position size limits
        max_position_size = self.config.get('max_position_size', 0.1)  # 10% of portfolio
        self.add_limit(
            RiskLimitType.POSITION_SIZE, 
            max_position_size,
            action='reject'
        )
        
        # Daily loss limit (5% of portfolio)
        self.add_limit(
            RiskLimitType.DAILY_LOSS,
            0.05,
            action='reduce_positions'
        )
        
        # Portfolio VaR limit (1% daily, 95% confidence)
        self.add_limit(
            RiskLimitType.VAR,
            0.01,
            action='reduce_risk'
        )
        
        # Maximum drawdown limit (10% from peak)
        self.add_limit(
            RiskLimitType.DRAWDOWN,
            0.10,
            action='stop_trading'
        )
        
        # Maximum leverage (2x)
        self.add_limit(
            RiskLimitType.LEVERAGE,
            2.0,
            action='reduce_leverage'
        )
    
    def add_limit(self, 
                 limit_type: RiskLimitType, 
                 value: float, 
                 symbol: Optional[str] = None,
                 severity: RiskViolationSeverity = RiskViolationSeverity.WARNING,
                 action: Optional[str] = None) -> None:
        """
        Add a risk limit.
        
        Args:
            limit_type: Type of risk limit
            value: Limit value
            symbol: Symbol the limit applies to (None for portfolio-wide)
            severity: Severity of violation
            action: Action to take when limit is breached
        """
        limit = RiskLimit(
            limit_type=limit_type,
            value=value,
            symbol=symbol,
            severity=severity,
            action=action
        )
        self.limits.append(limit)
    
    def update_portfolio(self, portfolio: Dict[str, float]) -> None:
        """
        Update the current portfolio positions.
        
        Args:
            portfolio: Dictionary of {symbol: quantity}
        """
        self.portfolio = portfolio.copy()
        
        # Record position history
        self.historical_positions.append({
            'timestamp': datetime.now(pytz.utc),
            'positions': portfolio.copy(),
            'portfolio_value': self._calculate_portfolio_value()
        })
        
        # Keep only recent history (last 30 days)
        cutoff = datetime.now(pytz.utc) - timedelta(days=30)
        self.historical_positions = [
            p for p in self.historical_positions 
            if p['timestamp'] >= cutoff
        ]
    
    def _calculate_portfolio_value(self) -> float:
        """
        Calculate the current portfolio value.
        
        Returns:
            Total portfolio value in the base currency
        """
        # In a real implementation, this would use market prices
        # For simplicity, we'll just sum the absolute positions
        return sum(abs(qty) for qty in self.portfolio.values())
    
    def check_risk_limits(self, 
                         market_data: Optional[Dict[str, Any]] = None) -> List[RiskViolation]:
        """
        Check all risk limits.
        
        Args:
            market_data: Current market data (optional)
            
        Returns:
            List of risk violations (empty if no violations)
        """
        self.metrics['total_checks'] += 1
        self.metrics['last_check'] = datetime.now(pytz.utc)
        
        violations = []
        
        # Check each limit
        for limit in self.limits:
            violation = self._check_limit(limit, market_data)
            if violation:
                violations.append(violation)
                
                # Update metrics
                self.metrics['total_violations'] += 1
                if limit.severity == RiskViolationSeverity.CRITICAL:
                    self.metrics['critical_violations'] += 1
                elif limit.severity == RiskViolationSeverity.WARNING:
                    self.metrics['warning_violations'] += 1
                else:
                    self.metrics['info_violations'] += 1
                
                # Take action if specified
                if limit.action:
                    self._take_action(limit.action, violation)
        
        return violations
    
    def _check_limit(self, 
                    limit: RiskLimit, 
                    market_data: Optional[Dict[str, Any]] = None) -> Optional[RiskViolation]:
        """
        Check a single risk limit.
        
        Args:
            limit: Risk limit to check
            market_data: Current market data (optional)
            
        Returns:
            RiskViolation if limit is breached, None otherwise
        """
        if limit.limit_type == RiskLimitType.POSITION_SIZE:
            return self._check_position_size_limit(limit)
        elif limit.limit_type == RiskLimitType.DAILY_LOSS:
            return self._check_daily_loss_limit(limit)
        elif limit.limit_type == RiskLimitType.VAR:
            return self._check_var_limit(limit, market_data)
        elif limit.limit_type == RiskLimitType.DRAWDOWN:
            return self._check_drawdown_limit(limit)
        elif limit.limit_type == RiskLimitType.LEVERAGE:
            return self._check_leverage_limit(limit)
        elif limit.limit_type == RiskLimitType.CONCENTRATION:
            return self._check_concentration_limit(limit)
        
        return None
    
    def _check_position_size_limit(self, limit: RiskLimit) -> Optional[RiskViolation]:
        """Check position size limit."""
        portfolio_value = self._calculate_portfolio_value()
        
        if limit.symbol:
            # Check specific symbol
            if limit.symbol in self.portfolio:
                position_value = abs(self.portfolio[limit.symbol])
                position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
                
                if position_pct > limit.value:
                    return RiskViolation(
                        limit=limit,
                        current_value=position_pct,
                        message=f"Position size {position_pct:.2%} exceeds limit of {limit.value:.2%} for {limit.symbol}"
                    )
        else:
            # Check all positions
            for symbol, qty in self.portfolio.items():
                position_value = abs(qty)
                position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
                
                if position_pct > limit.value:
                    return RiskViolation(
                        limit=limit,
                        current_value=position_pct,
                        message=f"Position size {position_pct:.2%} exceeds limit of {limit.value:.2%} for {symbol}"
                    )
        
        return None
    
    def _check_daily_loss_limit(self, limit: RiskLimit) -> Optional[RiskViolation]:
        """Check daily loss limit."""
        if len(self.historical_positions) < 2:
            return None
            
        # Get today's start time
        today = datetime.now(pytz.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        
        # Get start of day portfolio value
        start_value = None
        for pos in sorted(self.historical_positions, key=lambda x: x['timestamp']):
            if pos['timestamp'] >= today:
                start_value = pos['portfolio_value']
                break
                
        if start_value is None or start_value <= 0:
            return None
            
        # Calculate daily P&L
        current_value = self._calculate_portfolio_value()
        daily_pnl = current_value - start_value
        daily_return = daily_pnl / start_value
        
        # Check against limit
        if daily_return < -limit.value:
            return RiskViolation(
                limit=limit,
                current_value=daily_return,
                message=f"Daily loss {daily_return:.2%} exceeds limit of {-limit.value:.2%}"
            )
            
        return None
    
    def _check_var_limit(self, 
                        limit: RiskLimit, 
                        market_data: Optional[Dict[str, Any]] = None) -> Optional[RiskViolation]:
        """Check Value at Risk (VaR) limit."""
        # In a real implementation, this would use historical or simulated returns
        # For simplicity, we'll use a simplified approach
        
        if len(self.historical_positions) < 10:  # Need some history
            return None
            
        # Calculate portfolio returns
        portfolio_values = [p['portfolio_value'] for p in self.historical_positions]
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        if len(returns) < 5:  # Need at least 5 returns
            return None
            
        # Calculate 1-day 95% VaR
        var = calculate_value_at_risk(returns, confidence_level=0.95)
        
        # Check against limit
        if var < -limit.value:
            return RiskViolation(
                limit=limit,
                current_value=var,
                message=f"1-day 95% VaR {var:.2%} exceeds limit of {-limit.value:.2%}"
            )
            
        return None
    
    def _check_drawdown_limit(self, limit: RiskLimit) -> Optional[RiskViolation]:
        """Check maximum drawdown limit."""
        if len(self.historical_positions) < 2:
            return None
            
        # Calculate drawdown
        portfolio_values = [p['portfolio_value'] for p in self.historical_positions]
        peak = max(portfolio_values)
        current = portfolio_values[-1]
        drawdown = (peak - current) / peak if peak > 0 else 0
        
        # Check against limit
        if drawdown > limit.value:
            return RiskViolation(
                limit=limit,
                current_value=drawdown,
                message=f"Drawdown {drawdown:.2%} exceeds limit of {limit.value:.2%}"
            )
            
        return None
    
    def _check_leverage_limit(self, limit: RiskLimit) -> Optional[RiskViolation]:
        """Check leverage limit."""
        # In a real implementation, this would calculate actual leverage
        # For simplicity, we'll use a placeholder
        leverage = 1.0  # Replace with actual calculation
        
        if leverage > limit.value:
            return RiskViolation(
                limit=limit,
                current_value=leverage,
                message=f"Leverage {leverage:.2f}x exceeds limit of {limit.value:.2f}x"
            )
            
        return None
    
    def _check_concentration_limit(self, limit: RiskLimit) -> Optional[RiskViolation]:
        """Check concentration limit."""
        portfolio_value = self._calculate_portfolio_value()
        if portfolio_value <= 0:
            return None
            
        # Calculate concentration of each position
        for symbol, qty in self.portfolio.items():
            position_value = abs(qty)
            position_pct = position_value / portfolio_value
            
            if position_pct > limit.value:
                return RiskViolation(
                    limit=limit,
                    current_value=position_pct,
                    message=f"Concentration {position_pct:.2%} exceeds limit of {limit.value:.2%} for {symbol}"
                )
                
        return None
    
    def detect_anomalies(self, 
                        market_data: Dict[str, Any],
                        features: List[str] = None) -> Dict[str, Any]:
        """
        Detect anomalies in market data using isolation forest.
        
        Args:
            market_data: Dictionary of market data
            features: List of feature names to use for anomaly detection
            
        Returns:
            Dictionary with anomaly scores and flags
        """
        if not features:
            features = ['returns', 'volume', 'volatility']
            
        # Prepare feature matrix
        X = []
        valid_features = []
        
        for feature in features:
            if feature in market_data:
                X.append(market_data[feature])
                valid_features.append(feature)
                
        if not X:
            return {'anomaly_score': 0.0, 'is_anomaly': False, 'features': []}
            
        X = np.column_stack(X)
        
        # Fit the model if not already fitted
        if not hasattr(self.anomaly_detector, 'estimators_'):
            self.anomaly_detector.fit(X)
            
        # Predict anomalies
        anomaly_scores = -self.anomaly_detector.score_samples(X)  # Higher is more anomalous
        is_anomaly = anomaly_scores > np.percentile(anomaly_scores, 95)  # Top 5% are anomalies
        
        return {
            'anomaly_score': float(np.mean(anomaly_scores)),
            'is_anomaly': bool(np.any(is_anomaly)),
            'features': valid_features,
            'scores': {f: float(s) for f, s in zip(valid_features, anomaly_scores)}
        }
    
    def _take_action(self, action: str, violation: RiskViolation) -> None:
        """
        Take action in response to a risk limit violation.
        
        Args:
            action: Action to take
            violation: Risk violation that triggered the action
        """
        logger.warning(f"Taking action '{action}' for violation: {violation.message}")
        
        if action == 'reject':
            # Reject the order that caused the violation
            # This would be handled by the order management system
            pass
            
        elif action == 'reduce_positions':
            # Reduce positions to bring within limits
            self._reduce_positions(violation)
            
        elif action == 'reduce_risk':
            # Reduce overall risk exposure
            self._reduce_risk()
            
        elif action == 'stop_trading':
            # Halt all trading
            self._stop_trading()
            
        elif action == 'reduce_leverage':
            # Reduce leverage
            self._reduce_leverage()
            
        elif action == 'alert':
            # Just log a warning
            logger.warning(f"Risk alert: {violation.message}")
    
    def _reduce_positions(self, violation: RiskViolation) -> None:
        """Reduce positions to bring within limits."""
        # In a real implementation, this would determine which positions to reduce
        # and submit orders to the execution engine
        logger.info(f"Reducing positions due to: {violation.message}")
        
    def _reduce_risk(self) -> None:
        """Reduce overall risk exposure."""
        logger.info("Reducing overall risk exposure")
        
    def _stop_trading(self) -> None:
        """Halt all trading."""
        logger.critical("STOP TRADING: Risk limits breached")
        
    def _reduce_leverage(self) -> None:
        """Reduce leverage."""
        logger.info("Reducing leverage")
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get risk metrics.
        
        Returns:
            Dictionary of risk metrics
        """
        # Calculate additional metrics
        elapsed_hours = (datetime.now(pytz.utc) - self.metrics['start_time']).total_seconds() / 3600
        
        metrics = self.metrics.copy()
        metrics.update({
            'elapsed_hours': elapsed_hours,
            'checks_per_hour': metrics['total_checks'] / max(1, elapsed_hours),
            'violation_rate': metrics['total_violations'] / max(1, metrics['total_checks']),
            'last_updated': datetime.now(pytz.utc).isoformat()
        })
        
        return metrics


# Example usage
if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create a risk manager with default limits
    risk_manager = RiskManager()
    
    # Update portfolio positions
    portfolio = {
        'BTC/USDT': 1.5,
        'ETH/USDT': 10.0,
        'SOL/USDT': 100.0
    }
    risk_manager.update_portfolio(portfolio)
    
    # Check risk limits
    violations = risk_manager.check_risk_limits()
    
    if violations:
        print("Risk limit violations detected:")
        for violation in violations:
            print(f"- {violation.message} (severity: {violation.limit.severity.name})")
    else:
        print("No risk limit violations detected")
    
    # Get metrics
    metrics = risk_manager.get_metrics()
    print("\nRisk metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")
