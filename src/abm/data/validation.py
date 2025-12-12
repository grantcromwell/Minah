""
Data validation and quality assurance for market data.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import logging
from scipy import stats
import warnings

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

@dataclass
class DataQualityReport:
    """Container for data quality metrics and validation results."""
    summary: Dict[str, Any]
    issues: List[Dict[str, Any]]
    metrics: Dict[str, float]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the report to a dictionary."""
        return {
            'summary': self.summary,
            'issues': self.issues,
            'metrics': self.metrics
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert issues to a pandas DataFrame."''"
        if not self.issues:
            return pd.DataFrame()
        return pd.DataFrame(self.issues)

class MarketDataValidator:
    """
    Validates market data quality and detects anomalies.
    """
    
    def __init__(self, 
                 price_threshold: float = 0.05,  # 5% price change threshold
                 volume_threshold: float = 5.0,  # 5x volume change threshold
                 zscore_threshold: float = 3.0,  # Z-score for outlier detection
                 min_liquidity: float = 1e-6,    # Minimum liquidity threshold
                 **kwargs):
        """
        Initialize the market data validator.
        
        Args:
            price_threshold: Maximum allowed price change between consecutive ticks
            volume_threshold: Maximum allowed volume change between consecutive ticks
            zscore_threshold: Z-score threshold for statistical outlier detection
            min_liquidity: Minimum liquidity threshold for valid price data
        """
        self.price_threshold = price_threshold
        self.volume_threshold = volume_threshold
        self.zscore_threshold = zscore_threshold
        self.min_liquidity = min_liquidity
        
    def validate_ohlcv(self, 
                      df: pd.DataFrame,
                      price_col: str = 'close',
                      volume_col: str = 'volume',
                      timestamp_col: str = 'timestamp') -> DataQualityReport:
        """
        Validate OHLCV data quality.
        
        Args:
            df: DataFrame with OHLCV data
            price_col: Name of the price column
            volume_col: Name of the volume column
            timestamp_col: Name of the timestamp column
            
        Returns:
            DataQualityReport with validation results
        """
        if df.empty:
            return DataQualityReport(
                summary={'status': 'error', 'message': 'Empty DataFrame'},
                issues=[],
                metrics={}
            )
        
        # Initialize report
        issues = []
        metrics = {}
        
        # Basic statistics
        metrics['num_rows'] = len(df)
        metrics['date_range'] = {
            'start': df[timestamp_col].min(),
            'end': df[timestamp_col].max()
        }
        
        # Check for missing values
        missing = df.isnull().sum()
        missing_pct = (missing / len(df)) * 100
        
        for col, count in missing.items():
            if count > 0:
                issues.append({
                    'type': 'missing_values',
                    'column': col,
                    'count': int(count),
                    'pct': float(missing_pct[col]),
                    'severity': 'high' if missing_pct[col] > 1.0 else 'medium',
                    'suggestion': 'Impute or remove rows with missing values'
                })
        
        # Check for zero or negative prices/volumes
        for col in [price_col, 'open', 'high', 'low']:
            if col in df.columns:
                zero_or_neg = (df[col] <= 0).sum()
                if zero_or_neg > 0:
                    issues.append({
                        'type': 'invalid_values',
                        'column': col,
                        'count': int(zero_or_neg),
                        'pct': float((zero_or_neg / len(df)) * 100),
                        'severity': 'high',
                        'suggestion': 'Investigate and fix invalid price values'
                    })
        
        # Check for zero volumes
        if volume_col in df.columns:
            zero_volume = (df[volume_col] == 0).sum()
            if zero_volume > 0:
                issues.append({
                    'type': 'zero_volume',
                    'column': volume_col,
                    'count': int(zero_volume),
                    'pct': float((zero_volume / len(df)) * 100),
                    'severity': 'medium',
                    'suggestion': 'Check for data issues or illiquid periods'
                })
        
        # Check for price jumps
        if price_col in df.columns:
            price_changes = df[price_col].pct_change().abs()
            large_jumps = (price_changes > self.price_threshold).sum()
            
            if large_jumps > 0:
                issues.append({
                    'type': 'price_jumps',
                    'column': price_col,
                    'count': int(large_jumps),
                    'pct': float((large_jumps / (len(df) - 1)) * 100),
                    'severity': 'high' if (large_jumps / (len(df) - 1)) > 0.01 else 'medium',
                    'suggestion': 'Verify large price movements or implement filtering'
                })
            
            # Calculate price statistics
            returns = df[price_col].pct_change().dropna()
            metrics['price_stats'] = {
                'mean_return': float(returns.mean()),
                'volatility': float(returns.std()),
                'sharpe_ratio': float(np.sqrt(252) * returns.mean() / (returns.std() + 1e-6)),
                'max_drawdown': float(self._calculate_max_drawdown(df[price_col]))
            }
        
        # Check for volume anomalies
        if volume_col in df.columns:
            volume_changes = df[volume_col].pct_change().abs()
            volume_spikes = (volume_changes > self.volume_threshold).sum()
            
            if volume_spikes > 0:
                issues.append({
                    'type': 'volume_spikes',
                    'column': volume_col,
                    'count': int(volume_spikes),
                    'pct': float((volume_spikes / (len(df) - 1)) * 100),
                    'severity': 'medium',
                    'suggestion': 'Investigate unusual trading activity'
                })
            
            # Calculate volume statistics
            metrics['volume_stats'] = {
                'mean': float(df[volume_col].mean()),
                'std': float(df[volume_col].std()),
                'max': float(df[volume_col].max()),
                'min': float(df[volume_col].min())
            }
        
        # Check for duplicate timestamps
        if timestamp_col in df.columns:
            dup_timestamps = df.duplicated(subset=[timestamp_col]).sum()
            if dup_timestamps > 0:
                issues.append({
                    'type': 'duplicate_timestamps',
                    'column': timestamp_col,
                    'count': int(dup_timestamps),
                    'pct': float((dup_timestamps / len(df)) * 100),
                    'severity': 'high',
                    'suggestion': 'Resolve duplicate timestamps by aggregating or removing duplicates'
                })
        
        # Check for time gaps
        if timestamp_col in df.columns and len(df) > 1:
            time_diffs = df[timestamp_col].diff().dropna()
            avg_interval = time_diffs.median()
            
            # Detect irregular intervals
            irregular = (time_diffs > 1.5 * avg_interval).sum()
            if irregular > 0:
                issues.append({
                    'type': 'irregular_intervals',
                    'column': timestamp_col,
                    'count': int(irregular),
                    'pct': float((irregular / (len(df) - 1)) * 100),
                    'severity': 'medium',
                    'suggestion': 'Check for missing data points or implement gap filling'
                })
            
            metrics['time_stats'] = {
                'avg_interval_seconds': float(avg_interval.total_seconds() if hasattr(avg_interval, 'total_seconds') else avg_interval / 1e9),  # Handle both datetime and nanosecond timestamps
                'min_interval_seconds': float(time_diffs.min().total_seconds() if hasattr(time_diffs.min(), 'total_seconds') else time_diffs.min() / 1e9),
                'max_interval_seconds': float(time_diffs.max().total_seconds() if hasattr(time_diffs.max(), 'total_seconds') else time_diffs.max() / 1e9)
            }
        
        # Check for statistical outliers using Z-score
        if price_col in df.columns and len(df) > 10:  # Need sufficient data for Z-score
            z_scores = np.abs(stats.zscore(df[price_col].pct_change().dropna()))
            outliers = (z_scores > self.zscore_threshold).sum()
            
            if outliers > 0:
                issues.append({
                    'type': 'statistical_outliers',
                    'column': price_col,
                    'count': int(outliers),
                    'pct': float((outliers / len(z_scores)) * 100),
                    'severity': 'low',
                    'suggestion': 'Review potential outliers or adjust z-score threshold'
                })
        
        # Generate summary
        num_issues = len(issues)
        num_high = sum(1 for i in issues if i['severity'] == 'high')
        num_medium = sum(1 for i in issues if i['severity'] == 'medium')
        num_low = sum(1 for i in issues if i['severity'] == 'low')
        
        summary = {
            'status': 'warning' if num_high > 0 or num_medium > 5 else 'ok',
            'total_issues': num_issues,
            'high_priority_issues': num_high,
            'medium_priority_issues': num_medium,
            'low_priority_issues': num_low,
            'data_quality_score': max(0, 100 - (num_high * 10 + num_medium * 5 + num_low * 1))
        }
        
        return DataQualityReport(summary=summary, issues=issues, metrics=metrics)
    
    def validate_order_book(self, 
                          order_book: Dict[str, Any],
                          price_tolerance: float = 0.1) -> DataQualityReport:
        """
        Validate order book data quality.
        
        Args:
            order_book: Dictionary with 'bids' and 'asks' lists of [price, quantity] pairs
            price_tolerance: Maximum allowed spread as a fraction of mid price
            
        Returns:
            DataQualityReport with validation results
        """
        issues = []
        metrics = {}
        
        # Check if order book is empty
        if not order_book.get('bids') or not order_book.get('asks'):
            return DataQualityReport(
                summary={'status': 'error', 'message': 'Empty order book'},
                issues=[],
                metrics={}
            )
        
        bids = sorted(order_book['bids'], key=lambda x: -x[0])  # Sort bids descending
        asks = sorted(order_book['asks'], key=lambda x: x[0])   # Sort asks ascending
        
        # Calculate basic metrics
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        
        if best_bid is not None and best_ask is not None:
            spread = best_ask - best_bid
            mid_price = (best_bid + best_ask) / 2
            spread_bps = (spread / mid_price) * 10000
            
            metrics.update({
                'best_bid': float(best_bid),
                'best_ask': float(best_ask),
                'spread': float(spread),
                'spread_bps': float(spread_bps),
                'mid_price': float(mid_price),
                'bid_volume': float(sum(qty for _, qty in bids)),
                'ask_volume': float(sum(qty for _, qty in asks)),
                'order_imbalance': float((sum(qty for _, qty in bids) - sum(qty for _, qty in asks)) / 
                                     (sum(qty for _, qty in bids) + sum(qty for _, qty in asks) + 1e-6))
            })
            
            # Check for crossed book (bid > ask)
            if best_bid >= best_ask:
                issues.append({
                    'type': 'crossed_book',
                    'severity': 'high',
                    'message': f'Crossed book detected: bid={best_bid}, ask={best_ask}',
                    'suggestion': 'Check data source for synchronization issues'
                })
            
            # Check for wide spreads
            if spread_bps > price_tolerance * 100:  # Convert to basis points
                issues.append({
                    'type': 'wide_spread',
                    'severity': 'medium',
                    'message': f'Wide spread: {spread_bps:.2f} bps',
                    'suggestion': 'Check for low liquidity or stale data'
                })
        
        # Check for price dislocations in order book levels
        if len(bids) > 1 and len(asks) > 1:
            bid_prices = [p for p, _ in bids]
            ask_prices = [p for p, _ in asks]
            
            # Check for price dislocations (gaps in price levels)
            bid_diffs = np.diff(bid_prices)
            ask_diffs = np.diff(ask_prices)
            
            if any(d > 0 for d in bid_diffs):
                issues.append({
                    'type': 'bid_price_dislocation',
                    'severity': 'medium',
                    'message': 'Non-monotonic bid prices detected',
                    'suggestion': 'Check data source for synchronization issues'
                })
                
            if any(d < 0 for d in ask_diffs):
                issues.append({
                    'type': 'ask_price_dislocation',
                    'severity': 'medium',
                    'message': 'Non-monotonic ask prices detected',
                    'suggestion': 'Check data source for synchronization issues'
                })
        
        # Generate summary
        summary = {
            'status': 'ok' if not issues else 'warning',
            'total_issues': len(issues),
            'high_priority_issues': sum(1 for i in issues if i['severity'] == 'high'),
            'medium_priority_issues': sum(1 for i in issues if i['severity'] == 'medium'),
            'low_priority_issues': sum(1 for i in issues if i['severity'] == 'low'),
        }
        
        return DataQualityReport(summary=summary, issues=issues, metrics=metrics)
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown."""
        if len(prices) < 2:
            return 0.0
            
        cummax = prices.cummax()
        drawdowns = (prices - cummax) / (cummax + 1e-6)
        return float(drawdowns.min())
    
    def generate_validation_report(self, 
                                 ohlcv_data: pd.DataFrame,
                                 order_book: Optional[Dict[str, Any]] = None,
                                 price_col: str = 'close',
                                 volume_col: str = 'volume',
                                 timestamp_col: str = 'timestamp') -> Dict[str, Any]:
        """
        Generate a comprehensive validation report for market data.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            order_book: Optional order book data
            price_col: Name of the price column
            volume_col: Name of the volume column
            timestamp_col: Name of the timestamp column
            
        Returns:
            Dictionary with validation results
        """
        report = {}
        
        # Validate OHLCV data
        ohlcv_report = self.validate_ohlcv(
            df=ohlcv_data,
            price_col=price_col,
            volume_col=volume_col,
            timestamp_col=timestamp_col
        )
        
        report['ohlcv_validation'] = {
            'summary': ohlcv_report.summary,
            'metrics': ohlcv_report.metrics,
            'issues': [i for i in ohlcv_report.issues if i['severity'] in ('high', 'medium')]  # Only include high/medium issues in main report
        }
        
        # Validate order book if provided
        if order_book:
            ob_report = self.validate_order_book(order_book)
            report['order_book_validation'] = {
                'summary': ob_report.summary,
                'metrics': ob_report.metrics,
                'issues': ob_report.issues
            }
        
        # Generate overall status
        has_high_issues = any(r['summary'].get('high_priority_issues', 0) > 0 
                             for r in report.values() if 'summary' in r)
        has_medium_issues = any(r['summary'].get('medium_priority_issues', 0) > 0 
                              for r in report.values() if 'summary' in r)
        
        report['overall_status'] = {
            'status': 'error' if has_high_issues else 'warning' if has_medium_issues else 'ok',
            'timestamp': pd.Timestamp.utcnow().isoformat(),
            'checks_performed': list(report.keys())
        }
        
        return report
