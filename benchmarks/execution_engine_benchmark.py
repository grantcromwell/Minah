"""
Benchmark for comparing the original and optimized execution engines.
"""
import time
import random
import statistics
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import pytz
import pandas as pd
import numpy as np

# Add parent directory to path
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import the execution engines
from src.abm.execution.engine import ExecutionEngine as OriginalExecutionEngine
from src.abm.execution.optimized_engine import OptimizedExecutionEngine, result_pool
from src.abm.interfaces.order_book import Order, OrderSide, OrderBook
from src.data import MarketDataConnector, DataPipeline

# Configure logging
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DummyOrderBook(OrderBook):
    """Dummy order book for testing."""
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.bid = 100.0
        self.ask = 101.0
        self.spread = 1.0
        
    def get_bid_ask(self) -> Tuple[float, float]:
        """Get current bid and ask prices."""
        return self.bid, self.ask
        
    def get_best_bid_ask(self) -> Tuple[float, float]:
        """Get best bid and ask prices."""
        return self.bid, self.ask
        
    def get_bid(self) -> float:
        """Get current bid price."""
        return self.bid
        
    def get_ask(self) -> float:
        """Get current ask price."""
        return self.ask

def generate_test_orders(num_orders: int, symbols: List[str]) -> List[Order]:
    """Generate a list of test orders."""
    orders = []
    order_id = 1
    
    for _ in range(num_orders):
        symbol = random.choice(symbols)
        side = random.choice([OrderSide.BUY, OrderSide.SELL])
        order_type = random.choices(
            [OrderType.MARKET, OrderType.LIMIT],
            weights=[0.3, 0.7],  # 30% market, 70% limit orders
            k=1
        )[0]
        
        price = 100.0 + random.uniform(-5.0, 5.0) if order_type == OrderType.LIMIT else None
        quantity = random.randint(1, 100)
        
        orders.append(Order(
            order_id=f"test_{order_id}",
            symbol=symbol,
            side=side,
            order_type=order_type,
            quantity=quantity,
            price=price,
            timestamp=datetime.now(pytz.utc).timestamp()
        ))
        
        order_id += 1
    
    return orders

class ExecutionEngineBenchmark:
    """Benchmark for comparing execution engine performance."""
    
    def __init__(self, num_orders: int = 10000, num_symbols: int = 10):
        """Initialize the benchmark."""
        self.num_orders = num_orders
        self.symbols = [f"SYM_{i}" for i in range(1, num_symbols + 1)]
        self.orders = generate_test_orders(num_orders, self.symbols)
        
        # Set up order books
        self.order_books = {symbol: DummyOrderBook(symbol) for symbol in self.symbols}
        
        # Initialize engines
        self.original_engine = OriginalExecutionEngine(
            order_books=self.order_books,
            data_connector=MarketDataConnector(),
            data_pipeline=DataPipeline(),
            config={
                'risk_limits': {
                    'position_limits': {
                        'max_position_size': 10000.0,
                        'max_portfolio_exposure': 0.5,
                        'max_leverage': 10.0,
                        'max_drawdown': 0.1
                    },
                    'order_limits': {
                        'max_order_size': 1000.0,
                        'max_order_value': 100000.0,
                        'max_orders_per_minute': 10000,
                        'min_order_size': 0.01,
                        'price_band_pct': 0.1
                    }
                }
            }
        )
        
        self.optimized_engine = OptimizedExecutionEngine(
            order_books=self.order_books,
            data_connector=MarketDataConnector(),
            data_pipeline=DataPipeline(),
            config={
                'max_workers': 4,
                'risk_limits': {
                    'position_limits': {
                        'max_position_size': 10000.0,
                        'max_portfolio_exposure': 0.5,
                        'max_leverage': 10.0,
                        'max_drawdown': 0.1
                    },
                    'order_limits': {
                        'max_order_size': 1000.0,
                        'max_order_value': 100000.0,
                        'max_orders_per_minute': 10000,
                        'min_order_size': 0.01,
                        'price_band_pct': 0.1
                    }
                }
            }
        )
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the benchmark and return results."""
        logger.info(f"Starting benchmark with {self.num_orders} orders...")
        
        # Test original engine
        logger.info("Testing original execution engine...")
        orig_results = self._test_engine(self.original_engine, self.orders)
        
        # Test optimized engine
        logger.info("Testing optimized execution engine...")
        opt_results = self._test_engine(self.optimized_engine, self.orders)
        
        # Calculate improvements
        orig_avg_latency = statistics.mean(orig_results['latencies'])
        opt_avg_latency = statistics.mean(opt_results['latencies'])
        
        orig_throughput = self.num_orders / orig_results['total_time']
        opt_throughput = self.num_orders / opt_results['total_time']
        
        return {
            'original': {
                'total_time': orig_results['total_time'],
                'avg_latency_ms': orig_avg_latency * 1000,
                'throughput_ops': orig_throughput,
                'metrics': orig_results['metrics']
            },
            'optimized': {
                'total_time': opt_results['total_time'],
                'avg_latency_ms': opt_avg_latency * 1000,
                'throughput_ops': opt_throughput,
                'metrics': opt_results['metrics']
            },
            'improvement': {
                'time_reduction_pct': (1 - (opt_results['total_time'] / orig_results['total_time'])) * 100,
                'latency_improvement': (1 - (opt_avg_latency / orig_avg_latency)) * 100,
                'throughput_improvement': ((opt_throughput / orig_throughput) - 1) * 100
            }
        }
    
    def _test_engine(self, engine, orders: List[Order]) -> Dict[str, Any]:
        """Test a single execution engine."""
        latencies = []
        results = []
        
        start_time = time.time()
        
        for order in orders:
            # Create a copy of the order to avoid modifying the original
            order_copy = Order(
                order_id=order.order_id,
                symbol=order.symbol,
                side=order.side,
                order_type=order.order_type,
                quantity=order.quantity,
                price=order.price,
                timestamp=order.timestamp
            )
            
            # Process the order and measure latency
            order_start = time.time()
            result = engine.submit_order(order_copy)
            order_latency = time.time() - order_start
            
            latencies.append(order_latency)
            results.append(result)
            
            # Release the result if it's from the optimized engine
            if hasattr(engine, 'optimized') and engine.optimized:
                result.release()
        
        total_time = time.time() - start_time
        
        # Collect metrics
        metrics = {
            'total_orders': len(results),
            'filled_orders': sum(1 for r in results if r.status.name in ['FILLED', 'PARTIALLY_FILLED']),
            'rejected_orders': sum(1 for r in results if r.status.name == 'REJECTED'),
            'pending_orders': sum(1 for r in results if r.status.name == 'PENDING'),
        }
        
        return {
            'total_time': total_time,
            'latencies': latencies,
            'metrics': metrics
        }

def print_benchmark_results(results: Dict[str, Any]) -> None:
    """Print benchmark results in a readable format."""
    orig = results['original']
    opt = results['optimized']
    imp = results['improvement']
    
    print("\n" + "="*80)
    print("EXECUTION ENGINE BENCHMARK RESULTS")
    print("="*80)
    
    print(f"\n{'METRIC':<30} | {'ORIGINAL':<20} | {'OPTIMIZED':<20} | {'IMPROVEMENT':<20}")
    print("-" * 80)
    
    print(f"{'Total Time (s)':<30} | {orig['total_time']:<20.4f} | {opt['total_time']:<20.4f} | {imp['time_reduction_pct']:>18.2f}%")
    print(f"{'Avg Latency (ms)':<30} | {orig['avg_latency_ms']:<20.4f} | {opt['avg_latency_ms']:<20.4f} | {imp['latency_improvement']:>18.2f}%")
    print(f"{'Throughput (orders/s)':<30} | {orig['throughput_ops']:<20.2f} | {opt['throughput_ops']:<20.2f} | {imp['throughput_improvement']:>18.2f}%")
    
    print("\nORDER METRICS:")
    print(f"- Filled Orders:    {orig['metrics']['filled_orders']} (Original) vs {opt['metrics']['filled_orders']} (Optimized)")
    print(f"- Rejected Orders:  {orig['metrics']['rejected_orders']} (Original) vs {opt['metrics']['rejected_orders']} (Optimized)")
    print(f"- Pending Orders:   {orig['metrics']['pending_orders']} (Original) vs {opt['metrics']['pending_orders']} (Optimized)")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Benchmark execution engine performance')
    parser.add_argument('--orders', type=int, default=10000, help='Number of orders to process')
    parser.add_argument('--symbols', type=int, default=10, help='Number of symbols to trade')
    parser.add_argument('--runs', type=int, default=3, help='Number of benchmark runs')
    
    args = parser.parse_args()
    
    all_results = []
    
    for run in range(1, args.runs + 1):
        print(f"\n{'='*40} RUN {run}/{args.runs} {'='*40}")
        
        benchmark = ExecutionEngineBenchmark(
            num_orders=args.orders,
            num_symbols=args.symbols
        )
        
        results = benchmark.run_benchmark()
        print_benchmark_results(results)
        all_results.append(results)
    
    # Calculate average results
    if args.runs > 1:
        avg_results = {
            'original': {
                'total_time': statistics.mean(r['original']['total_time'] for r in all_results),
                'avg_latency_ms': statistics.mean(r['original']['avg_latency_ms'] for r in all_results),
                'throughput_ops': statistics.mean(r['original']['throughput_ops'] for r in all_results),
                'metrics': {
                    'filled_orders': int(statistics.mean(r['original']['metrics']['filled_orders'] for r in all_results)),
                    'rejected_orders': int(statistics.mean(r['original']['metrics']['rejected_orders'] for r in all_results)),
                    'pending_orders': int(statistics.mean(r['original']['metrics']['pending_orders'] for r in all_results)),
                }
            },
            'optimized': {
                'total_time': statistics.mean(r['optimized']['total_time'] for r in all_results),
                'avg_latency_ms': statistics.mean(r['optimized']['avg_latency_ms'] for r in all_results),
                'throughput_ops': statistics.mean(r['optimized']['throughput_ops'] for r in all_results),
                'metrics': {
                    'filled_orders': int(statistics.mean(r['optimized']['metrics']['filled_orders'] for r in all_results)),
                    'rejected_orders': int(statistics.mean(r['optimized']['metrics']['rejected_orders'] for r in all_results)),
                    'pending_orders': int(statistics.mean(r['optimized']['metrics']['pending_orders'] for r in all_results)),
                }
            },
            'improvement': {
                'time_reduction_pct': statistics.mean(r['improvement']['time_reduction_pct'] for r in all_results),
                'latency_improvement': statistics.mean(r['improvement']['latency_improvement'] for r in all_results),
                'throughput_improvement': statistics.mean(r['improvement']['throughput_improvement'] for r in all_results),
            }
        }
        
        print("\n" + "="*80)
        print(f"AVERAGE RESULTS OVER {args.runs} RUNS")
        print("="*80)
        print_benchmark_results(avg_results)
