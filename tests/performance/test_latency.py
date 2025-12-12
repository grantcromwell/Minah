""
Performance and latency tests for the trading system
"""
import time
import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

class TestLatency:
    """Tests for system latency characteristics"""
    
    @pytest.mark.parametrize("num_orders", [100, 1000, 10000])
    def test_order_processing_latency(self, benchmark, num_orders):
        """Test order processing latency under different loads"""
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        
        def setup(num):
            ob = OrderBook(symbol="TEST")
            engine = ExecutionEngine()
            orders = [
                {
                    'type': 'limit',
                    'side': 'buy' if i % 2 == 0 else 'sell',
                    'price': 100.0 + (0.1 * (i % 10)),
                    'size': 1.0,
                    'trader_id': f'TRADER_{i}',
                    'timestamp': datetime.now()
                }
                for i in range(num)
            ]
            return ob, engine, orders
        
        # Benchmark the order processing
        def process_orders(ob, engine, orders):
            results = []
            for order in orders:
                start = time.perf_counter_ns()
                result = engine.execute_order(order, ob)
                end = time.perf_counter_ns()
                results.append((end - start) / 1e6)  # Convert to milliseconds
            return results
        
        # Run benchmark
        ob, engine, orders = setup(num_orders)
        latencies = benchmark(process_orders, ob, engine, orders)
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'num_orders': num_orders,
            'avg_latency_ms': np.mean(latencies),
            'p50_ms': np.percentile(latencies, 50),
            'p95_ms': np.percentile(latencies, 95),
            'p99_ms': np.percentile(latencies, 99),
            'max_ms': np.max(latencies)
        }
        
        # Log results
        print(f"\nOrder Processing Latency (n={num_orders}):")
        for k, v in stats.items():
            if 'latency' in k or '_ms' in k:
                print(f"  {k}: {v:.3f} ms")
            else:
                print(f"  {k}: {v}")
        
        # Assert performance requirements
        assert stats['p99_ms'] < 10.0, "99th percentile latency exceeds 10ms"
    
    def test_market_data_throughput(self, benchmark):
        """Test market data processing throughput"""
        from src.abm.market_data import MarketDataProcessor
        
        # Generate sample market data
        num_updates = 10000
        updates = [
            {
                'timestamp': datetime.now() + timedelta(milliseconds=i),
                'symbol': 'TEST',
                'bid': 99.9 + np.random.random() * 0.2,
                'ask': 100.1 + np.random.random() * 0.2,
                'bid_size': np.random.uniform(1, 10),
                'ask_size': np.random.uniform(1, 10)
            }
            for i in range(num_updates)
        ]
        
        # Initialize processor
        processor = MarketDataProcessor()
        
        # Benchmark processing
        def process_updates():
            for update in updates:
                processor.process_update(update)
        
        # Run benchmark
        benchmark(process_updates)
        
        # Calculate throughput
        elapsed = benchmark.stats['mean']
        throughput = num_updates / elapsed
        
        print(f"\nMarket Data Throughput: {throughput:,.0f} updates/second")
        
        # Assert minimum throughput requirement
        assert throughput > 100000, "Throughput below 100K updates/second"

class TestMemoryUsage:
    """Tests for memory usage characteristics"""
    
    def test_order_book_memory_usage(self):
        """Test memory usage of order book with large number of orders"""
        import tracemalloc
        from src.abm.order_book import OrderBook
        
        # Start tracking memory
        tracemalloc.start()
        
        # Initial memory usage
        snapshot1 = tracemalloc.take_snapshot()
        
        # Create order book and add orders
        ob = OrderBook(symbol="TEST")
        num_orders = 10000
        
        for i in range(num_orders):
            side = 'bid' if i % 2 == 0 else 'ask'
            price = 100.0 + (0.1 * (i % 100))
            ob.add_order(price=price, size=1.0, side=side)
        
        # Take another snapshot
        snapshot2 = tracemalloc.take_snapshot()
        
        # Calculate memory usage
        diff = snapshot2.compare_to(snapshot1, 'lineno')
        total_memory = sum(stat.size for stat in diff)
        memory_per_order = total_memory / num_orders
        
        print(f"\nOrder Book Memory Usage:")
        print(f"  Total orders: {num_orders:,}")
        print(f"  Total memory: {total_memory / 1024:.2f} KB")
        print(f"  Memory per order: {memory_per_order:.2f} bytes")
        
        # Assert memory efficiency
        assert memory_per_order < 1000, "Memory per order exceeds 1KB"
        
        # Cleanup
        tracemalloc.stop()

class TestConcurrency:
    """Tests for concurrent operation"""
    
    @pytest.mark.parametrize("num_threads", [1, 2, 4, 8])
    def test_concurrent_order_processing(self, benchmark, num_threads):
        """Test order processing with multiple threads"""
        import concurrent.futures
        from src.abm.order_book import OrderBook
        from src.abm.execution import ExecutionEngine
        
        def worker(worker_id, num_orders, results):
            ob = OrderBook(symbol=f"TEST_{worker_id}")
            engine = ExecutionEngine()
            
            for i in range(num_orders):
                order = {
                    'type': 'limit',
                    'side': 'buy' if (i + worker_id) % 2 == 0 else 'sell',
                    'price': 100.0 + (0.1 * (i % 10)),
                    'size': 1.0,
                    'trader_id': f'WORKER_{worker_id}_{i}',
                    'timestamp': datetime.now()
                }
                start = time.perf_counter_ns()
                engine.execute_order(order, ob)
                end = time.perf_counter_ns()
                results.append((end - start) / 1e6)  # ms
        
        def run_test():
            num_orders_per_thread = 1000
            results = []
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [
                    executor.submit(worker, i, num_orders_per_thread, results)
                    for i in range(num_threads)
                ]
                concurrent.futures.wait(futures)
            
            return results
        
        # Run benchmark
        latencies = benchmark(run_test)
        
        # Calculate statistics
        latencies = np.array(latencies)
        stats = {
            'num_threads': num_threads,
            'total_orders': len(latencies),
            'avg_latency_ms': np.mean(latencies),
            'p99_latency_ms': np.percentile(latencies, 99)
        }
        
        # Log results
        print(f"\nConcurrent Order Processing ({num_threads} threads):")
        for k, v in stats.items():
            if 'latency' in k or '_ms' in k:
                print(f"  {k}: {v:.3f} ms")
            else:
                print(f"  {k}: {v:,}")
        
        # Assert performance doesn't degrade too much with more threads
        if num_threads > 1:
            single_thread_avg = 5.0  # Expected single-thread latency in ms
            expected_throughput = num_threads * (1000 / single_thread_avg)
            actual_throughput = len(latencies) / (np.sum(latencies) / 1000)
            efficiency = (actual_throughput / expected_throughput) * 100
            
            print(f"  Parallel efficiency: {efficiency:.1f}%")
            assert efficiency > 70, f"Parallel efficiency below 70%: {efficiency:.1f}%"
