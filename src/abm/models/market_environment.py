from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor
import cupy as cp
from numba import cuda
import logging

from ..agents.base_agent import BaseAgent, AgentType
from ..interfaces.order_book import OrderBook

logger = logging.getLogger(__name__)

class MarketEnvironment:
    """Core market environment for the ABM system."""
    
    def __init__(self, 
                 symbols: List[str], 
                 initial_prices: Dict[str, float],
                 num_agents: int = 100,
                 use_gpu: bool = True,
                 **kwargs):
        """
        Initialize the market environment.
        
        Args:
            symbols: List of trading symbols
            initial_prices: Initial prices for each symbol
            num_agents: Number of agents to initialize
            use_gpu: Whether to use GPU acceleration
        """
        self.symbols = symbols
        self.current_step = 0
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.agents = []
        self.order_books = {symbol: OrderBook(symbol) for symbol in symbols}
        self.market_data = self._initialize_market_data(initial_prices)
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.performance_metrics = {
            'step_times': [],
            'agent_update_times': [],
            'order_processing_times': []
        }
        
        # Initialize GPU if available
        if self.use_gpu:
            self._initialize_gpu()
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU is available for computation."""
        try:
            return cuda.is_available()
        except Exception as e:
            logger.warning(f"GPU check failed: {e}")
            return False
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU resources if available."""
        if not self.use_gpu:
            return
            
        try:
            # Initialize CUDA context
            cuda.select_device(0)
            logger.info(f"Using GPU: {cuda.gpus[0].name.decode('utf-8')}")
            
            # Warm up GPU
            a = cp.ones((1000, 1000))
            cp.dot(a, a)
            
        except Exception as e:
            logger.error(f"Failed to initialize GPU: {e}")
            self.use_gpu = False
    
    def _initialize_market_data(self, initial_prices: Dict[str, float]) -> Dict[str, Any]:
        """Initialize the market data structure."""
        return {
            'prices': initial_prices.copy(),
            'volumes': {symbol: 0.0 for symbol in self.symbols},
            'bid_ask_spread': {symbol: 0.0 for symbol in self.symbols},
            'order_imbalance': {symbol: 0.0 for symbol in self.symbols},
            'timestamp': datetime.utcnow().timestamp(),
            'metrics': {}
        }
    
    def add_agent(self, agent: BaseAgent) -> None:
        """Add an agent to the simulation."""
        self.agents.append(agent)
    
    def step(self) -> None:
        """Advance the simulation by one time step."""
        step_start = time.time()
        self.current_step += 1
        
        # Update market data
        self._update_market_data()
        
        # Let agents decide their actions
        orders = self._gather_agent_actions()
        
        # Process orders
        self._process_orders(orders)
        
        # Update performance metrics
        step_time = time.time() - step_start
        self.performance_metrics['step_times'].append(step_time)
        
        logger.debug(f"Step {self.current_step} completed in {step_time:.4f}s")
    
    def _update_market_data(self) -> None:
        """Update market data based on current order book state."""
        for symbol in self.symbols:
            book = self.order_books[symbol]
            self.market_data['prices'][symbol] = book.get_mid_price()
            self.market_data['bid_ask_spread'][symbol] = book.get_spread()
            self.market_data['order_imbalance'][symbol] = book.get_order_imbalance()
        
        self.market_data['timestamp'] = datetime.utcnow().timestamp()
    
    def _gather_agent_actions(self) -> List[Dict]:
        """Gather actions from all agents."""
        orders = []
        agent_update_times = []
        
        for agent in self.agents:
            try:
                start_time = time.time()
                agent_orders = agent.decide_actions(self.market_data, self.current_step)
                agent_update_times.append(time.time() - start_time)
                
                if agent_orders:
                    if isinstance(agent_orders, dict):
                        agent_orders = [agent_orders]
                    orders.extend(agent_orders)
            
            except Exception as e:
                logger.error(f"Error in agent {agent.agent_id}: {str(e)}", exc_info=True)
        
        if agent_update_times:
            avg_agent_time = sum(agent_update_times) / len(agent_update_times)
            self.performance_metrics['agent_update_times'].append(avg_agent_time)
        
        return orders
    
    def _process_orders(self, orders: List[Dict]) -> None:
        """Process a batch of orders."""
        if not orders:
            return
        
        start_time = time.time()
        
        # Group orders by symbol for batch processing
        orders_by_symbol = {}
        for order in orders:
            symbol = order.get('symbol')
            if symbol not in orders_by_symbol:
                orders_by_symbol[symbol] = []
            orders_by_symbol[symbol].append(order)
        
        # Process orders for each symbol
        for symbol, symbol_orders in orders_by_symbol.items():
            if symbol not in self.order_books:
                logger.warning(f"Skipping orders for unknown symbol: {symbol}")
                continue
                
            order_book = self.order_books[symbol]
            
            # Process orders in parallel if using GPU
            if self.use_gpu and len(symbol_orders) > 100:  # Threshold for GPU processing
                self._process_orders_gpu(order_book, symbol_orders)
            else:
                for order in symbol_orders:
                    order_book.process_order(order)
        
        # Record order processing performance
        proc_time = time.time() - start_time
        self.performance_metrics['order_processing_times'].append(proc_time)
    
    def _process_orders_gpu(self, order_book, orders: List[Dict]) -> None:
        """Process a batch of orders using GPU acceleration."""
        try:
            # Convert orders to GPU arrays for batch processing
            # This is a simplified example - actual implementation would depend on order structure
            prices = cp.array([o.get('price', 0) for o in orders])
            quantities = cp.array([o.get('quantity', 0) for o in orders])
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            for i in range(0, len(orders), batch_size):
                batch_prices = prices[i:i+batch_size]
                batch_quantities = quantities[i:i+batch_size]
                
                # Simulate order processing (replace with actual GPU processing)
                # In a real implementation, this would use CUDA kernels
                cp.dot(batch_prices, batch_quantities)
                
            # Process orders in the order book (still CPU-bound for this example)
            for order in orders:
                order_book.process_order(order)
                
        except Exception as e:
            logger.error(f"GPU order processing failed: {e}")
            # Fall back to CPU processing
            for order in orders:
                order_book.process_order(order)
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for the simulation."""
        if not self.performance_metrics['step_times']:
            return {}
            
        return {
            'avg_step_time': np.mean(self.performance_metrics['step_times']),
            'max_step_time': np.max(self.performance_metrics['step_times']),
            'min_step_time': np.min(self.performance_metrics['step_times']),
            'avg_agent_update_time': np.mean(self.performance_metrics['agent_update_times']) if self.performance_metrics['agent_update_times'] else 0,
            'avg_order_processing_time': np.mean(self.performance_metrics['order_processing_times']) if self.performance_metrics['order_processing_times'] else 0,
            'total_steps': len(self.performance_metrics['step_times']),
            'gpu_enabled': self.use_gpu
        }
    
    def reset(self) -> None:
        """Reset the environment to its initial state."""
        self.current_step = 0
        self.performance_metrics = {k: [] for k in self.performance_metrics}
        
        # Reset order books
        for book in self.order_books.values():
            book.reset()
            
        # Reset agents
        for agent in self.agents:
            if hasattr(agent, 'reset'):
                agent.reset()
    
    def __del__(self):
        """Clean up resources."""
        self.executor.shutdown(wait=False)
