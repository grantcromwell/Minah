#!/usr/bin/env python3
"""
ABM Trading System Simulation

This script demonstrates the Agent-Based Model (ABM) trading system with market maker
and trend-following agents interacting in a simulated market environment.
"""
import os
import sys
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.abm.models.market_environment import MarketEnvironment
from src.abm.agents.market_maker import MarketMakerAgent
from src.abm.agents.trend_follower import TrendFollowingAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('abm_simulation.log')
    ]
)
logger = logging.getLogger(__name__)

def run_simulation():
    """Run the ABM trading simulation."""
    # Simulation parameters
    symbols = ['BTC-USD', 'ETH-USD']
    initial_prices = {'BTC-USD': 50000.0, 'ETH-USD': 3000.0}
    num_steps = 1000
    
    # Create market environment
    logger.info("Initializing market environment...")
    env = MarketEnvironment(
        symbols=symbols,
        initial_prices=initial_prices,
        use_gpu=True
    )
    
    # Create and add agents
    logger.info("Creating agents...")
    
    # Market Makers
    for i, symbol in enumerate(symbols):
        mm = MarketMakerAgent(
            symbol=symbol,
            spread_target=0.001,  # 10bps
            order_size=0.1 if symbol == 'BTC-USD' else 1.0,
            inventory_target=0.0,
            max_position=100.0,
            agent_id=f'mm_{symbol}_{i}'
        )
        env.add_agent(mm)
    
    # Trend Followers
    for i, symbol in enumerate(symbols):
        for j in range(5):  # 5 trend followers per symbol
            tf = TrendFollowingAgent(
                symbol=symbol,
                lookback_period=20,
                entry_threshold=0.001,
                exit_threshold=0.0005,
                position_size=0.5 if symbol == 'BTC-USD' else 5.0,
                max_position=10.0,
                agent_id=f'tf_{symbol}_{i}_{j}'
            )
            env.add_agent(tf)
    
    # Run simulation
    logger.info(f"Starting simulation with {len(env.agents)} agents for {num_steps} steps...")
    
    # Data collection
    price_history = {symbol: [] for symbol in symbols}
    spread_history = {symbol: [] for symbol in symbols}
    volume_history = {symbol: [] for symbol in symbols}
    
    try:
        for step in range(num_steps):
            # Add some price movement
            if step > 0:
                for symbol in symbols:
                    # Random walk with some momentum
                    prev_price = env.market_data['prices'][symbol]
                    momentum = 0.7  # High momentum for trending behavior
                    shock = np.random.normal(0, 0.001)  # Small random shocks
                    
                    # Add some periodic trends
                    trend = 0.0005 * np.sin(step / 50)  # Long-term trend
                    
                    # Update price
                    new_price = prev_price * (1 + momentum * (prev_price / initial_prices[symbol] - 1) / 100 + shock + trend)
                    env.market_data['prices'][symbol] = max(0.01, new_price)  # Prevent negative prices
            
            # Step the environment
            env.step()
            
            # Record data
            for symbol in symbols:
                price_history[symbol].append(env.market_data['prices'][symbol])
                spread_history[symbol].append(env.market_data['bid_ask_spread'][symbol])
                volume_history[symbol].append(env.market_data['volumes'][symbol])
            
            # Log progress
            if (step + 1) % 100 == 0:
                logger.info(f"Step {step + 1}/{num_steps} completed")
                
                # Log performance metrics
                metrics = env.get_performance_metrics()
                logger.info(f"Performance: {metrics}")
    
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
    
    # Save results
    logger.info("Saving simulation results...")
    results = {}
    for symbol in symbols:
        results[f'{symbol}_price'] = price_history[symbol]
        results[f'{symbol}_spread'] = spread_history[symbol]
        results[f'{symbol}_volume'] = volume_history[symbol]
    
    df = pd.DataFrame(results)
    df.to_csv('abm_simulation_results.csv', index=False)
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot prices
    plt.subplot(3, 1, 1)
    for symbol in symbols:
        plt.plot(price_history[symbol], label=f'{symbol} Price')
    plt.title('Price Movement')
    plt.legend()
    plt.grid(True)
    
    # Plot spreads
    plt.subplot(3, 1, 2)
    for symbol in symbols:
        plt.plot(spread_history[symbol], label=f'{symbol} Spread')
    plt.title('Bid-Ask Spread')
    plt.legend()
    plt.grid(True)
    
    # Plot volumes
    plt.subplot(3, 1, 3)
    for symbol in symbols:
        plt.plot(volume_history[symbol], label=f'{symbol} Volume')
    plt.title('Trading Volume')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('abm_simulation_results.png')
    plt.close()
    
    logger.info("Simulation completed. Results saved to abm_simulation_results.csv and abm_simulation_results.png")

if __name__ == "__main__":
    run_simulation()
