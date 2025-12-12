"""
Utility functions for reinforcement learning.
"""

import numpy as np
import tensorflow as tf
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
import gym
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
import os
import json
import time
from collections import deque

from .environment import TradingEnvironment
from .agents import DQNAgent, PPOAgent, SACAgent, create_agent
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, NStepBuffer

def create_environment(
    data: pd.DataFrame,
    env_type: str = 'trading',
    **kwargs
) -> gym.Env:
    """
    Create a trading environment.
    
    Args:
        data: DataFrame with OHLCV data and features
        env_type: Type of environment ('trading' or 'gym' for standard Gym envs)
        **kwargs: Additional arguments for the environment
        
    Returns:
        A Gym environment
    """
    if env_type == 'trading':
        return TradingEnvironment(data, **kwargs)
    else:
        # For standard Gym environments
        return gym.make(env_type, **kwargs)

def create_vectorized_envs(
    env_fn: Callable,
    num_envs: int = 4,
    **kwargs
) -> List[gym.Env]:
    """
    Create multiple environments for parallel training.
    
    Args:
        env_fn: Function that creates an environment
        num_envs: Number of environments to create
        **kwargs: Additional arguments for the environment
        
    Returns:
        List of environments
    """
    return [env_fn(**kwargs) for _ in range(num_envs)]

def setup_experiment(
    experiment_name: str,
    base_dir: str = 'experiments',
    config: Optional[Dict[str, Any]] = None
) -> str:
    """
    Set up an experiment directory and save the configuration.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for experiments
        config: Experiment configuration
        
    Returns:
        Path to the experiment directory
    """
    # Create experiment directory
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    exp_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Save configuration
    if config is not None:
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    
    # Create subdirectories
    os.makedirs(os.path.join(exp_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'results'), exist_ok=True)
    
    return exp_dir

def evaluate_agent(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 10,
    render: bool = False,
    deterministic: bool = True
) -> Dict[str, float]:
    """
    Evaluate an agent on an environment.
    
    Args:
        agent: The agent to evaluate
        env: The environment to evaluate on
        n_episodes: Number of episodes to run
        render: Whether to render the environment
        deterministic: Whether to use deterministic actions
        
    Returns:
        Dictionary of evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    
    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            if render:
                env.render()
                
            # Get action from agent
            if hasattr(agent, 'get_action'):
                action = agent.get_action(state, training=False)
            elif hasattr(agent, 'predict'):
                action, _ = agent.predict(state, deterministic=deterministic)
            else:
                raise ValueError("Agent must have 'get_action' or 'predict' method")
            
            # Take a step in the environment
            next_state, reward, done, _, _ = env.step(action)
            
            episode_reward += reward
            episode_length += 1
            state = next_state
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Calculate metrics
    metrics = {
        'mean_reward': float(np.mean(episode_rewards)),
        'std_reward': float(np.std(episode_rewards)),
        'min_reward': float(np.min(episode_rewards)),
        'max_reward': float(np.max(episode_rewards)),
        'mean_episode_length': float(np.mean(episode_lengths)),
        'n_episodes': n_episodes
    }
    
    return metrics

def plot_training_metrics(
    metrics: Dict[str, List[float]],
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary of metrics to plot
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    plt.figure(figsize=(12, 8))
    
    # Plot rewards
    plt.subplot(2, 1, 1)
    if 'episode_rewards' in metrics:
        plt.plot(metrics['episode_rewards'], label='Episode Reward')
    if 'mean_rewards' in metrics:
        plt.plot(metrics['mean_rewards'], label='Mean Reward (100 episodes)')
    plt.title('Training Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # Plot losses
    plt.subplot(2, 1, 2)
    if 'losses' in metrics:
        plt.plot(metrics['losses'], label='Loss')
    if 'critic_losses' in metrics:
        plt.plot(metrics['critic_losses'], label='Critic Loss')
    if 'actor_losses' in metrics:
        plt.plot(metrics['actor_losses'], label='Actor Loss')
    plt.title('Training Losses')
    plt.xlabel('Update Step')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def plot_portfolio_value(
    env: TradingEnvironment,
    save_path: Optional[str] = None,
    show: bool = True
) -> None:
    """
    Plot the portfolio value over time.
    
    Args:
        env: The trading environment with portfolio history
        save_path: Path to save the plot (optional)
        show: Whether to display the plot
    """
    if not hasattr(env, 'portfolio_history') or not env.portfolio_history:
        print("No portfolio history available.")
        return
    
    history = pd.DataFrame(env.portfolio_history)
    
    plt.figure(figsize=(12, 8))
    
    # Plot portfolio value and price
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(history['step'], history['portfolio_value'], 'b-', label='Portfolio Value')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Portfolio Value', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    ax1.grid(True)
    
    ax2 = ax1.twinx()
    ax2.plot(history['step'], history['price'], 'r-', label='Price')
    ax2.set_ylabel('Price', color='r')
    ax2.tick_params(axis='y', labelcolor='r')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Plot position
    plt.subplot(2, 1, 2)
    plt.bar(history['step'], history['position_value'], width=1.0, alpha=0.5, label='Position Value')
    plt.xlabel('Step')
    plt.ylabel('Position Value')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    
    if show:
        plt.show()
    else:
        plt.close()

def save_model(
    agent: Any,
    path: str,
    save_replay_buffer: bool = False
) -> None:
    """
    Save an agent's model and optional replay buffer.
    
    Args:
        agent: The agent to save
        path: Path to save the model
        save_replay_buffer: Whether to save the replay buffer
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Save the agent's model
    if hasattr(agent, 'save'):
        agent.save(path)
    elif hasattr(agent, 'save_weights'):
        agent.save_weights(path + ".h5")
    else:
        raise ValueError("Agent does not have a save method")
    
    # Save replay buffer if requested and available
    if save_replay_buffer and hasattr(agent, 'replay_buffer'):
        buffer_path = path + "_buffer.pkl"
        with open(buffer_path, 'wb') as f:
            import pickle
            pickle.dump(agent.replay_buffer, f)

def load_model(
    agent_class: Any,
    path: str,
    load_replay_buffer: bool = False,
    **kwargs
) -> Any:
    """
    Load an agent's model and optional replay buffer.
    
    Args:
        agent_class: The agent class to instantiate
        path: Path to the saved model
        load_replay_buffer: Whether to load the replay buffer
        **kwargs: Additional arguments for agent initialization
        
    Returns:
        The loaded agent
    """
    # Create agent instance
    agent = agent_class(**kwargs)
    
    # Load the model
    if hasattr(agent, 'load'):
        agent = agent.load(path)
    elif hasattr(agent, 'load_weights'):
        agent.load_weights(path + ".h5")
    else:
        raise ValueError("Agent does not have a load method")
    
    # Load replay buffer if requested and available
    if load_replay_buffer and hasattr(agent, 'replay_buffer'):
        buffer_path = path + "_buffer.pkl"
        if os.path.exists(buffer_path):
            with open(buffer_path, 'rb') as f:
                import pickle
                agent.replay_buffer = pickle.load(f)
    
    return agent

def train_agent(
    agent: Any,
    env: gym.Env,
    n_episodes: int = 1000,
    max_steps: int = 1000,
    eval_every: int = 100,
    n_eval_episodes: int = 10,
    save_path: Optional[str] = None,
    render: bool = False,
    verbose: int = 1
) -> Dict[str, List[float]]:
    """
    Train an agent on an environment.
    
    Args:
        agent: The agent to train
        env: The environment to train on
        n_episodes: Number of episodes to train for
        max_steps: Maximum number of steps per episode
        eval_every: Evaluate the agent every N episodes
        n_eval_episodes: Number of episodes to evaluate for
        save_path: Path to save the model (optional)
        render: Whether to render the environment during training
        verbose: Verbosity level (0 = no output, 1 = episode info, 2 = step info)
        
    Returns:
        Dictionary of training metrics
    """
    # Initialize metrics
    metrics = {
        'episode_rewards': [],
        'episode_lengths': [],
        'mean_rewards': [],
        'eval_rewards': [],
        'losses': []
    }
    
    # Training loop
    for episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        episode_reward = 0
        episode_loss = 0
        n_updates = 0
        
        for step in range(max_steps):
            # Get action from agent
            if hasattr(agent, 'get_action'):
                action = agent.get_action(state, training=True)
            elif hasattr(agent, 'predict'):
                action, _ = agent.predict(state, deterministic=False)
            else:
                raise ValueError("Agent must have 'get_action' or 'predict' method")
            
            # Take a step in the environment
            next_state, reward, done, _, _ = env.step(action)
            
            # Store experience in replay buffer
            if hasattr(agent, 'remember'):
                agent.remember(state, action, reward, next_state, done)
            
            # Update the agent
            if hasattr(agent, 'learn'):
                loss = agent.learn()
                if loss is not None:
                    episode_loss += loss
                    n_updates += 1
            
            episode_reward += reward
            state = next_state
            
            if verbose >= 2:
                print(f"Step {step + 1}/{max_steps}")
                print(f"Action: {action}")
                print(f"Reward: {reward:.4f}")
                print(f"Total Reward: {episode_reward:.2f}")
                if loss is not None:
                    print(f"Loss: {loss:.6f}")
                print("-" * 30)
            
            if render:
                env.render()
            
            if done:
                break
        
        # Update metrics
        metrics['episode_rewards'].append(episode_reward)
        metrics['episode_lengths'].append(step + 1)
        
        # Calculate mean reward over last 100 episodes
        mean_reward = np.mean(metrics['episode_rewards'][-100:])
        metrics['mean_rewards'].append(mean_reward)
        
        # Calculate average loss
        if n_updates > 0:
            avg_loss = episode_loss / n_updates
            metrics['losses'].append(avg_loss)
        
        # Print episode info
        if verbose >= 1:
            print(f"Episode {episode}/{n_episodes}")
            print(f"Steps: {step + 1}")
            print(f"Total Reward: {episode_reward:.2f}")
            print(f"Mean Reward (100 episodes): {mean_reward:.2f}")
            if n_updates > 0:
                print(f"Average Loss: {avg_loss:.6f}")
            print("=" * 50)
        
        # Evaluate the agent
        if episode % eval_every == 0 or episode == n_episodes:
            eval_metrics = evaluate_agent(agent, env, n_episodes=n_eval_episodes)
            metrics['eval_rewards'].append(eval_metrics['mean_reward'])
            
            if verbose >= 1:
                print(f"Evaluation after {episode} episodes:")
                print(f"Mean Reward: {eval_metrics['mean_reward']:.2f} ± {eval_metrics['std_reward']:.2f}")
                print(f"Min/Max Reward: {eval_metrics['min_reward']:.2f}/{eval_metrics['max_reward']:.2f}")
                print("=" * 50)
            
            # Save the model if it's the best so far
            if save_path and (len(metrics['eval_rewards']) == 1 or 
                             eval_metrics['mean_reward'] >= max(metrics['eval_rewards'])):
                save_model(agent, os.path.join(save_path, f"best_model"))
                if verbose >= 1:
                    print(f"Saved best model with reward {eval_metrics['mean_reward']:.2f}")
        
        # Save the model periodically
        if save_path and episode % (n_episodes // 10) == 0:
            save_model(agent, os.path.join(save_path, f"model_episode_{episode}"))
    
    return metrics
