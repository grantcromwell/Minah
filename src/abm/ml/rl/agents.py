"""
Reinforcement Learning Agents for Trading

This module implements various RL algorithms for trading.
"""

import os
import random
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
import logging
from collections import deque, namedtuple
import pickle

from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)

tf.keras.backend.set_floatx('float32')

# Ensure TensorFlow is using GPU if available
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        logger.warning(f"Could not set GPU memory growth: {e}")

Transition = namedtuple('Transition', 
    ['state', 'action', 'reward', 'next_state', 'done'])

class BaseAgent:
    """Base class for all RL agents."""
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        memory_size: int = 100000,
        tau: float = 0.005,
        update_every: int = 1,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            gamma: Discount factor
            lr: Learning rate
            batch_size: Batch size for training
            memory_size: Size of the replay buffer
            tau: Soft update parameter
            update_every: Update the network every N steps
            seed: Random seed
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.tau = tau
        self.update_every = update_every
        self.seed = seed
        
        # Set random seeds
        self._set_seeds()
        
        # Initialize replay buffer
        self.memory = ReplayBuffer(memory_size, batch_size)
        
        # Initialize networks
        self.policy_net = self._build_network()
        self.target_net = self._build_network()
        self._update_target_network(tau=1.0)  # Hard update
        
        # Optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        # Training variables
        self.steps_done = 0
        self.episode = 0
        self.episode_reward = 0
        self.episode_loss = 0
        self.episode_steps = 0
        
        # Logging
        self.loss_history = []
        self.reward_history = []
        self.episode_lengths = []
    
    def _set_seeds(self) -> None:
        """Set random seeds for reproducibility."""
        random.seed(self.seed)
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        tf.keras.utils.set_random_seed(self.seed)
    
    def _build_network(self) -> tf.keras.Model:
        """Build the neural network model."""
        raise NotImplementedError("Subclasses must implement _build_network")
    
    def get_action(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        """Select an action using the policy."""
        raise NotImplementedError("Subclasses must implement get_action")
    
    def _update_target_network(self, tau: Optional[float] = None) -> None:
        """Update the target network weights."""
        if tau is None:
            tau = self.tau
            
        for target_param, param in zip(self.target_net.trainable_variables, 
                                     self.policy_net.trainable_variables):
            target_param.assign(tau * param + (1.0 - tau) * target_param)
    
    def remember(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool
    ) -> None:
        """Store experience in replay memory."""
        self.memory.add(state, action, reward, next_state, done)
    
    def learn(self) -> Optional[float]:
        """Update the model based on experience replay."""
        if len(self.memory) < self.batch_size:
            return None
            
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Compute loss and update weights
        with tf.GradientTape() as tape:
            loss = self._compute_loss(states, actions, rewards, next_states, dones)
        
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
        
        # Update target network
        if self.steps_done % self.update_every == 0:
            self._update_target_network()
        
        self.steps_done += 1
        
        return float(loss.numpy())
    
    def _compute_loss(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> tf.Tensor:
        """Compute the loss for a batch of transitions."""
        raise NotImplementedError("Subclasses must implement _compute_loss")
    
    def save(self, path: str) -> None:
        """Save the model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.policy_net.save_weights(path + "_policy.h5")
        self.target_net.save_weights(path + "_target.h5")
        
        # Save other parameters
        params = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'tau': self.tau,
            'update_every': self.update_every,
            'seed': self.seed,
            'steps_done': self.steps_done,
            'episode': self.episode,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'episode_lengths': self.episode_lengths
        }
        
        with open(path + "_params.pkl", 'wb') as f:
            pickle.dump(params, f)
    
    @classmethod
    def load(cls, path: str) -> 'BaseAgent':
        """Load a saved model from disk."""
        # Load parameters
        with open(path + "_params.pkl", 'rb') as f:
            params = pickle.load(f)
        
        # Create agent instance
        agent = cls(**params)
        
        # Load weights
        agent.policy_net.load_weights(path + "_policy.h5")
        agent.target_net.load_weights(path + "_target.h5")
        
        return agent

class DQNAgent(BaseAgent):
    """
    Deep Q-Network (DQN) agent with experience replay and target network.
    
    References:
        "Playing Atari with Deep Reinforcement Learning"
        Mnih et al., 2013
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        gamma: float = 0.99,
        lr: float = 1e-4,
        batch_size: int = 64,
        memory_size: int = 100000,
        tau: float = 0.005,
        update_every: int = 1,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of discrete actions
            gamma: Discount factor
            lr: Learning rate
            batch_size: Batch size for training
            memory_size: Size of the replay buffer
            tau: Soft update parameter for target network
            update_every: Update the target network every N steps
            epsilon_start: Starting value of epsilon (exploration rate)
            epsilon_end: Minimum value of epsilon
            epsilon_decay: Decay rate for epsilon
            seed: Random seed
        """
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            memory_size=memory_size,
            tau=tau,
            update_every=update_every,
            seed=seed,
            **kwargs
        )
    
    def _build_network(self) -> tf.keras.Model:
        """Build the Q-network."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(self.action_dim, activation='linear')
        ])
        return model
    
    def get_action(self, state: np.ndarray, training: bool = False) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether the agent is in training mode
            
        Returns:
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        q_values = self.policy_net(state, training=False)
        return int(tf.argmax(q_values[0]).numpy())
    
    def _compute_loss(
        self,
        states: tf.Tensor,
        actions: tf.Tensor,
        rewards: tf.Tensor,
        next_states: tf.Tensor,
        dones: tf.Tensor
    ) -> tf.Tensor:
        """
        Compute the DQN loss (Huber loss).
        
        L = E[(r + γ * max_a' Q_target(s', a') - Q(s, a))^2]
        """
        # Convert actions to indices (for one-hot encoding)
        actions = tf.cast(actions, tf.int32)
        actions_one_hot = tf.one_hot(actions, self.action_dim, dtype=tf.float32)
        
        # Current Q-values for the chosen actions
        current_q = tf.reduce_sum(
            self.policy_net(states, training=True) * actions_one_hot,
            axis=1
        )
        
        # Compute the target Q-values
        next_q_values = self.target_net(next_states, training=False)
        max_next_q = tf.reduce_max(next_q_values, axis=1)
        
        # Set target to r if done, otherwise r + γ * max Q(s', a')
        target_q = rewards + self.gamma * max_next_q * (1.0 - dones)
        
        # Compute Huber loss
        loss = tf.keras.losses.Huber()(target_q, current_q)
        
        return loss
    
    def learn(self) -> Optional[float]:
        """Update the model based on experience replay."""
        loss = super().learn()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon = max(
                self.epsilon_end,
                self.epsilon * self.epsilon_decay
            )
        
        return loss

class PPOAgent(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    
    References:
        "Proximal Policy Optimization Algorithms"
        Schulman et al., 2017
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_std: float = 0.5,
        gamma: float = 0.99,
        lr: float = 3e-4,
        batch_size: int = 64,
        memory_size: int = 100000,
        tau: float = 0.005,
        update_every: int = 1,
        clip_ratio: float = 0.2,
        target_kl: float = 0.01,
        train_iters: int = 80,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the PPO agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            action_std: Standard deviation for action noise
            gamma: Discount factor
            lr: Learning rate
            batch_size: Batch size for training
            memory_size: Size of the replay buffer
            tau: Not used in PPO (kept for compatibility)
            update_every: Not used in PPO (kept for compatibility)
            clip_ratio: Clip ratio for PPO objective
            target_kl: Target KL divergence for early stopping
            train_iters: Number of training iterations per update
            seed: Random seed
        """
        self.action_std = action_std
        self.clip_ratio = clip_ratio
        self.target_kl = target_kl
        self.train_iters = train_iters
        
        # PPO uses a different memory structure
        self.memory = []
        self.buffer = {
            'states': [],
            'actions': [],
            'rewards': [],
            'values': [],
            'log_probs': [],
            'dones': []
        }
        
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            memory_size=memory_size,
            tau=tau,
            update_every=update_every,
            seed=seed,
            **kwargs
        )
        
        # Optimizer for both actor and critic
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def _build_network(self) -> tf.keras.Model:
        """Build the actor-critic network."""
        # Shared feature extractor
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(128, activation='tanh')(inputs)
        x = tf.keras.layers.Dense(128, activation='tanh')(x)
        
        # Actor head (policy)
        actor_mean = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        actor_log_std = tf.Variable(
            initial_value=np.log(self.action_std) * tf.ones(self.action_dim, dtype=tf.float32),
            trainable=True
        )
        
        # Critic head (value function)
        critic = tf.keras.layers.Dense(1)(x)
        
        # Create model
        model = tf.keras.Model(inputs=inputs, outputs=[actor_mean, actor_log_std, critic])
        return model
    
    def _get_distribution(self, states: tf.Tensor) -> tfp.distributions.Distribution:
        """Get the action distribution for given states."""
        means, log_stds, _ = self.policy_net(states, training=False)
        stds = tf.exp(log_stds)
        return tfp.distributions.Normal(means, stds)
    
    def get_action(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        """
        Sample an action from the policy.
        
        Args:
            state: Current state
            training: Whether the agent is in training mode
            
        Returns:
            Selected action and log probability
        """
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        means, log_stds, _ = self.policy_net(state, training=False)
        stds = tf.exp(log_stds)
        
        dist = tfp.distributions.Normal(means, stds)
        
        if training:
            action = dist.sample()
        else:
            action = means
        
        # Clip and scale action to [-1, 1] if needed
        action = tf.clip_by_value(action, -1.0, 1.0)
        
        # Calculate log probability
        log_prob = dist.log_prob(action)
        
        return action.numpy()[0], log_prob.numpy()[0]
    
    def remember(
        self, 
        state: np.ndarray, 
        action: np.ndarray, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        value: float = 0.0,
        log_prob: float = 0.0
    ) -> None:
        """Store experience in the buffer."""
        self.buffer['states'].append(state)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)
    
    def _compute_advantages(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        next_value: float = 0.0,
        gamma: float = 0.99,
        lam: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages and returns using GAE."""
        advantages = np.zeros_like(rewards)
        last_gae = 0.0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_non_terminal = 1.0 - done
                next_value = next_value
            else:
                next_non_terminal = 1.0 - dones[t+1]
                next_value = values[t+1]
            
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + gamma * lam * next_non_terminal * last_gae
            advantages[t] = last_gae
        
        returns = advantages + values
        return advantages, returns
    
    def learn(self) -> Optional[float]:
        """Update the model using PPO."""
        if len(self.buffer['states']) < self.batch_size:
            return None
        
        # Convert buffer to numpy arrays
        states = np.array(self.buffer['states'])
        actions = np.array(self.buffer['actions'])
        rewards = np.array(self.buffer['rewards'])
        old_values = np.array(self.buffer['values'])
        old_log_probs = np.array(self.buffer['log_probs'])
        dones = np.array(self.buffer['dones'])
        
        # Compute advantages and returns
        advantages, returns = self._compute_advantages(rewards, old_values, dones)
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        
        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        old_log_probs = tf.convert_to_tensor(old_log_probs, dtype=tf.float32)
        advantages = tf.convert_to_tensor(advantages, dtype=tf.float32)
        returns = tf.convert_to_tensor(returns, dtype=tf.float32)
        
        # Train for multiple epochs
        for _ in range(self.train_iters):
            # Sample mini-batches
            indices = np.random.permutation(len(states))
            
            for start in range(0, len(states), self.batch_size):
                batch_indices = indices[start:start + self.batch_size]
                
                with tf.GradientTape() as tape:
                    # Get current policy and value predictions
                    means, log_stds, values = self.policy_net(tf.gather(states, batch_indices), training=True)
                    stds = tf.exp(log_stds)
                    dist = tfp.distributions.Normal(means, stds)
                    
                    # Calculate new log probabilities
                    new_log_probs = dist.log_prob(tf.gather(actions, batch_indices))
                    new_log_probs = tf.reduce_sum(new_log_probs, axis=-1)
                    
                    # Calculate ratio (pi_theta / pi_theta_old)
                    ratio = tf.exp(new_log_probs - tf.gather(old_log_probs, batch_indices))
                    
                    # Calculate surrogate loss
                    surr1 = ratio * tf.gather(advantages, batch_indices)
                    surr2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * tf.gather(advantages, batch_indices)
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2))
                    
                    # Calculate value loss
                    value_loss = tf.reduce_mean(tf.square(tf.gather(returns, batch_indices) - tf.squeeze(values)))
                    
                    # Entropy bonus
                    entropy = tf.reduce_mean(dist.entropy())
                    
                    # Total loss
                    loss = actor_loss + 0.5 * value_loss - 0.01 * entropy
                
                # Compute gradients and update
                grads = tape.gradient(loss, self.policy_net.trainable_variables)
                self.optimizer.apply_gradients(zip(grads, self.policy_net.trainable_variables))
                
                # Early stopping with KL divergence
                with tf.GradientTape() as tape_kl:
                    means, log_stds, _ = self.policy_net(tf.gather(states, batch_indices), training=True)
                    stds = tf.exp(log_stds)
                    dist = tfp.distributions.Normal(means, stds)
                    new_log_probs = dist.log_prob(tf.gather(actions, batch_indices))
                    new_log_probs = tf.reduce_sum(new_log_probs, axis=-1)
                
                kl = tf.reduce_mean(tf.gather(old_log_probs, batch_indices) - new_log_probs)
                if kl > 1.5 * self.target_kl:
                    break
        
        # Clear buffer
        self.buffer = {k: [] for k in self.buffer}
        
        return float(loss.numpy())

class SACAgent(BaseAgent):
    """
    Soft Actor-Critic (SAC) agent.
    
    References:
        "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning"
        Haarnoja et al., 2018
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_high: float = 1.0,
        action_low: float = -1.0,
        gamma: float = 0.99,
        lr: float = 3e-4,
        batch_size: int = 256,
        memory_size: int = 1000000,
        tau: float = 0.005,
        alpha: float = 0.2,
        autotune_alpha: bool = True,
        target_entropy: Optional[float] = None,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the SAC agent.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Dimension of the action space
            action_high: Upper bound of the action space
            action_low: Lower bound of the action space
            gamma: Discount factor
            lr: Learning rate
            batch_size: Batch size for training
            memory_size: Size of the replay buffer
            tau: Soft update parameter for target networks
            alpha: Temperature parameter (entropy weight)
            autotune_alpha: Whether to automatically tune alpha
            target_entropy: Target entropy for automatic alpha tuning
            seed: Random seed
        """
        self.action_high = action_high
        self.action_low = action_low
        self.alpha = alpha
        self.autotune_alpha = autotune_alpha
        self.target_entropy = target_entropy or -np.prod(action_dim).astype(np.float32)
        
        # Initialize replay buffer with a larger size
        self.memory = ReplayBuffer(memory_size, batch_size)
        
        # Call parent initializer with updated parameters
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            gamma=gamma,
            lr=lr,
            batch_size=batch_size,
            memory_size=memory_size,
            tau=tau,
            update_every=1,  # SAC updates every step
            seed=seed,
            **kwargs
        )
        
        # Initialize alpha
        if self.autotune_alpha:
            self.log_alpha = tf.Variable(tf.math.log(alpha), dtype=tf.float32)
            self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    
    def _build_network(self) -> tf.keras.Model:
        """Build the actor and critic networks."""
        # Actor (policy) network
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(256, activation='relu')(inputs)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        
        # Output mean and log_std for each action dimension
        means = tf.keras.layers.Dense(self.action_dim, activation='tanh')(x)
        log_stds = tf.keras.layers.Dense(self.action_dim)(x)
        
        # Scale means to action space
        means = (self.action_high - self.action_low) * 0.5 * means + \
                (self.action_high + self.action_low) * 0.5
        
        # Clip log_std for numerical stability
        log_stds = tf.clip_by_value(log_stds, -20, 2)
        
        # Create model
        actor = tf.keras.Model(inputs=inputs, outputs=[means, log_stds])
        
        # Critic networks (Q-functions)
        state_input = tf.keras.layers.Input(shape=(self.state_dim,))
        action_input = tf.keras.layers.Input(shape=(self.action_dim,))
        
        # First Q-network
        x = tf.keras.layers.Concatenate()([state_input, action_input])
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        q1_output = tf.keras.layers.Dense(1)(x)
        
        # Second Q-network
        x = tf.keras.layers.Concatenate()([state_input, action_input])
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        q2_output = tf.keras.layers.Dense(1)(x)
        
        # Create critic model
        critic = tf.keras.Model(
            inputs=[state_input, action_input],
            outputs=[q1_output, q2_output]
        )
        
        # Value network (for automatic alpha tuning)
        x = tf.keras.layers.Dense(256, activation='relu')(state_input)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        value_output = tf.keras.layers.Dense(1)(x)
        value_net = tf.keras.Model(inputs=state_input, outputs=value_output)
        
        return {
            'actor': actor,
            'critic': critic,
            'target_critic': tf.keras.models.clone_model(critic),
            'value': value_net,
            'target_value': tf.keras.models.clone_model(value_net)
        }
    
    def _sample_action(self, state: np.ndarray, training: bool = True) -> Tuple[np.ndarray, tf.Tensor]:
        """Sample an action from the policy."""
        state = tf.convert_to_tensor([state], dtype=tf.float32)
        means, log_stds = self.policy_net['actor'](state, training=training)
        
        if training:
            # Reparameterization trick
            stds = tf.exp(log_stds)
            noise = tf.random.normal(shape=means.shape)
            actions = means + noise * stds
        else:
            actions = means
        
        # Clip actions to valid range
        actions = tf.clip_by_value(actions, self.action_low, self.action_high)
        
        # Calculate log probability
        log_stds = tf.clip_by_value(log_stds, -20, 2)
        stds = tf.exp(log_stds)
        dist = tfp.distributions.Normal(means, stds)
        log_probs = dist.log_prob(actions)
        
        # Sum over action dimensions
        log_probs = tf.reduce_sum(log_probs, axis=1, keepdims=True)
        
        return actions[0].numpy(), log_probs[0].numpy()
    
    def get_action(self, state: np.ndarray, training: bool = False) -> np.ndarray:
        """Get an action from the policy."""
        action, _ = self._sample_action(state, training=training)
        return action
    
    def _update_networks(self, states: tf.Tensor, actions: tf.Tensor, 
                        rewards: tf.Tensor, next_states: tf.Tensor, 
                        dones: tf.Tensor) -> Dict[str, float]:
        """Update the SAC networks."""
        # Convert to float32
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.float32)
        
        # Update alpha if autotuning is enabled
        if self.autotune_alpha:
            with tf.GradientTape() as tape:
                # Sample actions and log probs
                _, log_probs = self._sample_action(states)
                
                # Calculate alpha loss
                alpha_loss = -tf.reduce_mean(
                    self.log_alpha * tf.stop_gradient(log_probs + self.target_entropy)
                )
            
            # Update alpha
            alpha_grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(alpha_grads, [self.log_alpha]))
            
            # Update alpha value
            self.alpha = tf.exp(self.log_alpha).numpy()
        
        # Update critic (Q-functions)
        with tf.GradientTape(persistent=True) as tape:
            # Sample actions and log probs for next states
            next_actions, next_log_probs = self._sample_action(next_states)
            
            # Target Q-values
            target_q1, target_q2 = self.policy_net['target_critic'](
                [next_states, next_actions], training=False
            )
            target_q = tf.minimum(target_q1, target_q2)
            
            # Target value = reward + γ * (min(Q1, Q2) - α * log π(a|s))
            target_value = rewards + self.gamma * (1.0 - dones) * \
                (target_q - self.alpha * next_log_probs)
            
            # Current Q-values
            current_q1, current_q2 = self.policy_net['critic'](
                [states, actions], training=True
            )
            
            # Critic loss (MSE)
            critic1_loss = tf.reduce_mean(tf.square(current_q1 - target_value))
            critic2_loss = tf.reduce_mean(tf.square(current_q2 - target_value))
            critic_loss = critic1_loss + critic2_loss
        
        # Update critic
        critic_vars = self.policy_net['critic'].trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_vars)
        self.optimizer.apply_gradients(zip(critic_grads, critic_vars))
        
        # Update actor (policy)
        with tf.GradientTape() as tape:
            # Sample actions and log probs
            new_actions, log_probs = self._sample_action(states)
            
            # Q-values for the new actions
            q1, q2 = self.policy_net['critic'](
                [states, new_actions], training=False
            )
            q = tf.minimum(q1, q2)
            
            # Actor loss = (α * log π(a|s) - Q(s,a)).mean()
            actor_loss = tf.reduce_mean(self.alpha * log_probs - q)
        
        # Update actor
        actor_vars = self.policy_net['actor'].trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_vars)
        self.optimizer.apply_gradients(zip(actor_grads, actor_vars))
        
        # Update target networks
        self._update_target_network()
        
        return {
            'critic1_loss': float(critic1_loss.numpy()),
            'critic2_loss': float(critic2_loss.numpy()),
            'actor_loss': float(actor_loss.numpy()),
            'alpha': float(self.alpha)
        }
    
    def _update_target_network(self, tau: Optional[float] = None) -> None:
        """Update the target networks using soft updates."""
        if tau is None:
            tau = self.tau
        
        # Update target critic
        for target_param, param in zip(
            self.policy_net['target_critic'].trainable_variables,
            self.policy_net['critic'].trainable_variables
        ):
            target_param.assign(tau * param + (1.0 - tau) * target_param)
        
        # Update target value network (if exists)
        if 'target_value' in self.policy_net:
            for target_param, param in zip(
                self.policy_net['target_value'].trainable_variables,
                self.policy_net['value'].trainable_variables
            ):
                target_param.assign(tau * param + (1.0 - tau) * target_param)
    
    def learn(self) -> Optional[Dict[str, float]]:
        """Update the model based on experience replay."""
        if len(self.memory) < self.batch_size:
            return None
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample()
        
        # Update networks
        metrics = self._update_networks(states, actions, rewards, next_states, dones)
        
        return metrics
    
    def save(self, path: str) -> None:
        """Save the model weights to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save actor and critic networks
        self.policy_net['actor'].save_weights(path + "_actor.h5")
        self.policy_net['critic'].save_weights(path + "_critic.h5")
        self.policy_net['target_critic'].save_weights(path + "_target_critic.h5")
        
        if 'value' in self.policy_net:
            self.policy_net['value'].save_weights(path + "_value.h5")
            self.policy_net['target_value'].save_weights(path + "_target_value.h5")
        
        # Save other parameters
        params = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'action_high': self.action_high,
            'action_low': self.action_low,
            'gamma': self.gamma,
            'lr': self.lr,
            'batch_size': self.batch_size,
            'memory_size': self.memory_size,
            'tau': self.tau,
            'alpha': self.alpha,
            'autotune_alpha': self.autotune_alpha,
            'target_entropy': self.target_entropy,
            'seed': self.seed,
            'steps_done': self.steps_done,
            'episode': self.episode,
            'loss_history': self.loss_history,
            'reward_history': self.reward_history,
            'episode_lengths': self.episode_lengths
        }
        
        with open(path + "_params.pkl", 'wb') as f:
            pickle.dump(params, f)
    
    @classmethod
    def load(cls, path: str) -> 'SACAgent':
        """Load a saved model from disk."""
        # Load parameters
        with open(path + "_params.pkl", 'rb') as f:
            params = pickle.load(f)
        
        # Create agent instance
        agent = cls(**params)
        
        # Load weights
        agent.policy_net['actor'].load_weights(path + "_actor.h5")
        agent.policy_net['critic'].load_weights(path + "_critic.h5")
        agent.policy_net['target_critic'].load_weights(path + "_target_critic.h5")
        
        if os.path.exists(path + "_value.h5") and 'value' in agent.policy_net:
            agent.policy_net['value'].load_weights(path + "_value.h5")
            agent.policy_net['target_value'].load_weights(path + "_target_value.h5")
        
        return agent

def create_agent(
    agent_type: str,
    state_dim: int,
    action_dim: int,
    **kwargs
) -> BaseAgent:
    """
    Factory function to create an RL agent.
    
    Args:
        agent_type: Type of agent ('dqn', 'ppo', or 'sac')
        state_dim: Dimension of the state space
        action_dim: Dimension of the action space
        **kwargs: Additional arguments for the agent
        
    Returns:
        An instance of the specified agent
    """
    if agent_type.lower() == 'dqn':
        return DQNAgent(state_dim, action_dim, **kwargs)
    elif agent_type.lower() == 'ppo':
        return PPOAgent(state_dim, action_dim, **kwargs)
    elif agent_type.lower() == 'sac':
        return SACAgent(state_dim, action_dim, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
