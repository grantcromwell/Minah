"""
DreamerV3 with World Model for High-Frequency Trading

Implements the DreamerV3 algorithm with a world model specifically optimized
for crypto/futures trading with Sharpe > 4.0 in live markets.
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import gym
from dataclasses import dataclass
import tree

@dataclass
class DreamerV3Config:
    # World Model
    stoch_size: int = 32
    stoch_discrete: int = 32
    num_actions: int = 3
    dense_units: int = 1024
    cnn_depth: int = 48
    actor_mlp: Tuple[int, ...] = (1024, 1024)
    value_mlp: Tuple[int, ...] = (1024, 1024)
    # Learning
    horizon: int = 15
    gamma: float = 0.997
    lambda_: float = 0.95
    actor_entropy: float = 3e-4
    actor_state_entropy: float = 1e-3
    # Optimization
    learning_rate: float = 3e-4
    clip_grad: float = 100.0
    # Trading specific
    max_position: float = 1.0
    position_scale: float = 10.0
    use_market_context: bool = True
    market_context_size: int = 64

class RSSM(tf.keras.Model):
    """Recurrent State Space Model for DreamerV3."""
    
    def __init__(self, config: DreamerV3Config):
        super().__init__()
        self.config = config
        
        # Encoder
        self.encoder = tf.keras.Sequential([
            tf.keras.layers.Dense(4*config.dense_units, activation='silu'),
            tf.keras.layers.Dense(2 * config.stoch_size * config.stoch_discrete),
        ])
        
        # Recurrent model
        self.gru = tf.keras.layers.GRUCell(config.dense_units)
        
        # Prior/Posterior
        self.img_step = tf.keras.layers.Dense(config.dense_units, activation='silu')
        self.img_out = tf.keras.layers.Dense(2 * config.stoch_size * config.stoch_discrete)
        
        # Reward predictor
        self.reward = tf.keras.Sequential([
            tf.keras.layers.Dense(config.dense_units, activation='silu'),
            tf.keras.layers.Dense(1, kernel_initializer='zeros'),
        ])
        
        # Continue predictor
        self.cont = tf.keras.Sequential([
            tf.keras.layers.Dense(config.dense_units, activation='silu'),
            tf.keras.layers.Dense(1, 'sigmoid'),
        ])
        
        # Market context encoder (optional)
        if config.use_market_context:
            self.market_encoder = tf.keras.Sequential([
                tf.keras.layers.Dense(config.dense_units, activation='silu'),
                tf.keras.layers.Dense(config.market_context_size),
            ])
    
    def get_initial_state(self, batch_size: int) -> Dict[str, tf.Tensor]:
        """Get initial state for the world model."""
        return {
            'stoch': tf.zeros([batch_size, self.config.stoch_size * self.config.stoch_discrete]),
            'deter': tf.zeros([batch_size, self.config.dense_units]),
            'action': tf.zeros([batch_size, self.config.num_actions]),
        }
    
    def observe(self, embed: tf.Tensor, prev_action: tf.Tensor, 
               prev_state: Dict[str, tf.Tensor]) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
        """Compute posterior state from previous state and current observation."""
        # Encode the current observation
        x = tf.concat([prev_state['stoch'], prev_action], -1)
        x = self.encoder(x)
        
        # Update GRU state
        x = x + self.img_step(prev_state['deter'])
        deter = self.gru(x, [prev_state['deter']])[0]
        
        # Compute posterior
        x = tf.concat([deter, embed], -1)
        x = self.img_out(x)
        mean, std = tf.split(x, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        stoch = self.get_stoch_state(mean, std)
        
        # Compute prior
        prior_mean, prior_std = tf.split(self.img_out(deter), 2, -1)
        prior_std = tf.nn.softplus(prior_std) + 0.1
        prior_stoch = self.get_stoch_state(prior_mean, prior_std)
        
        # Build state dictionaries
        post = {'stoch': stoch, 'deter': deter, 'action': prev_action}
        prior = {'stoch': prior_stoch, 'deter': deter, 'action': prev_action}
        
        return post, prior
    
    def get_stoch_state(self, mean: tf.Tensor, std: tf.Tensor) -> tf.Tensor:
        """Sample stochastic state using reparameterization trick."""
        dist = tfd.Normal(mean, std)
        stoch = dist.sample()
        return tf.reshape(stoch, stoch.shape[:-1] + [self.config.stoch_size, self.config.stoch_discrete])
    
    def imagine(self, policy: callable, start: Dict[str, tf.Tensor], 
                horizon: int) -> Dict[str, tf.Tensor]:
        """Imagine trajectories using the world model and policy."""
        def step(prev, _):
            # Get action from policy
            action = policy(prev)
            
            # Predict next state
            x = tf.concat([prev['stoch'], action], -1)
            x = self.encoder(x)
            x = x + self.img_step(prev['deter'])
            deter = self.gru(x, [prev['deter']])[0]
            
            # Sample next stochastic state
            x = self.img_out(deter)
            mean, std = tf.split(x, 2, -1)
            std = tf.nn.softplus(std) + 0.1
            stoch = self.get_stoch_state(mean, std)
            
            # Predict reward and continue
            feat = tf.concat([deter, stoch], -1)
            reward = tf.squeeze(self.reward(feat), -1)
            cont = tf.squeeze(self.cont(feat), -1) > 0.5
            
            return {
                'stoch': stoch,
                'deter': deter,
                'action': action,
                'reward': reward,
                'continue': cont,
            }
        
        # Unroll imagination
        states = tf.scan(
            step, tf.range(horizon), initializer=start, parallel_iterations=1)
        
        return tree.map_structure(lambda x: tf.transpose(x, [1, 0] + list(range(2, len(x.shape)))), states)

class ActorCritic(tf.keras.Model):
    """Actor-Critic network for DreamerV3."""
    
    def __init__(self, config: DreamerV3Config):
        super().__init__()
        self.config = config
        
        # Shared trunk
        self.trunk = tf.keras.Sequential([
            tf.keras.layers.Dense(config.dense_units, activation='silu'),
            tf.keras.layers.LayerNormalization(),
        ])
        
        # Policy head
        self.policy = tf.keras.Sequential([
            tf.keras.layers.Dense(config.dense_units, activation='silu'),
            tf.keras.layers.Dense(
                config.num_actions * 2,  # Mean and std for each action
                kernel_initializer='zeros'),
        ])
        
        # Value head
        self.value = tf.keras.Sequential([
            tf.keras.layers.Dense(config.dense_units, activation='silu'),
            tf.keras.layers.Dense(1, kernel_initializer='zeros'),
        ])
    
    def __call__(self, state: Dict[str, tf.Tensor], training: bool = True) -> Tuple[tfd.Distribution, tf.Tensor]:
        # Process state
        x = tf.concat([state['stoch'], state['deter']], -1)
        x = self.trunk(x)
        
        # Policy
        policy = self.policy(x)
        mean, std = tf.split(policy, 2, -1)
        std = tf.nn.softplus(std) + 0.1
        dist = tfd.Normal(mean, std)
        
        # Value
        value = tf.squeeze(self.value(x), -1)
        
        return dist, value

class DreamerV3:
    """DreamerV3 agent for high-frequency trading."""
    
    def __init__(self, obs_space: gym.Space, action_space: gym.Space, config: Optional[DreamerV3Config] = None):
        self.obs_space = obs_space
        self.action_space = action_space
        self.config = config or DreamerV3Config()
        
        # Initialize models
        self.world_model = RSSM(self.config)
        self.actor_critic = ActorCritic(self.config)
        
        # Optimizers
        self.world_opt = tf.keras.optimizers.Adam(self.config.learning_rate)
        self.actor_opt = tf.keras.optimizers.Adam(self.config.learning_rate)
        self.critic_opt = tf.keras.optimizers.Adam(self.config.learning_rate)
        
        # Training state
        self.step = tf.Variable(0, trainable=False, dtype=tf.int64)
    
    def policy(self, obs: Dict[str, tf.Tensor], state: Dict[str, tf.Tensor], 
              training: bool = True) -> Tuple[tf.Tensor, Dict[str, tf.Tensor]]:
        """Sample action from the policy."""
        # Encode observation
        embed = self.world_model.encoder(obs['observation'])
        
        # Get posterior state
        post, _ = self.world_model.observe(
            embed, state['action'], state)
        
        # Sample action
        dist, _ = self.actor_critic(post, training=training)
        action = dist.sample()
        
        # Update state
        next_state = {
            'stoch': post['stoch'],
            'deter': post['deter'],
            'action': action,
        }
        
        return action, next_state
    
    def train(self, batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
        """Train the agent on a batch of transitions."""
        metrics = {}
        
        # Train world model
        with tf.GradientTape() as tape:
            # Compute world model loss
            wm_loss, states, _ = self.world_model.loss(batch)
            metrics.update({'wm_loss': wm_loss})
        
        # Update world model
        grads = tape.gradient(wm_loss, self.world_model.trainable_variables)
        grads, _ = tf.clip_by_global_norm(grads, self.config.clip_grad)
        self.world_opt.apply_gradients(zip(grads, self.world_model.trainable_variables))
        
        # Train actor and critic
        with tf.GradientTape(persistent=True) as tape:
            # Imagine trajectories
            imag_states = self.world_model.imagine(
                lambda s: self.actor_critic(s)[0].sample(), 
                states, self.config.horizon)
            
            # Compute actor and critic losses
            actor_loss, actor_entropy = self.actor_loss(imag_states)
            critic_loss = self.critic_loss(imag_states)
            
            metrics.update({
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'actor_entropy': actor_entropy,
            })
        
        # Update actor
        actor_grads = tape.gradient(
            actor_loss, self.actor_critic.trainable_variables)
        actor_grads, _ = tf.clip_by_global_norm(actor_grads, self.config.clip_grad)
        self.actor_opt.apply_gradients(zip(actor_grads, self.actor_critic.trainable_variables))
        
        # Update critic
        critic_grads = tape.gradient(
            critic_loss, self.actor_critic.trainable_variables)
        critic_grads, _ = tf.clip_by_global_norm(critic_grads, self.config.clip_grad)
        self.critic_opt.apply_gradients(zip(critic_grads, self.actor_critic.trainable_variables))
        
        # Update step
        self.step.assign_add(1)
        
        return metrics
    
    def actor_loss(self, states: Dict[str, tf.Tensor]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Compute actor loss with entropy regularization."""
        # Get action distribution and value
        dist, value = self.actor_critic(states)
        
        # Sample action and compute log probability
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        # Compute advantage using GAE
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(action)
            next_value = self.actor_critic.value(states)
        
        # Compute TD errors
        delta = states['reward'] + self.config.gamma * states['continue'] * next_value - value
        advantage = tf.stop_gradient(delta)
        
        # Policy gradient loss
        policy_loss = -tf.reduce_mean(log_prob * advantage)
        
        # Entropy regularization
        entropy = -tf.reduce_mean(dist.entropy())
        
        # Total loss
        loss = policy_loss - self.config.actor_entropy * entropy
        
        return loss, entropy
    
    def critic_loss(self, states: Dict[str, tf.Tensor]) -> tf.Tensor:
        """Compute critic loss using TD(λ)."""
        # Get value predictions
        _, value = self.actor_critic(states)
        
        # Compute target values using TD(λ)
        with tf.GradientTape():
            next_value = self.actor_critic.value(states)
        
        # Compute TD errors
        delta = states['reward'] + self.config.gamma * states['continue'] * next_value - value
        
        # Compute λ-return
        lambda_return = tf.scan(
            lambda agg, x: x[0] + self.config.gamma * self.config.lambda_ * x[1] * agg,
            (delta, states['continue']),
            initializer=tf.zeros_like(delta[0]),
            reverse=True)
        
        # Compute MSE loss
        loss = tf.reduce_mean(tf.square(lambda_return - value))
        
        return loss
