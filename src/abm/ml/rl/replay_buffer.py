"""
Replay Buffer for Reinforcement Learning

This module implements experience replay buffers for off-policy RL algorithms.
"""

import numpy as np
import random
from typing import List, Tuple, Optional, Dict, Any, Union
import tensorflow as tf

class ReplayBuffer:
    """
    A simple experience replay buffer for off-policy RL algorithms.
    """
    
    def __init__(self, capacity: int, batch_size: int, seed: Optional[int] = None):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Number of transitions to sample in a batch
            seed: Random seed for reproducibility
        """
        self.capacity = capacity
        self.batch_size = batch_size
        self.buffer = []
        self.position = 0
        
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated
        """
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions from the buffer.
        
        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32)
        )
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        return len(self.buffer)

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.
    
    Implements proportional prioritization with importance sampling.
    """
    
    def __init__(
        self, 
        capacity: int, 
        batch_size: int, 
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        seed: Optional[int] = None
    ):
        """
        Initialize the prioritized replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            batch_size: Number of transitions to sample in a batch
            alpha: Controls the amount of prioritization (0 = uniform sampling)
            beta: Controls the amount of importance sampling correction
            beta_increment: Rate at which beta approaches 1.0
            epsilon: Small constant to ensure all transitions have non-zero probability
            seed: Random seed for reproducibility
        """
        super().__init__(capacity, batch_size, seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        # Use a sum tree for efficient priority updates and sampling
        self.priorities = np.zeros((2 * capacity,), dtype=np.float32)
        self.max_priority = 1.0
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, next_state: np.ndarray, done: bool) -> None:
        """
        Add a transition to the buffer with maximum priority.
        """
        idx = self.position
        super().add(state, action, reward, next_state, done)
        
        # Set initial priority to maximum
        self._set_priority(idx, self.max_priority ** self.alpha)
    
    def _set_priority(self, idx: int, priority: float) -> None:
        """Set the priority of a transition."""
        # Update the priority in the sum tree
        idx += self.capacity  # Shift index to leaf nodes
        self.priorities[idx] = priority
        
        # Propagate the change through the tree
        while idx > 1:
            idx //= 2
            self.priorities[idx] = self.priorities[2 * idx] + self.priorities[2 * idx + 1]
    
    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportional prioritization."""
        indices = []
        total_priority = self.priorities[1]  # Root node contains sum of all priorities
        segment = total_priority / self.batch_size
        
        for i in range(self.batch_size):
            mass = random.random() * segment + i * segment
            idx = self._retrieve(1, mass)  # Start from root
            indices.append(idx - self.capacity)  # Convert to buffer index
        
        return indices
    
    def _retrieve(self, idx: int, mass: float) -> int:
        """Traverse the tree to find the leaf node corresponding to the given mass."""
        left = 2 * idx
        right = left + 1
        
        if left >= len(self.priorities):
            return idx
        
        if mass <= self.priorities[left]:
            return self._retrieve(left, mass)
        else:
            return self._retrieve(right, mass - self.priorities[left])
    
    def sample(self) -> Tuple[Tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions with importance sampling weights.
        
        Returns:
            Tuple of (transitions, indices, weights)
            where transitions is (states, actions, rewards, next_states, dones)
        """
        assert len(self) >= self.batch_size, "Not enough samples in the buffer"
        
        # Sample indices with probability proportional to priority^alpha
        indices = self._sample_proportional()
        
        # Get the corresponding transitions
        states, actions, rewards, next_states, dones = zip(
            *[self.buffer[i] for i in indices]
        )
        
        # Convert to numpy arrays
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.float32)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)
        dones = np.array(dones, dtype=np.float32)
        
        # Calculate importance sampling weights
        total_priority = self.priorities[1]  # Sum of all priorities
        probs = np.array([self.priorities[i + self.capacity] / total_priority 
                         for i in indices])
        
        # Importance sampling weights: (1/N * 1/p_i)^beta
        weights = (len(self) * probs) ** -self.beta
        weights /= np.max(weights)  # Normalize to [0, 1]
        
        # Update beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return (states, actions, rewards, next_states, dones), indices, weights
    
    def update_priorities(self, indices: List[int], priorities: np.ndarray) -> None:
        """
        Update the priorities of transitions.
        
        Args:
            indices: Indices of transitions to update
            priorities: New priorities
        """
        assert len(indices) == len(priorities)
        
        for idx, priority in zip(indices, priorities):
            # Add epsilon to ensure non-zero priority
            priority = (np.abs(priority) + self.epsilon) ** self.alpha
            self._set_priority(idx, priority)
            self.max_priority = max(self.max_priority, priority)

class NStepBuffer:
    """
    A buffer for n-step returns calculation.
    """
    
    def __init__(self, gamma: float, n_steps: int = 3):
        """
        Initialize the n-step buffer.
        
        Args:
            gamma: Discount factor
            n_steps: Number of steps to look ahead for n-step returns
        """
        self.gamma = gamma
        self.n_steps = n_steps
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def add(self, state: np.ndarray, action: np.ndarray, 
            reward: float, done: bool) -> None:
        """Add a transition to the buffer."""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def get_n_step_transitions(self) -> List[Tuple]:
        """
        Calculate n-step returns and return transitions.
        
        Returns:
            List of (state, action, n_step_return, n_step_state, n_step_done)
        """
        transitions = []
        
        for i in range(len(self.states) - self.n_steps + 1):
            # Calculate n-step return
            n_step_return = 0
            for j in range(self.n_steps):
                n_step_return += (self.gamma ** j) * self.rewards[i + j]
                if self.dones[i + j]:
                    break
            
            # Get n-step state and done flag
            n_step_state = self.states[i + self.n_steps - 1] if i + self.n_steps - 1 < len(self.states) else None
            n_step_done = any(self.dones[i:i+self.n_steps])
            
            transitions.append((
                self.states[i],
                self.actions[i],
                n_step_return,
                n_step_state,
                n_step_done
            ))
        
        return transitions
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.dones = []
    
    def __len__(self) -> int:
        """Return the number of transitions in the buffer."""
        return len(self.states)
