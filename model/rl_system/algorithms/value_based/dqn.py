"""
Deep Q-Network (DQN) Agent

Implementation of the DQN algorithm for value-based reinforcement learning.
"""

import numpy as np
from typing import Dict, List, Tuple, Union
from collections import deque
import random

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from model.rl_system.base_agent import BaseAgent


class DQNNetwork(nn.Module):
    """Neural network for DQN."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ReplayMemory:
    """Experience replay memory for DQN."""
    
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Store a transition in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class DQNAgent(BaseAgent):
    """DQN Agent implementation."""
    
    def __init__(self, observation_space, action_space, config: Dict):
        super().__init__(observation_space, action_space, config)
        
        # Get dimensions
        if hasattr(observation_space, 'shape'):
            self.state_size = np.prod(observation_space.shape)
        else:
            self.state_size = getattr(observation_space, 'n', config.get('observation_space_size', 4))
        
        if hasattr(action_space, 'n'):
            self.action_size = action_space.n
        else:
            self.action_size = config.get('action_space_size', 2)
        
        # DQN specific parameters
        self.memory_size = config.get('memory_size', 10000)
        self.batch_size = config.get('batch_size', 32)
        self.target_update_frequency = config.get('target_update_frequency', 1000)
        self.exploration_rate = config.get('exploration_rate', 1.0)
        self.exploration_decay = config.get('exploration_decay', 0.995)
        self.exploration_min = config.get('exploration_min', 0.01)
        
        # Initialize memory
        self.memory = ReplayMemory(self.memory_size)
        
        if TORCH_AVAILABLE:
            # Initialize neural networks
            self.device = torch.device(self.device if torch.cuda.is_available() else 'cpu')
            self.q_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.target_network = DQNNetwork(self.state_size, self.action_size).to(self.device)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
            
            # Copy weights to target network
            self.update_target_network()
        else:
            # Fallback to simple table-based Q-learning
            self.q_table = np.zeros((100, self.action_size))  # Simple fixed-size table
            print("⚠️  PyTorch not available, using simple table-based Q-learning")
    
    def act(self, observation: np.ndarray) -> int:
        """Select an action using epsilon-greedy policy."""
        # Exploration vs exploitation
        if self.training and np.random.random() < self.exploration_rate:
            return np.random.randint(0, self.action_size)
        
        if TORCH_AVAILABLE and hasattr(self, 'q_network'):
            # Neural network-based action selection
            state = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state)
                return q_values.argmax().item()
        else:
            # Table-based action selection
            state_index = min(int(np.sum(observation) * 10) % 100, 99)  # Simple state hashing
            return np.argmax(self.q_table[state_index])
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory."""
        self.memory.push(state, action, reward, next_state, done)
    
    def learn(self, experiences: List[Tuple] = None) -> Dict[str, float]:
        """Learn from experiences using DQN algorithm."""
        if experiences is None and len(self.memory) < self.batch_size:
            return {'loss': 0.0}
        
        if TORCH_AVAILABLE and hasattr(self, 'q_network'):
            return self._learn_neural_network(experiences)
        else:
            return self._learn_table_based(experiences)
    
    def _learn_neural_network(self, experiences: List[Tuple] = None) -> Dict[str, float]:
        """Learn using neural network."""
        if experiences is None:
            experiences = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update exploration rate
        if self.training:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_rate * self.exploration_decay
            )
        
        # Update target network
        self.total_steps += 1
        if self.total_steps % self.target_update_frequency == 0:
            self.update_target_network()
        
        return {'loss': loss.item(), 'exploration_rate': self.exploration_rate}
    
    def _learn_table_based(self, experiences: List[Tuple] = None) -> Dict[str, float]:
        """Learn using Q-table (fallback method)."""
        if experiences is None:
            if len(self.memory) > 0:
                experiences = random.sample(list(self.memory.memory), min(len(self.memory), 10))
            else:
                return {'loss': 0.0}
        
        total_loss = 0.0
        
        for state, action, reward, next_state, done in experiences:
            state_index = min(int(np.sum(state) * 10) % 100, 99)
            next_state_index = min(int(np.sum(next_state) * 10) % 100, 99)
            
            # Q-learning update
            if done:
                target = reward
            else:
                target = reward + self.gamma * np.max(self.q_table[next_state_index])
            
            current_q = self.q_table[state_index, action]
            self.q_table[state_index, action] += self.learning_rate * (target - current_q)
            
            total_loss += abs(target - current_q)
        
        # Update exploration rate
        if self.training:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_rate * self.exploration_decay
            )
        
        return {'loss': total_loss / len(experiences), 'exploration_rate': self.exploration_rate}
    
    def update_target_network(self):
        """Copy weights from main network to target network."""
        if TORCH_AVAILABLE and hasattr(self, 'target_network'):
            self.target_network.load_state_dict(self.q_network.state_dict())
    
    def save_model(self, filepath: str):
        """Save the DQN model."""
        if TORCH_AVAILABLE and hasattr(self, 'q_network'):
            torch.save({
                'q_network_state_dict': self.q_network.state_dict(),
                'target_network_state_dict': self.target_network.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'exploration_rate': self.exploration_rate,
                'total_steps': self.total_steps
            }, filepath)
        else:
            np.save(filepath, self.q_table)
        print(f"✅ DQN model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load the DQN model."""
        if TORCH_AVAILABLE and hasattr(self, 'q_network'):
            checkpoint = torch.load(filepath, map_location=self.device)
            self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
            self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.exploration_rate = checkpoint.get('exploration_rate', self.exploration_rate)
            self.total_steps = checkpoint.get('total_steps', 0)
        else:
            self.q_table = np.load(filepath)
        print(f"✅ DQN model loaded from {filepath}")


class DoubleDQNAgent(DQNAgent):
    """Double DQN Agent - extends DQN with double Q-learning."""
    
    def _learn_neural_network(self, experiences: List[Tuple] = None) -> Dict[str, float]:
        """Learn using Double DQN algorithm."""
        if experiences is None:
            experiences = self.memory.sample(self.batch_size)
        
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Double DQN: use main network to select actions, target network to evaluate
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update exploration rate and target network
        if self.training:
            self.exploration_rate = max(
                self.exploration_min,
                self.exploration_rate * self.exploration_decay
            )
        
        self.total_steps += 1
        if self.total_steps % self.target_update_frequency == 0:
            self.update_target_network()
        
        return {'loss': loss.item(), 'exploration_rate': self.exploration_rate}
