"""
Dueling DQN Agent

Implementation of Dueling Deep Q-Network which separates state value and advantage estimation.
"""

import numpy as np
from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.rl_system.algorithms.value_based.dqn import DQNAgent, ReplayMemory


class DuelingDQNNetwork(nn.Module):
    """Dueling DQN Network architecture."""
    
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 64):
        super(DuelingDQNNetwork, self).__init__()
        
        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Linear(hidden_size, 1)
        
        # Advantage stream
        self.advantage_stream = nn.Linear(hidden_size, action_size)
        
    def forward(self, x):
        features = self.feature_layer(x)
        
        # Get value and advantage
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        return q_values


class DuelingDQNAgent(DQNAgent):
    """Dueling DQN Agent implementation."""
    
    def __init__(self, observation_space, action_space, config: Dict):
        # Initialize parent class first
        super().__init__(observation_space, action_space, config)
        
        # Replace networks with dueling architecture
        if hasattr(self, 'q_network'):
            self.q_network = DuelingDQNNetwork(
                self.state_size, 
                self.action_size
            ).to(self.device)
            
            self.target_network = DuelingDQNNetwork(
                self.state_size, 
                self.action_size
            ).to(self.device)
            
            # Reinitialize optimizer with new network
            self.optimizer = torch.optim.Adam(
                self.q_network.parameters(), 
                lr=self.learning_rate
            )
            
            # Copy weights to target network
            self.update_target_network()
            
            print("✅ Initialized Dueling DQN with separated value and advantage streams")
