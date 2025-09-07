"""
Intrinsic Curiosity Module (ICM) for Enhanced Exploration.

This module implements the Intrinsic Curiosity Module from:
"Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)

The ICM consists of:
1. Inverse Model: Predicts action given current and next state
2. Forward Model: Predicts next state features given current state features and action
3. Intrinsic Reward: Based on prediction error of the forward model
"""

import logging
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

logger = logging.getLogger(__name__)


class FeatureExtractor(nn.Module):
    """
    Feature extractor network for ICM.

    Extracts meaningful features from state observations
    for use in forward and inverse models.
    """

    def __init__(self, input_dim: int, feature_dim: int = 512):
        """
        Initialize feature extractor.

        Args:
            input_dim: Dimension of input state
            feature_dim: Dimension of extracted features
        """
        super(FeatureExtractor, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim

        # Feature extraction layers
        self.features = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU()
        )

        logger.info(f"FeatureExtractor initialized - Input: {input_dim}, Features: {feature_dim}")

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Extract features from state.

        Args:
            state: Input state tensor

        Returns:
            Extracted features
        """
        return self.features(state)


class InverseModel(nn.Module):
    """
    Inverse model for ICM.

    Predicts the action taken given current and next state features.
    This helps learn meaningful state representations.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize inverse model.

        Args:
            feature_dim: Dimension of state features
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(InverseModel, self).__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # Inverse model network
        self.inverse_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        logger.info(f"InverseModel initialized - Features: {feature_dim}, Actions: {action_dim}")

    def forward(self, current_features: torch.Tensor,
                next_features: torch.Tensor) -> torch.Tensor:
        """
        Predict action from state features.

        Args:
            current_features: Current state features
            next_features: Next state features

        Returns:
            Predicted action logits
        """
        # Concatenate current and next features
        combined_features = torch.cat([current_features, next_features], dim=-1)
        return self.inverse_net(combined_features)


class ForwardModel(nn.Module):
    """
    Forward model for ICM.

    Predicts next state features given current state features and action.
    Prediction error provides intrinsic reward signal.
    """

    def __init__(self, feature_dim: int, action_dim: int, hidden_dim: int = 256):
        """
        Initialize forward model.

        Args:
            feature_dim: Dimension of state features
            action_dim: Dimension of action space
            hidden_dim: Hidden layer dimension
        """
        super(ForwardModel, self).__init__()

        self.feature_dim = feature_dim
        self.action_dim = action_dim

        # Forward model network
        self.forward_net = nn.Sequential(
            nn.Linear(feature_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim)
        )

        logger.info(f"ForwardModel initialized - Features: {feature_dim}, Actions: {action_dim}")

    def forward(self, current_features: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        """
        Predict next state features.

        Args:
            current_features: Current state features
            action: Action taken

        Returns:
            Predicted next state features
        """
        # Concatenate features and action
        combined_input = torch.cat([current_features, action], dim=-1)
        return self.forward_net(combined_input)


class ICMModule(nn.Module):
    """
    Complete Intrinsic Curiosity Module.

    Combines feature extractor, inverse model, and forward model
    to provide intrinsic motivation for exploration.
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 feature_dim: int = 512,
                 hidden_dim: int = 256,
                 eta: float = 0.01,
                 beta: float = 0.2,
                 lambda_inverse: float = 0.1,
                 device: str = 'cpu'):
        """
        Initialize ICM module.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            feature_dim: Dimension of extracted features
            hidden_dim: Hidden layer dimension
            eta: Scaling factor for intrinsic reward
            beta: Weight for forward loss vs inverse loss
            lambda_inverse: Weight for inverse model loss
            device: Device to run on
        """
        super(ICMModule, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.eta = eta
        self.beta = beta
        self.lambda_inverse = lambda_inverse
        self.device = device

        # ICM components
        self.feature_extractor = FeatureExtractor(state_dim, feature_dim)
        self.inverse_model = InverseModel(feature_dim, action_dim, hidden_dim)
        self.forward_model = ForwardModel(feature_dim, action_dim, hidden_dim)

        # Move to device
        self.to(device)

        # Optimizer for ICM
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

        # Loss tracking
        self.forward_losses = []
        self.inverse_losses = []
        self.total_losses = []

        logger.info(f"ICMModule initialized - State: {state_dim}, Action: {action_dim}, "
                   f"Features: {feature_dim}, Device: {device}")

    def get_intrinsic_reward(self,
                           current_state: torch.Tensor,
                           action: torch.Tensor,
                           next_state: torch.Tensor) -> torch.Tensor:
        """
        Calculate intrinsic reward based on forward model prediction error.

        Args:
            current_state: Current state
            action: Action taken
            next_state: Next state

        Returns:
            Intrinsic reward
        """
        with torch.no_grad():
            # Extract features
            current_features = self.feature_extractor(current_state)
            next_features = self.feature_extractor(next_state)

            # Predict next features
            predicted_next_features = self.forward_model(current_features, action)

            # Calculate prediction error (intrinsic reward)
            prediction_error = F.mse_loss(predicted_next_features, next_features, reduction='none')
            intrinsic_reward = self.eta * prediction_error.mean(dim=-1)

            return intrinsic_reward

    def update(self,
               states: torch.Tensor,
               actions: torch.Tensor,
               next_states: torch.Tensor) -> Dict[str, float]:
        """
        Update ICM parameters using collected transitions.

        Args:
            states: Batch of current states
            actions: Batch of actions
            next_states: Batch of next states

        Returns:
            Dictionary of losses
        """
        # Extract features
        current_features = self.feature_extractor(states)
        next_features = self.feature_extractor(next_states)

        # Forward model prediction
        predicted_next_features = self.forward_model(current_features, actions)
        forward_loss = F.mse_loss(predicted_next_features, next_features.detach())

        # Inverse model prediction
        predicted_actions = self.inverse_model(current_features, next_features)

        # For discrete actions, use cross-entropy loss
        if actions.dtype == torch.long:
            inverse_loss = F.cross_entropy(predicted_actions, actions.squeeze())
        else:
            # For continuous actions, use MSE loss
            inverse_loss = F.mse_loss(predicted_actions, actions)

        # Combined ICM loss
        total_loss = (1 - self.beta) * inverse_loss + self.beta * forward_loss

        # Update parameters
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=0.5)
        self.optimizer.step()

        # Track losses
        forward_loss_val = forward_loss.item()
        inverse_loss_val = inverse_loss.item()
        total_loss_val = total_loss.item()

        self.forward_losses.append(forward_loss_val)
        self.inverse_losses.append(inverse_loss_val)
        self.total_losses.append(total_loss_val)

        return {
            'forward_loss': forward_loss_val,
            'inverse_loss': inverse_loss_val,
            'total_icm_loss': total_loss_val
        }

    def get_curiosity_stats(self) -> Dict[str, float]:
        """
        Get statistics about curiosity module performance.

        Returns:
            Dictionary of statistics
        """
        if not self.forward_losses:
            return {}

        return {
            'avg_forward_loss': np.mean(self.forward_losses[-100:]),
            'avg_inverse_loss': np.mean(self.inverse_losses[-100:]),
            'avg_total_loss': np.mean(self.total_losses[-100:]),
            'forward_loss_trend': np.mean(self.forward_losses[-10:]) - np.mean(self.forward_losses[-50:-10]) if len(self.forward_losses) > 50 else 0
        }

    def reset_stats(self):
        """Reset loss tracking statistics."""
        self.forward_losses = []
        self.inverse_losses = []
        self.total_losses = []
        logger.info("ICM statistics reset")

    def save_state(self, filepath: str):
        """
        Save ICM state to file.

        Args:
            filepath: Path to save state
        """
        state = {
            'feature_extractor_state': self.feature_extractor.state_dict(),
            'inverse_model_state': self.inverse_model.state_dict(),
            'forward_model_state': self.forward_model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'config': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'feature_dim': self.feature_dim,
                'eta': self.eta,
                'beta': self.beta,
                'lambda_inverse': self.lambda_inverse
            },
            'losses': {
                'forward_losses': self.forward_losses,
                'inverse_losses': self.inverse_losses,
                'total_losses': self.total_losses
            }
        }

        torch.save(state, filepath)
        logger.info(f"ICM state saved to {filepath}")

    def load_state(self, filepath: str):
        """
        Load ICM state from file.

        Args:
            filepath: Path to load state from
        """
        state = torch.load(filepath, map_location=self.device)

        self.feature_extractor.load_state_dict(state['feature_extractor_state'])
        self.inverse_model.load_state_dict(state['inverse_model_state'])
        self.forward_model.load_state_dict(state['forward_model_state'])
        self.optimizer.load_state_dict(state['optimizer_state'])

        # Restore loss tracking
        if 'losses' in state:
            self.forward_losses = state['losses']['forward_losses']
            self.inverse_losses = state['losses']['inverse_losses']
            self.total_losses = state['losses']['total_losses']

        logger.info(f"ICM state loaded from {filepath}")


class CuriosityDrivenAgent:
    """
    Wrapper class that integrates ICM with any RL agent.

    Provides intrinsic rewards to enhance exploration in sparse reward environments.
    """

    def __init__(self,
                 base_agent,
                 icm_module: ICMModule,
                 intrinsic_reward_weight: float = 0.5,
                 update_icm_frequency: int = 1):
        """
        Initialize curiosity-driven agent.

        Args:
            base_agent: Base RL agent (DQN, PPO, etc.)
            icm_module: ICM module for intrinsic rewards
            intrinsic_reward_weight: Weight for intrinsic vs extrinsic rewards
            update_icm_frequency: Frequency of ICM updates
        """
        self.base_agent = base_agent
        self.icm_module = icm_module
        self.intrinsic_reward_weight = intrinsic_reward_weight
        self.update_icm_frequency = update_icm_frequency

        # Tracking
        self.step_count = 0
        self.intrinsic_rewards_history = []
        self.extrinsic_rewards_history = []

        logger.info(f"CuriosityDrivenAgent initialized - Intrinsic weight: {intrinsic_reward_weight}")

    def get_action(self, state):
        """Get action from base agent."""
        return self.base_agent.get_action(state)

    def store_transition(self, state, action, reward, next_state, done):
        """
        Store transition with combined intrinsic and extrinsic reward.

        Args:
            state: Current state
            action: Action taken
            reward: Extrinsic reward
            next_state: Next state
            done: Episode termination flag
        """
        # Convert to tensors if needed
        if not isinstance(state, torch.Tensor):
            state = torch.FloatTensor(state).to(self.icm_module.device)
        if not isinstance(next_state, torch.Tensor):
            next_state = torch.FloatTensor(next_state).to(self.icm_module.device)
        if not isinstance(action, torch.Tensor):
            if isinstance(action, (int, np.integer)):
                action = torch.LongTensor([action]).to(self.icm_module.device)
            else:
                action = torch.FloatTensor(action).to(self.icm_module.device)

        # Get intrinsic reward
        intrinsic_reward = self.icm_module.get_intrinsic_reward(
            state.unsqueeze(0), action.unsqueeze(0), next_state.unsqueeze(0)
        ).item()

        # Combine rewards
        combined_reward = reward + self.intrinsic_reward_weight * intrinsic_reward

        # Store in base agent
        self.base_agent.store_transition(state, action, combined_reward, next_state, done)

        # Track rewards
        self.intrinsic_rewards_history.append(intrinsic_reward)
        self.extrinsic_rewards_history.append(reward)

        self.step_count += 1

    def update(self):
        """Update both base agent and ICM."""
        # Update base agent
        base_losses = self.base_agent.update()

        # Update ICM periodically
        icm_losses = {}
        if self.step_count % self.update_icm_frequency == 0:
            # Get batch of transitions for ICM update
            if hasattr(self.base_agent, 'memory') and len(self.base_agent.memory) > 32:
                # Sample batch for ICM update
                batch = self.base_agent.memory.sample(32)

                states = torch.stack([transition.state for transition in batch])
                actions = torch.stack([transition.action for transition in batch])
                next_states = torch.stack([transition.next_state for transition in batch])

                icm_losses = self.icm_module.update(states, actions, next_states)

        # Combine losses
        if isinstance(base_losses, dict):
            combined_losses = base_losses.copy()
            combined_losses.update(icm_losses)
        else:
            combined_losses = icm_losses

        return combined_losses

    def get_exploration_stats(self) -> Dict[str, float]:
        """
        Get exploration statistics.

        Returns:
            Dictionary of exploration metrics
        """
        stats = {}

        if self.intrinsic_rewards_history:
            stats.update({
                'avg_intrinsic_reward': np.mean(self.intrinsic_rewards_history[-100:]),
                'avg_extrinsic_reward': np.mean(self.extrinsic_rewards_history[-100:]),
                'intrinsic_reward_std': np.std(self.intrinsic_rewards_history[-100:]),
                'exploration_bonus_ratio': np.mean(self.intrinsic_rewards_history[-100:]) / (np.mean(self.extrinsic_rewards_history[-100:]) + 1e-8)
            })

        # Add ICM stats
        stats.update(self.icm_module.get_curiosity_stats())

        return stats

    def save_state(self, filepath: str):
        """Save curiosity-driven agent state."""
        # Save base agent if it has save method
        if hasattr(self.base_agent, 'save_state'):
            base_filepath = filepath.replace('.pth', '_base.pth')
            self.base_agent.save_state(base_filepath)

        # Save ICM
        icm_filepath = filepath.replace('.pth', '_icm.pth')
        self.icm_module.save_state(icm_filepath)

        # Save wrapper state
        wrapper_state = {
            'intrinsic_reward_weight': self.intrinsic_reward_weight,
            'update_icm_frequency': self.update_icm_frequency,
            'step_count': self.step_count,
            'intrinsic_rewards_history': self.intrinsic_rewards_history,
            'extrinsic_rewards_history': self.extrinsic_rewards_history
        }

        wrapper_filepath = filepath.replace('.pth', '_wrapper.pth')
        torch.save(wrapper_state, wrapper_filepath)

        logger.info(f"CuriosityDrivenAgent state saved to {filepath}")

    def load_state(self, filepath: str):
        """Load curiosity-driven agent state."""
        # Load base agent if it has load method
        if hasattr(self.base_agent, 'load_state'):
            base_filepath = filepath.replace('.pth', '_base.pth')
            self.base_agent.load_state(base_filepath)

        # Load ICM
        icm_filepath = filepath.replace('.pth', '_icm.pth')
        self.icm_module.load_state(icm_filepath)

        # Load wrapper state
        wrapper_filepath = filepath.replace('.pth', '_wrapper.pth')
        wrapper_state = torch.load(wrapper_filepath)

        self.intrinsic_reward_weight = wrapper_state['intrinsic_reward_weight']
        self.update_icm_frequency = wrapper_state['update_icm_frequency']
        self.step_count = wrapper_state['step_count']
        self.intrinsic_rewards_history = wrapper_state['intrinsic_rewards_history']
        self.extrinsic_rewards_history = wrapper_state['extrinsic_rewards_history']

        logger.info(f"CuriosityDrivenAgent state loaded from {filepath}")


# Factory function for creating ICM-enhanced agents
def create_curiosity_driven_agent(base_agent,
                                 state_dim: int,
                                 action_dim: int,
                                 intrinsic_reward_weight: float = 0.5,
                                 device: str = 'cpu') -> CuriosityDrivenAgent:
    """
    Factory function to create a curiosity-driven agent.

    Args:
        base_agent: Base RL agent
        state_dim: State space dimension
        action_dim: Action space dimension
        intrinsic_reward_weight: Weight for intrinsic rewards
        device: Device to run on

    Returns:
        CuriosityDrivenAgent instance
    """
    icm_module = ICMModule(
        state_dim=state_dim,
        action_dim=action_dim,
        device=device
    )

    return CuriosityDrivenAgent(
        base_agent=base_agent,
        icm_module=icm_module,
        intrinsic_reward_weight=intrinsic_reward_weight
    )
