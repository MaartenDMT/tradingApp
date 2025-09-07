# RL Trading System Examples

This directory contains comprehensive examples demonstrating the capabilities of the RL Trading System. The examples are organized by complexity and feature type to help you learn and use the system effectively.

## 📁 Directory Structure

```
examples/
├── basic_usage/           # Fundamental examples for getting started
├── advanced_features/     # Advanced capabilities and features
├── trading_scenarios/     # Real-world trading applications
└── README.md             # This file
```

## 🚀 Quick Start

### Prerequisites

Make sure you have the RL Trading System properly installed and configured:

```bash
# From the project root directory
pip install -e .
```

### Running Examples

Each example is self-contained and can be run independently:

```bash
# Basic DQN example
python examples/basic_usage/dqn_example.py

# PPO continuous control example
python examples/basic_usage/ppo_example.py

# Algorithm switching demonstration
python examples/advanced_features/algorithm_switching.py

# ICM exploration enhancement
python examples/advanced_features/icm_exploration.py
```

## 📚 Example Categories

### 🔰 Basic Usage Examples

These examples demonstrate fundamental concepts and basic algorithm usage:

#### `dqn_example.py` - DQN Trading Basics
- **Purpose**: Learn basic DQN training for discrete trading actions
- **Concepts**: Environment setup, agent creation, training loop, evaluation
- **Environment**: Simple trading with Hold/Buy/Sell actions
- **Key Learning**:
  - How to create and configure DQN agents
  - Basic training and evaluation workflows
  - Performance visualization
  - Agent state saving/loading

#### `ppo_example.py` - PPO Continuous Control
- **Purpose**: Demonstrate PPO for continuous portfolio allocation
- **Concepts**: Continuous action spaces, portfolio weights, risk management
- **Environment**: Multi-asset portfolio with continuous allocation
- **Key Learning**:
  - Continuous control vs discrete actions
  - Portfolio allocation strategies
  - Risk-adjusted performance metrics (Sharpe ratio)
  - Advanced visualization techniques

### 🎯 Advanced Features Examples

These examples showcase sophisticated system capabilities:

#### `algorithm_switching.py` - Adaptive Algorithm Selection
- **Purpose**: Demonstrate automatic algorithm switching based on performance
- **Concepts**: Performance monitoring, state transfer, adaptive strategies
- **Environment**: Volatile market with regime changes
- **Key Learning**:
  - Setting up switching conditions
  - State transfer between algorithms
  - Performance-based adaptation
  - Comparative analysis

#### `icm_exploration.py` - Intrinsic Curiosity Module
- **Purpose**: Show ICM benefits in sparse reward environments
- **Concepts**: Exploration enhancement, intrinsic motivation, sparse rewards
- **Environment**: Trading with sparse milestone-based rewards
- **Key Learning**:
  - ICM integration with base algorithms
  - Exploration vs exploitation in sparse rewards
  - Curiosity-driven learning analysis
  - Comparative performance evaluation

### 💼 Trading Scenarios Examples

Real-world trading applications and scenarios:

#### `multi_asset_portfolio.py` - Portfolio Management
- **Purpose**: Professional portfolio management with multiple assets
- **Concepts**: Asset allocation, rebalancing, risk constraints
- **Environment**: Multi-asset portfolio with realistic constraints
- **Key Learning**:
  - Portfolio optimization
  - Risk management
  - Transaction costs
  - Performance attribution

#### `high_frequency_trading.py` - HFT Strategies
- **Purpose**: High-frequency trading with latency considerations
- **Concepts**: Order book dynamics, market microstructure, latency
- **Environment**: Limit order book with realistic market dynamics
- **Key Learning**:
  - Market making strategies
  - Order flow analysis
  - Latency-sensitive decision making
  - Risk management at high frequency

## 🎨 Visualization Features

All examples include comprehensive visualization capabilities:

- **Training Progress**: Episode rewards, loss curves, convergence analysis
- **Performance Metrics**: Risk-adjusted returns, drawdowns, Sharpe ratios
- **Algorithm Comparison**: Side-by-side performance comparisons
- **Exploration Analysis**: State visitation, curiosity patterns (for ICM)
- **Trading Behavior**: Action distributions, portfolio evolution

## 🛠️ Customization Guide

### Creating Your Own Examples

1. **Choose Base Template**: Start with the example closest to your use case
2. **Modify Environment**: Adapt the environment to your specific scenario
3. **Select Algorithm**: Choose appropriate algorithm(s) for your problem
4. **Configure Training**: Adjust hyperparameters and training settings
5. **Add Visualization**: Customize plots for your specific metrics

### Environment Customization

```python
# Example environment modification
class CustomTradingEnvironment:
    def __init__(self, custom_params):
        # Your custom initialization
        self.state_dim = your_state_dimension
        self.action_space = your_action_space

    def step(self, action):
        # Your custom environment logic
        return next_state, reward, done, info
```

### Algorithm Configuration

```python
# Example algorithm customization
agent = create_agent(
    algorithm_type="your_algorithm",
    state_dim=env.state_dim,
    action_dim=env.action_dim,
    config={
        'learning_rate': 0.001,
        'custom_param': your_value,
        # Add your custom parameters
    }
)
```

## 📊 Performance Metrics

### Standard Metrics

All examples report standard performance metrics:

- **Total Return**: Overall portfolio performance
- **Sharpe Ratio**: Risk-adjusted return measure
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades/episodes
- **Average Trade Duration**: Typical holding period

### Algorithm-Specific Metrics

- **DQN**: Epsilon decay, Q-value distributions, exploration rate
- **PPO**: Policy entropy, advantage estimation, clipping ratio
- **ICM**: Intrinsic vs extrinsic rewards, exploration coverage
- **Switching**: Switch frequency, transfer efficiency, adaptation speed

## 🔧 Troubleshooting

### Common Issues

1. **Import Errors**: Ensure the project is properly installed (`pip install -e .`)
2. **Memory Issues**: Reduce batch size or episode length for large examples
3. **Slow Training**: Enable GPU acceleration if available
4. **Visualization Errors**: Install matplotlib and seaborn dependencies

### Performance Optimization

```python
# GPU acceleration (if available)
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Memory optimization
config = {
    'batch_size': 32,        # Reduce if memory limited
    'memory_size': 10000,    # Adjust replay buffer size
    'update_frequency': 4    # Update less frequently
}
```

### Debugging Tips

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Monitor training progress
if episode % 10 == 0:
    print(f"Episode {episode}: Reward = {episode_reward:.2f}")

# Save intermediate checkpoints
if episode % 50 == 0:
    agent.save_state(f'checkpoint_episode_{episode}.pth')
```

## 🎓 Learning Path

### Beginner Path
1. Start with `dqn_example.py` to understand basics
2. Try `ppo_example.py` for continuous control
3. Experiment with different hyperparameters
4. Create your own simple environment

### Intermediate Path
1. Study `algorithm_switching.py` for adaptive strategies
2. Explore `icm_exploration.py` for advanced exploration
3. Combine multiple algorithms in ensembles
4. Implement custom reward functions

### Advanced Path
1. Study real-world trading scenarios
2. Implement custom algorithms
3. Add new features to the system
4. Contribute improvements back to the project

## 📖 Additional Resources

### Documentation
- [Algorithm Reference](../docs/algorithms.md)
- [Integration Guide](../docs/integration.md)
- [Visualization Tools](../docs/visualization.md)

### Papers and Research
- DQN: "Human-level control through deep reinforcement learning" (Mnih et al., 2015)
- PPO: "Proximal Policy Optimization Algorithms" (Schulman et al., 2017)
- ICM: "Curiosity-driven Exploration by Self-supervised Prediction" (Pathak et al., 2017)

### Community
- [GitHub Issues](../../issues) for bug reports and feature requests
- [Discussions](../../discussions) for questions and community support

## 🤝 Contributing

We welcome contributions to the examples collection:

1. **New Examples**: Add examples for new algorithms or scenarios
2. **Improvements**: Enhance existing examples with better visualization or explanation
3. **Documentation**: Improve README files and code comments
4. **Bug Fixes**: Report and fix issues in existing examples

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for detailed guidelines.

## 📄 License

This examples collection is part of the RL Trading System and follows the same license terms.

---

**Happy Learning and Trading! 🚀📈**

For questions or support, please open an issue or start a discussion in the repository.
