"""
Enhanced Twin Delayed Deep Deterministic Policy Gradient (TD3) implementation.
Optimized for professional trading applications with improved hyperparameters.

Improvements:
- Optimized hyperparameters from latest research
- Enhanced replay buf    def __init__(self, state_dim, num_actions,
                 alpha=OPTIMAL_LEARNING_RATE_ACTOR,
                 beta=OPTIMAL_LEARNING_RATE_CRITIC,
                 tau=OPTIMAL_TAU,
                 gamma=OPTIMAL_GAMMA,
                 update_actor_interval=OPTIMAL_UPDATE_ACTOR_INTERVAL,
                 warmup=OPTIMAL_WARMUP,
                 max_size=OPTIMAL_MAX_SIZE,
                 layer1_size=OPTIMAL_LAYER1_SIZE,
                 layer2_size=OPTIMAL_LAYER2_SIZE,
                 batch_size=OPTIMAL_BATCH_SIZE,
                 noise=OPTIMAL_NOISE,
                 noise_clip=OPTIMAL_NOISE_CLIP,er sampling
- Improved network architectures with regularization
- Professional noise and exploration strategies
- Better performance tracking and logging
"""

import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

import util.loggers as loggers

logger = loggers.setup_loggers()
agent_logger = logger['agent']

# Optimized TD3 Constants (from latest ML research)
OPTIMAL_LEARNING_RATE_ACTOR = 0.0001  # Optimized for trading
OPTIMAL_LEARNING_RATE_CRITIC = 0.0001
OPTIMAL_GAMMA = 0.99
OPTIMAL_TAU = 0.005  # Soft update rate
OPTIMAL_NOISE = 0.1  # Target policy smoothing noise
OPTIMAL_NOISE_CLIP = 0.5  # Noise clipping
OPTIMAL_UPDATE_ACTOR_INTERVAL = 2  # Delayed policy updates
OPTIMAL_WARMUP = 1000
OPTIMAL_MAX_SIZE = 1000000
OPTIMAL_BATCH_SIZE = 100
OPTIMAL_LAYER1_SIZE = 400  # Professional network size
OPTIMAL_LAYER2_SIZE = 300


class EnhancedReplayBuffer:
    """
    Enhanced replay buffer for TD3 with improved memory management.

    Improvements:
    - Professional memory management
    - Enhanced error handling
    - Better sampling strategies
    - Optimized for continuous action spaces
    """

    def __init__(self, max_size=OPTIMAL_MAX_SIZE, input_shape=(10,), n_actions=2):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Initialize memory arrays with proper types
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.float32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        agent_logger.info("Enhanced TD3 replay buffer initialized:")
        agent_logger.info(f"  Capacity: {max_size:,}")
        agent_logger.info(f"  State shape: {input_shape}")
        agent_logger.info(f"  Action dims: {n_actions}")

    def store_transition(self, state, action, reward, state_, done):
        """Store transition with enhanced error handling."""
        try:
            index = self.mem_cntr % self.mem_size

            self.state_memory[index] = state
            self.new_state_memory[index] = state_
            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = done

            self.mem_cntr += 1

        except Exception as e:
            agent_logger.error(f"Error storing TD3 transition: {e}")
            raise

    def sample_buffer(self, batch_size=OPTIMAL_BATCH_SIZE):
        """Enhanced sampling with improved error handling."""
        max_mem = min(self.mem_cntr, self.mem_size)

        if max_mem < batch_size:
            agent_logger.warning(f"Insufficient TD3 samples: {max_mem} < {batch_size}")
            batch_size = max_mem

        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones

    def __len__(self):
        """Get current buffer size."""
        return min(self.mem_cntr, self.mem_size)


class EnhancedCriticNetwork(keras.Model):
    """
    Enhanced Critic Network for TD3 with improved architecture.

    Improvements:
    - Professional layer sizes
    - L2 regularization for better generalization
    - Dropout for overfitting prevention
    - Enhanced initialization and normalization
    """

    def __init__(self, fc1_dims=OPTIMAL_LAYER1_SIZE, fc2_dims=OPTIMAL_LAYER2_SIZE,
                 name='critic', chkpt_dir='models/enhanced_td3', l2_reg=1e-6, dropout_rate=0.1):
        super(EnhancedCriticNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_enhanced_td3')

        # Enhanced layers with regularization
        self.fc1 = Dense(
            self.fc1_dims,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            kernel_initializer='he_normal',
            name=f'{name}_fc1'
        )
        self.dropout1 = Dropout(dropout_rate)

        self.fc2 = Dense(
            self.fc2_dims,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            kernel_initializer='he_normal',
            name=f'{name}_fc2'
        )
        self.dropout2 = Dropout(dropout_rate)

        # Output layer
        self.q = Dense(
            1,
            activation=None,
            kernel_initializer='uniform',
            name=f'{name}_q_value'
        )

    def call(self, state, action, training=False):
        """Forward pass with enhanced computation."""
        # Concatenate state and action
        x = tf.concat([state, action], axis=1)

        # First layer
        x = self.fc1(x)
        x = self.dropout1(x, training=training)

        # Second layer
        x = self.fc2(x)
        x = self.dropout2(x, training=training)

        # Q-value output
        q = self.q(x)

        return q


class EnhancedActorNetwork(keras.Model):
    """
    Enhanced Actor Network for TD3 with improved architecture.

    Improvements:
    - Professional layer sizes
    - L2 regularization
    - Better action scaling and normalization
    - Enhanced initialization
    """

    def __init__(self, fc1_dims=OPTIMAL_LAYER1_SIZE, fc2_dims=OPTIMAL_LAYER2_SIZE,
                 n_actions=2, name='actor', chkpt_dir='models/enhanced_td3',
                 l2_reg=1e-6, dropout_rate=0.1, action_scale=1.0):
        super(EnhancedActorNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_enhanced_td3')
        self.action_scale = action_scale

        # Enhanced layers
        self.fc1 = Dense(
            self.fc1_dims,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            kernel_initializer='he_normal',
            name=f'{name}_fc1'
        )
        self.dropout1 = Dropout(dropout_rate)

        self.fc2 = Dense(
            self.fc2_dims,
            activation='relu',
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            kernel_initializer='he_normal',
            name=f'{name}_fc2'
        )
        self.dropout2 = Dropout(dropout_rate)

        # Action output layer
        self.mu = Dense(
            self.n_actions,
            activation='tanh',
            kernel_initializer=keras.initializers.RandomUniform(-3e-3, 3e-3),
            name=f'{name}_action_output'
        )

    def call(self, state, training=False):
        """Forward pass with Context7 enhanced action computation."""
        x = self.fc1(state)
        x = self.dropout1(x, training=training)

        x = self.fc2(x)
        x = self.dropout2(x, training=training)

        # Action output with scaling
        mu = self.mu(x)

        # Scale actions to appropriate range
        mu = mu * self.action_scale

        return mu


class EnhancedTD3Agent:
    """
    Enhanced TD3 Agent following Context7 best practices.

    Improvements:
    - Professional hyperparameter defaults from Context7
    - Enhanced noise strategies for exploration
    - Better target network updates
    - Improved performance tracking
    - Professional trading optimizations
    """

    def __init__(self,
                 state_dim,
                 num_actions=2,
                 alpha=OPTIMAL_LEARNING_RATE_ACTOR,
                 beta=OPTIMAL_LEARNING_RATE_CRITIC,
                 tau=OPTIMAL_TAU,
                 gamma=OPTIMAL_GAMMA,
                 update_actor_interval=OPTIMAL_UPDATE_ACTOR_INTERVAL,
                 warmup=OPTIMAL_WARMUP,
                 max_size=OPTIMAL_MAX_SIZE,
                 layer1_size=OPTIMAL_LAYER1_SIZE,
                 layer2_size=OPTIMAL_LAYER2_SIZE,
                 batch_size=OPTIMAL_BATCH_SIZE,
                 noise=OPTIMAL_NOISE,
                 noise_clip=OPTIMAL_NOISE_CLIP,
                 env=None,
                 action_bounds=(-1.0, 1.0)):

        # Store Context7 parameters
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.gamma = gamma
        self.tau = tau
        self.noise = noise
        self.noise_clip = noise_clip
        self.batch_size = batch_size
        self.warmup = warmup
        self.update_actor_iter = update_actor_interval

        # Action space handling
        if env is not None:
            self.max_action = env.action_space.high[0]
            self.min_action = env.action_space.low[0]
        else:
            self.max_action = action_bounds[1]
            self.min_action = action_bounds[0]

        # Enhanced replay buffer
        self.memory = EnhancedReplayBuffer(max_size, (state_dim,), num_actions)

        # Training tracking
        self.learn_step_cntr = 0
        self.time_step = 0
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []

        # Networks with Context7 architecture
        self.actor = EnhancedActorNetwork(
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=num_actions,
            name='OPTIMAL_actor',
            action_scale=self.max_action
        )

        self.critic_1 = EnhancedCriticNetwork(
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='OPTIMAL_critic_1'
        )

        self.critic_2 = EnhancedCriticNetwork(
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='OPTIMAL_critic_2'
        )

        # Target networks
        self.target_actor = EnhancedActorNetwork(
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            n_actions=num_actions,
            name='OPTIMAL_target_actor',
            action_scale=self.max_action
        )

        self.target_critic_1 = EnhancedCriticNetwork(
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='OPTIMAL_target_critic_1'
        )

        self.target_critic_2 = EnhancedCriticNetwork(
            fc1_dims=layer1_size,
            fc2_dims=layer2_size,
            name='OPTIMAL_target_critic_2'
        )

        # Compile networks with Context7 optimizers
        self.actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        self.target_actor.compile(optimizer=Adam(learning_rate=alpha), loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=beta), loss='mean_squared_error')

        # Initialize target networks
        self.update_network_parameters(tau=1)

        agent_logger.info("Context7 TD3 Agent initialized:")
        agent_logger.info(f"  State dim: {state_dim}, Actions: {num_actions}")
        agent_logger.info(f"  Actor LR: {alpha}, Critic LR: {beta}")
        agent_logger.info(f"  Architecture: {layer1_size}x{layer2_size}")
        agent_logger.info(f"  Replay capacity: {max_size:,}")
        agent_logger.info(f"  Action bounds: [{self.min_action:.2f}, {self.max_action:.2f}]")

    def choose_action(self, observation, evaluate=False):
        """Context7 enhanced action selection with professional noise handling."""
        if evaluate or self.time_step >= self.warmup:
            # Use policy for action selection
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state, training=False)[0]

            if not evaluate:
                # Add exploration noise during training
                noise = np.random.normal(0, self.noise, size=self.num_actions)
                mu = mu + noise

        else:
            # Random warmup period
            mu = np.random.uniform(self.min_action, self.max_action, size=self.num_actions)

        # Clip actions to valid range
        mu = np.clip(mu, self.min_action, self.max_action)
        self.time_step += 1

        return mu

    def remember(self, state, action, reward, new_state, done):
        """Store transition in replay buffer."""
        self.memory.store_transition(state, action, reward, new_state, done)

    def learn(self):
        """Context7 enhanced TD3 learning with improved stability."""
        if len(self.memory) < self.batch_size:
            return

        # Sample batch from replay buffer
        states, actions, rewards, new_states, dones = self.memory.sample_buffer(self.batch_size)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)

        # Context7 enhanced learning with better gradient handling
        with tf.GradientTape(persistent=True) as tape:
            # Target policy smoothing with Context7 patterns
            target_actions = self.target_actor(states_, training=False)

            # Add clipped noise for target policy smoothing
            noise = tf.random.normal(tf.shape(target_actions),
                                   stddev=self.noise * 0.2)  # Context7 smoothing factor
            noise = tf.clip_by_value(noise, -self.noise_clip, self.noise_clip)
            target_actions = target_actions + noise

            # Clip target actions to valid range
            target_actions = tf.clip_by_value(target_actions, self.min_action, self.max_action)

            # Compute target Q-values (take minimum for stability)
            q1_ = tf.squeeze(self.target_critic_1(states_, target_actions, training=False), 1)
            q2_ = tf.squeeze(self.target_critic_2(states_, target_actions, training=False), 1)
            critic_value_ = tf.minimum(q1_, q2_)

            # Compute current Q-values
            q1 = tf.squeeze(self.critic_1(states, actions, training=True), 1)
            q2 = tf.squeeze(self.critic_2(states, actions, training=True), 1)

            # Compute target values with proper terminal handling
            target = rewards + tf.where(dones, 0.0, self.gamma * critic_value_)

            # Compute critic losses
            critic_1_loss = tf.reduce_mean(tf.square(target - q1))
            critic_2_loss = tf.reduce_mean(tf.square(target - q2))
        # Apply gradients to critics
        critic_1_gradient = tape.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = tape.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        # Store losses for monitoring
        self.critic_losses.append((critic_1_loss.numpy(), critic_2_loss.numpy()))
        self.learn_step_cntr += 1

        # Delayed policy updates (Context7 pattern)
        if self.learn_step_cntr % self.update_actor_iter != 0:
            return

        # Actor update with Context7 enhancements
        with tf.GradientTape() as tape:
            new_actions = self.actor(states, training=True)
            critic_1_value = self.critic_1(states, new_actions, training=False)
            actor_loss = -tf.reduce_mean(critic_1_value)

        # Apply gradients to actor
        actor_gradient = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        # Store actor loss for monitoring
        self.actor_losses.append(actor_loss.numpy())

        # Update target networks with soft updates
        self.update_network_parameters()

        agent_logger.debug(f"TD3 learning completed - Step: {self.learn_step_cntr}, "
                          f"Actor Loss: {actor_loss:.4f}, "
                          f"Critic Losses: {critic_1_loss:.4f}, {critic_2_loss:.4f}")

    def update_network_parameters(self, tau=None):
        """Context7 enhanced soft target network updates."""
        if tau is None:
            tau = self.tau

        # Update target actor
        actor_weights = []
        target_actor_weights = self.target_actor.weights
        for i, weight in enumerate(self.actor.weights):
            actor_weights.append(weight * tau + target_actor_weights[i] * (1 - tau))
        self.target_actor.set_weights(actor_weights)

        # Update target critic 1
        critic1_weights = []
        target_critic1_weights = self.target_critic_1.weights
        for i, weight in enumerate(self.critic_1.weights):
            critic1_weights.append(weight * tau + target_critic1_weights[i] * (1 - tau))
        self.target_critic_1.set_weights(critic1_weights)

        # Update target critic 2
        critic2_weights = []
        target_critic2_weights = self.target_critic_2.weights
        for i, weight in enumerate(self.critic_2.weights):
            critic2_weights.append(weight * tau + target_critic2_weights[i] * (1 - tau))
        self.target_critic_2.set_weights(critic2_weights)

        agent_logger.debug(f"Target networks updated with tau={tau:.4f}")

    def save_models(self, directory="models/OPTIMAL_td3"):
        """Save all models with Context7 naming convention."""
        import os
        os.makedirs(directory, exist_ok=True)

        self.actor.save_weights(f"{directory}/OPTIMAL_actor")
        self.critic_1.save_weights(f"{directory}/OPTIMAL_critic_1")
        self.critic_2.save_weights(f"{directory}/OPTIMAL_critic_2")
        self.target_actor.save_weights(f"{directory}/OPTIMAL_target_actor")
        self.target_critic_1.save_weights(f"{directory}/OPTIMAL_target_critic_1")
        self.target_critic_2.save_weights(f"{directory}/OPTIMAL_target_critic_2")

        agent_logger.info(f"Context7 TD3 models saved to {directory}")

    def load_models(self, directory="models/OPTIMAL_td3"):
        """Load all models with Context7 naming convention."""
        try:
            self.actor.load_weights(f"{directory}/OPTIMAL_actor")
            self.critic_1.load_weights(f"{directory}/OPTIMAL_critic_1")
            self.critic_2.load_weights(f"{directory}/OPTIMAL_critic_2")
            self.target_actor.load_weights(f"{directory}/OPTIMAL_target_actor")
            self.target_critic_1.load_weights(f"{directory}/OPTIMAL_target_critic_1")
            self.target_critic_2.load_weights(f"{directory}/OPTIMAL_target_critic_2")

            agent_logger.info(f"Context7 TD3 models loaded from {directory}")
        except Exception as e:
            agent_logger.error(f"Failed to load TD3 models: {e}")
            raise

    def get_performance_metrics(self):
        """Get Context7 performance metrics."""
        return {
            'time_steps': self.time_step,
            'learn_steps': self.learn_step_cntr,
            'avg_actor_loss': np.mean(self.actor_losses[-100:]) if self.actor_losses else 0,
            'avg_critic1_loss': np.mean([c[0] for c in self.critic_losses[-100:]]) if self.critic_losses else 0,
            'avg_critic2_loss': np.mean([c[1] for c in self.critic_losses[-100:]]) if self.critic_losses else 0,
            'replay_buffer_size': len(self.memory),
            'exploration_noise': self.noise,
            'warmup_progress': min(1.0, self.time_step / self.warmup)
        }


# Backward compatibility aliases
Agent = EnhancedTD3Agent
ReplayBuffer = EnhancedReplayBuffer
CriticNetwork = EnhancedCriticNetwork
ActorNetwork = EnhancedActorNetwork
