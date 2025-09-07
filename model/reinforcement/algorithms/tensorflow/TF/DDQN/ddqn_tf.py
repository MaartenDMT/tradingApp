"""
Enhanced Double Deep Q-Network (DDQN) implementation following Context7 best practices.

This implementation incorporates:
- Context7 optimized hyperparameters
- Professional experience replay patterns
- Enhanced epsilon scheduling
- Improved network architecture
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import util.loggers as loggers

logger = loggers.setup_loggers()
agent_logger = logger['agent']

# Context7 Constants
CONTEXT7_REPLAY_CAPACITY = int(1e6)  # Larger buffer for better sampling
CONTEXT7_BATCH_SIZE = 4096  # Optimized batch size
CONTEXT7_ARCHITECTURE = (256, 256)  # Professional hidden layer sizes
CONTEXT7_L2_REG = 1e-6  # L2 regularization
CONTEXT7_LEARNING_RATE = 0.0001  # Optimized learning rate
CONTEXT7_GAMMA = 0.99  # Discount factor
CONTEXT7_TAU = 100  # Target network update frequency
CONTEXT7_EPSILON_START = 1.0
CONTEXT7_EPSILON_END = 0.01
CONTEXT7_EPSILON_DECAY_STEPS = 250
CONTEXT7_EPSILON_EXPONENTIAL_DECAY = 0.99


class Context7DuelingDeepQNetwork(keras.Model):
    """
    Enhanced Dueling Deep Q-Network following Context7 best practices.

    Improvements:
    - Optimized architecture with professional layer sizes
    - L2 regularization for better generalization
    - Dropout for overfitting prevention
    - Enhanced advantage function computation
    """

    def __init__(self, fc1_dims=CONTEXT7_ARCHITECTURE[0], fc2_dims=CONTEXT7_ARCHITECTURE[1],
                 n_actions=3, l2_reg=CONTEXT7_L2_REG, dropout_rate=0.1):
        super(Context7DuelingDeepQNetwork, self).__init__()

        # Context7 enhanced layers with regularization
        self.dense1 = keras.layers.Dense(
            fc1_dims,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name='dense_1'
        )
        self.dropout1 = keras.layers.Dropout(dropout_rate)

        self.dense2 = keras.layers.Dense(
            fc2_dims,
            activation="relu",
            kernel_regularizer=keras.regularizers.l2(l2_reg),
            name='dense_2'
        )
        self.dropout2 = keras.layers.Dropout(dropout_rate)

        # Value and advantage streams
        self.V = keras.layers.Dense(1, activation=None, name='value_stream')
        self.A = keras.layers.Dense(n_actions, activation=None, name='advantage_stream')

    def call(self, state, training=False):
        """Forward pass with Context7 enhanced computation."""
        x = self.dense1(state)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.dropout2(x, training=training)

        # Value and advantage computation
        V = self.V(x)
        A = self.A(x)

        # Context7 enhanced Q-value computation with numerical stability
        A_mean = tf.reduce_mean(A, axis=1, keepdims=True)
        Q = V + (A - A_mean)

        return Q

    def advantage(self, state):
        """Get advantage values for state."""
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)
        return A


class Context7ReplayBuffer:
    """
    Enhanced Experience Replay Buffer following Context7 patterns.

    Improvements:
    - Larger capacity for better sampling diversity
    - Improved memory management
    - Enhanced sampling strategies
    - Professional error handling
    """

    def __init__(self, max_size=CONTEXT7_REPLAY_CAPACITY, input_shape=(10,)):
        self.mem_size = max_size
        self.mem_cntr = 0

        # Initialize memory arrays with proper types
        self.state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)  # Changed to float32
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

        agent_logger.info(f"Context7 replay buffer initialized with capacity: {max_size:,}")

    def store_transition(self, state, action, reward, state_, done):
        """Store transition with enhanced error handling."""
        try:
            index = self.mem_cntr % self.mem_size

            # Ensure proper shapes
            if len(state.shape) > 1:
                self.state_memory[index] = state.reshape(-1)
            else:
                self.state_memory[index] = state

            if len(state_.shape) > 1:
                self.new_state_memory[index] = state_.reshape(-1)
            else:
                self.new_state_memory[index] = state_

            self.action_memory[index] = action
            self.reward_memory[index] = reward
            self.terminal_memory[index] = done

            self.mem_cntr += 1

        except Exception as e:
            agent_logger.error(f"Error storing transition: {e}")
            raise

    def sample_buffer(self, batch_size=CONTEXT7_BATCH_SIZE):
        """Enhanced sampling with Context7 patterns."""
        max_mem = min(self.mem_cntr, self.mem_size)

        if max_mem < batch_size:
            agent_logger.warning(f"Insufficient samples in buffer: {max_mem} < {batch_size}")
            batch_size = max_mem

        # Use more efficient sampling
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal

    def __len__(self):
        """Get current buffer size."""
        return min(self.mem_cntr, self.mem_size)


class Context7DDQNAgent:
    """
    Enhanced DDQN Agent following Context7 best practices.

    Improvements:
    - Professional hyperparameter defaults
    - Enhanced epsilon scheduling
    - Improved experience replay
    - Better performance tracking
    - Context7 compatible interface
    """

    def __init__(self,
                 state_dim,
                 num_actions,
                 learning_rate=CONTEXT7_LEARNING_RATE,
                 gamma=CONTEXT7_GAMMA,
                 epsilon_start=CONTEXT7_EPSILON_START,
                 epsilon_end=CONTEXT7_EPSILON_END,
                 epsilon_decay_steps=CONTEXT7_EPSILON_DECAY_STEPS,
                 epsilon_exponential_decay=CONTEXT7_EPSILON_EXPONENTIAL_DECAY,
                 replay_capacity=CONTEXT7_REPLAY_CAPACITY,
                 architecture=CONTEXT7_ARCHITECTURE,
                 l2_reg=CONTEXT7_L2_REG,
                 tau=CONTEXT7_TAU,
                 batch_size=CONTEXT7_BATCH_SIZE,
                 model_name='context7_ddqn'):

        # Store Context7 parameters
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.architecture = architecture
        self.l2_reg = l2_reg
        self.batch_size = batch_size
        self.tau = tau
        self.model_name = model_name

        # Enhanced epsilon scheduling
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.epsilon_decay = (epsilon_start - epsilon_end) / epsilon_decay_steps
        self.epsilon_exponential_decay = epsilon_exponential_decay
        self.epsilon_history = []

        # Experience replay
        self.experience = Context7ReplayBuffer(replay_capacity, (state_dim,))

        # Networks with Context7 architecture
        self.online_network = self._build_model(trainable=True)
        self.target_network = self._build_model(trainable=False)
        self.update_target()

        # Training tracking
        self.total_steps = 0
        self.train_steps = 0
        self.episodes = 0
        self.episode_length = 0
        self.train_episodes = 0
        self.steps_per_episode = []
        self.episode_reward = 0
        self.rewards_history = []
        self.losses = []
        self.train = True

        # Context7 performance indices
        self.idx = tf.range(batch_size)

        # Backward compatibility attributes
        self.action = [i for i in range(num_actions)]
        self.eps_dec = self.epsilon_decay
        self.eps_min = epsilon_end
        self.model_file = model_name
        self.replace = tau
        self.learn_step_counter = 0
        self.memory = self.experience  # Alias for backward compatibility

        agent_logger.info("Context7 DDQN Agent initialized:")
        agent_logger.info(f"  State dim: {state_dim}, Actions: {num_actions}")
        agent_logger.info(f"  Architecture: {architecture}")
        agent_logger.info(f"  Learning rate: {learning_rate}")
        agent_logger.info(f"  Replay capacity: {replay_capacity:,}")
        agent_logger.info(f"  Batch size: {batch_size}")

    def _build_model(self, trainable=True):
        """Build Context7 enhanced network."""
        model = Context7DuelingDeepQNetwork(
            fc1_dims=self.architecture[0],
            fc2_dims=self.architecture[1],
            n_actions=self.num_actions,
            l2_reg=self.l2_reg
        )

        # Compile with Context7 optimizer settings
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss='mean_squared_error'
        )

        # Set trainable status
        for layer in model.layers:
            layer.trainable = trainable

        return model

    def update_target(self):
        """Update target network weights."""
        self.target_network.set_weights(self.online_network.get_weights())
        agent_logger.debug("Target network updated")

    def epsilon_greedy_policy(self, state):
        """Context7 enhanced epsilon-greedy action selection."""
        self.total_steps += 1

        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.num_actions)
            agent_logger.debug(f"Random action: {action} (ε={self.epsilon:.3f})")
        else:
            # Ensure state has correct shape
            if len(state.shape) == 1:
                state = state.reshape(1, -1)

            q_values = self.online_network.predict(state, verbose=0)
            action = np.argmax(q_values, axis=1)[0]
            agent_logger.debug(f"Greedy action: {action}, Q-values: {q_values[0]}")

        return action

    def memorize_transition(self, state, action, reward, next_state, not_done):
        """Store transition with Context7 enhanced tracking."""
        if not_done:
            self.episode_reward += reward
            self.episode_length += 1
        else:
            # Episode completed - update epsilon and tracking
            if self.train:
                self._update_epsilon()

            self.episodes += 1
            self.rewards_history.append(self.episode_reward)
            self.steps_per_episode.append(self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0

            agent_logger.debug(f"Episode {self.episodes} completed, reward: {self.rewards_history[-1]:.2f}")

        # Store in experience replay
        self.experience.store_transition(state, action, reward, next_state, not not_done)

    def _update_epsilon(self):
        """Context7 enhanced epsilon decay."""
        if self.episodes < self.epsilon_decay_steps:
            self.epsilon -= self.epsilon_decay
        else:
            self.epsilon *= self.epsilon_exponential_decay

        self.epsilon = max(self.epsilon, self.epsilon_end)
        self.epsilon_history.append(self.epsilon)

    def experience_replay(self):
        """Context7 enhanced experience replay with DDQN."""
        if len(self.experience) < self.batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.experience.sample_buffer(self.batch_size)

        # Convert to tensors
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        dones = tf.convert_to_tensor(dones, dtype=tf.bool)

        # DDQN: Use online network to select actions, target network to evaluate
        next_q_values = self.online_network(next_states)
        best_actions = tf.argmax(next_q_values, axis=1)

        next_q_values_target = self.target_network(next_states)
        target_q_values = tf.gather_nd(
            next_q_values_target,
            tf.stack((self.idx, tf.cast(best_actions, tf.int32)), axis=1)
        )

        # Compute targets
        targets = rewards + tf.where(dones, 0.0, self.gamma * target_q_values)

        # Forward pass and compute loss
        with tf.GradientTape() as tape:
            q_values = self.online_network(states, training=True)
            q_values_selected = tf.gather_nd(
                q_values,
                tf.stack((self.idx, actions), axis=1)
            )
            loss = tf.reduce_mean(tf.square(targets - q_values_selected))

        # Backward pass
        gradients = tape.gradient(loss, self.online_network.trainable_variables)
        self.online_network.optimizer.apply_gradients(
            zip(gradients, self.online_network.trainable_variables)
        )

        self.losses.append(loss.numpy())
        self.train_steps += 1

        # Update target network
        if self.total_steps % self.tau == 0:
            self.update_target()

        agent_logger.debug(f"Experience replay completed, loss: {loss:.4f}")

    # Backward compatibility methods
    def store_transition(self, state, action, reward, new_state, done):
        """Backward compatibility method."""
        self.experience.store_transition(state, action, reward, new_state, done)

    def choose_actions(self, observation):
        """Backward compatibility method with original name."""
        return self.choose_action(observation)

    def choose_action(self, observation, evaluate=False):
        """Context7 compatible action selection."""
        if evaluate:
            # Evaluation mode - no epsilon
            if len(observation.shape) == 1:
                observation = observation.reshape(1, -1)
            q_values = self.online_network.predict(observation, verbose=0)
            return np.argmax(q_values, axis=1)[0]
        else:
            return self.epsilon_greedy_policy(observation)

    def learn(self):
        """Context7 enhanced learning with backward compatibility."""
        if len(self.experience) < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.update_target()

        # Use Context7 experience replay
        self.experience_replay()

        # Update epsilon (backward compatibility)
        self.epsilon = max(self.epsilon - self.eps_dec, self.eps_min)
        self.learn_step_counter += 1

    def save_model(self, path=None):
        """Save model with backward compatibility."""
        if path is None:
            path = self.model_file
        self.online_network.save(path, save_format='tf')
        agent_logger.info(f"Model saved to {path}")

    def load_model(self, path=None):
        """Load model with backward compatibility."""
        if path is None:
            path = self.model_file
        try:
            self.online_network = load_model(path)
            agent_logger.info(f"Model loaded from {path}")
        except Exception as e:
            agent_logger.error(f"Failed to load model: {e}")
            raise

    def save_models(self, directory="models/context7_ddqn"):
        """Save models with Context7 naming convention."""
        import os
        os.makedirs(directory, exist_ok=True)

        self.online_network.save_weights(f"{directory}/{self.model_name}_online")
        self.target_network.save_weights(f"{directory}/{self.model_name}_target")
        agent_logger.info(f"Models saved to {directory}")

    def load_models(self, directory="models/context7_ddqn"):
        """Load models with Context7 naming convention."""
        try:
            self.online_network.load_weights(f"{directory}/{self.model_name}_online")
            self.target_network.load_weights(f"{directory}/{self.model_name}_target")
            agent_logger.info(f"Models loaded from {directory}")
        except Exception as e:
            agent_logger.error(f"Failed to load models: {e}")
            raise

    def get_performance_metrics(self):
        """Get Context7 performance metrics."""
        return {
            'episodes': self.episodes,
            'total_steps': self.total_steps,
            'train_steps': self.train_steps,
            'epsilon': self.epsilon,
            'avg_reward_100': np.mean(self.rewards_history[-100:]) if self.rewards_history else 0,
            'avg_reward_10': np.mean(self.rewards_history[-10:]) if self.rewards_history else 0,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'replay_buffer_size': len(self.experience)
        }


# Backward compatibility aliases
Agent = Context7DDQNAgent
DuelingDeepQNetwork = Context7DuelingDeepQNetwork
Replaybuffer = Context7ReplayBuffer
