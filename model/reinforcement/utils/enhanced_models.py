"""
Enhanced Model Utilities for Reinforcement Learning.

Provides improved neural network architectures, optimization patterns,
and model management utilities based on latest research.
"""

import os
from typing import Dict, List, Optional

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.layers import BatchNormalization, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    HAS_TENSORFLOW = True
except ImportError:
    # TensorFlow not available - define placeholder classes
    tf = keras = EarlyStopping = ReduceLROnPlateau = object
    BatchNormalization = Dense = Dropout = Adam = l2 = object
    HAS_TENSORFLOW = False

import util.loggers as loggers

logger = loggers.setup_loggers()
model_logger = logger['model']

# Enhanced Model Constants (from latest research)
OPTIMAL_LEARNING_RATE = 0.0001
OPTIMAL_LAYER_SIZES = (256, 256)
OPTIMAL_L2_REG = 1e-6
OPTIMAL_DROPOUT_RATE = 0.1
OPTIMAL_BATCH_SIZE = 32

class EnhancedNetworkBuilder:
    """
    Enhanced neural network builder with improved architectures.

    Provides professional network architectures optimized for
    trading reinforcement learning applications.
    """

    @staticmethod
    def build_dqn_network(state_dim: int,
                         num_actions: int,
                         hidden_sizes: tuple = OPTIMAL_LAYER_SIZES,
                         learning_rate: float = OPTIMAL_LEARNING_RATE,
                         l2_reg: float = OPTIMAL_L2_REG,
                         dropout_rate: float = OPTIMAL_DROPOUT_RATE) -> keras.Model:
        """
        Build enhanced Deep Q-Network.

        Args:
            state_dim: Input state dimension
            num_actions: Number of output actions
            hidden_sizes: Tuple of hidden layer sizes
            learning_rate: Learning rate for optimizer
            l2_reg: L2 regularization strength
            dropout_rate: Dropout rate for regularization

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            Dense(hidden_sizes[0],
                  activation='relu',
                  input_dim=state_dim,
                  kernel_regularizer=l2(l2_reg),
                  name='hidden_1'),
            BatchNormalization(),
            Dropout(dropout_rate),

            Dense(hidden_sizes[1],
                  activation='relu',
                  kernel_regularizer=l2(l2_reg),
                  name='hidden_2'),
            BatchNormalization(),
            Dropout(dropout_rate),

            Dense(num_actions,
                  activation='linear',
                  name='q_values')
        ])

        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        model_logger.info(f"Enhanced DQN network built: {state_dim} -> {hidden_sizes} -> {num_actions}")
        return model

    @staticmethod
    def build_dueling_dqn_network(state_dim: int,
                                 num_actions: int,
                                 hidden_sizes: tuple = OPTIMAL_LAYER_SIZES,
                                 learning_rate: float = OPTIMAL_LEARNING_RATE,
                                 l2_reg: float = OPTIMAL_L2_REG,
                                 dropout_rate: float = OPTIMAL_DROPOUT_RATE) -> keras.Model:
        """
        Build enhanced Dueling Deep Q-Network.

        Implements the dueling architecture with separate value and advantage streams.
        """
        input_layer = keras.Input(shape=(state_dim,))

        # Shared layers
        x = Dense(hidden_sizes[0], activation='relu',
                 kernel_regularizer=l2(l2_reg))(input_layer)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        x = Dense(hidden_sizes[1], activation='relu',
                 kernel_regularizer=l2(l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(dropout_rate)(x)

        # Value stream
        value_stream = Dense(hidden_sizes[1] // 2, activation='relu',
                           kernel_regularizer=l2(l2_reg))(x)
        value_stream = Dense(1, activation='linear', name='state_value')(value_stream)

        # Advantage stream
        advantage_stream = Dense(hidden_sizes[1] // 2, activation='relu',
                               kernel_regularizer=l2(l2_reg))(x)
        advantage_stream = Dense(num_actions, activation='linear', name='advantages')(advantage_stream)

        # Combine streams (Q = V + A - mean(A))
        q_values = keras.layers.Lambda(
            lambda x: x[0] + (x[1] - tf.reduce_mean(x[1], axis=1, keepdims=True)),
            name='q_values'
        )([value_stream, advantage_stream])

        model = keras.Model(inputs=input_layer, outputs=q_values)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        model_logger.info(f"Enhanced Dueling DQN network built: {state_dim} -> {hidden_sizes} -> {num_actions}")
        return model

    @staticmethod
    def build_actor_network(state_dim: int,
                           num_actions: int,
                           hidden_sizes: tuple = OPTIMAL_LAYER_SIZES,
                           learning_rate: float = OPTIMAL_LEARNING_RATE,
                           l2_reg: float = OPTIMAL_L2_REG,
                           dropout_rate: float = OPTIMAL_DROPOUT_RATE,
                           action_bound: float = 1.0) -> keras.Model:
        """
        Build enhanced Actor network for policy-based methods.

        Args:
            state_dim: Input state dimension
            num_actions: Number of continuous actions
            hidden_sizes: Tuple of hidden layer sizes
            learning_rate: Learning rate for optimizer
            l2_reg: L2 regularization strength
            dropout_rate: Dropout rate
            action_bound: Action space bounds

        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            Dense(hidden_sizes[0],
                  activation='relu',
                  input_dim=state_dim,
                  kernel_regularizer=l2(l2_reg),
                  name='actor_hidden_1'),
            BatchNormalization(),
            Dropout(dropout_rate),

            Dense(hidden_sizes[1],
                  activation='relu',
                  kernel_regularizer=l2(l2_reg),
                  name='actor_hidden_2'),
            BatchNormalization(),
            Dropout(dropout_rate),

            Dense(num_actions,
                  activation='tanh',
                  kernel_initializer='random_uniform',
                  name='actions')
        ])

        # Scale actions to action bounds
        if action_bound != 1.0:
            scaled_output = keras.layers.Lambda(
                lambda x: x * action_bound,
                name='scaled_actions'
            )(model.output)
            model = keras.Model(inputs=model.input, outputs=scaled_output)

        model.compile(optimizer=Adam(learning_rate=learning_rate))

        model_logger.info(f"Enhanced Actor network built: {state_dim} -> {hidden_sizes} -> {num_actions}")
        return model

    @staticmethod
    def build_critic_network(state_dim: int,
                           action_dim: int,
                           hidden_sizes: tuple = OPTIMAL_LAYER_SIZES,
                           learning_rate: float = OPTIMAL_LEARNING_RATE,
                           l2_reg: float = OPTIMAL_L2_REG,
                           dropout_rate: float = OPTIMAL_DROPOUT_RATE) -> keras.Model:
        """
        Build enhanced Critic network for actor-critic methods.

        Args:
            state_dim: Input state dimension
            action_dim: Input action dimension
            hidden_sizes: Tuple of hidden layer sizes
            learning_rate: Learning rate for optimizer
            l2_reg: L2 regularization strength
            dropout_rate: Dropout rate

        Returns:
            Compiled Keras model
        """
        # State input
        state_input = keras.Input(shape=(state_dim,), name='state_input')
        state_h1 = Dense(hidden_sizes[0], activation='relu',
                        kernel_regularizer=l2(l2_reg))(state_input)
        state_h1 = BatchNormalization()(state_h1)

        # Action input
        action_input = keras.Input(shape=(action_dim,), name='action_input')
        action_h1 = Dense(hidden_sizes[0], activation='relu',
                         kernel_regularizer=l2(l2_reg))(action_input)

        # Concatenate state and action
        concat = keras.layers.Concatenate()([state_h1, action_h1])
        concat_h1 = Dense(hidden_sizes[1], activation='relu',
                         kernel_regularizer=l2(l2_reg))(concat)
        concat_h1 = BatchNormalization()(concat_h1)
        concat_h1 = Dropout(dropout_rate)(concat_h1)

        # Output Q-value
        q_value = Dense(1, activation='linear', name='q_value')(concat_h1)

        model = keras.Model(inputs=[state_input, action_input], outputs=q_value)
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        model_logger.info(f"Enhanced Critic network built: ({state_dim}, {action_dim}) -> {hidden_sizes} -> 1")
        return model


class ModelManager:
    """
    Enhanced model management with professional patterns.

    Handles model saving, loading, checkpointing, and versioning
    with improved error handling and logging.
    """

    def __init__(self, model_dir: str = 'models/enhanced_rl'):
        self.model_dir = model_dir
        self.ensure_model_dir()

    def ensure_model_dir(self):
        """Ensure model directory exists."""
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir, exist_ok=True)
            model_logger.info(f"Created model directory: {self.model_dir}")

    def save_model(self, model: keras.Model, name: str, episode: Optional[int] = None):
        """
        Save model with enhanced naming and versioning.

        Args:
            model: Keras model to save
            name: Base name for the model
            episode: Optional episode number for versioning
        """
        try:
            if episode is not None:
                filename = f"{name}_ep_{episode:06d}.h5"
            else:
                filename = f"{name}_latest.h5"

            filepath = os.path.join(self.model_dir, filename)
            model.save(filepath)
            model_logger.info(f"Model saved: {filepath}")

            # Also save weights separately for flexibility
            weights_path = filepath.replace('.h5', '_weights.h5')
            model.save_weights(weights_path)
            model_logger.info(f"Weights saved: {weights_path}")

        except Exception as e:
            model_logger.error(f"Failed to save model {name}: {e}")
            raise

    def load_model(self, name: str, episode: Optional[int] = None) -> keras.Model:
        """
        Load model with enhanced error handling.

        Args:
            name: Base name of the model
            episode: Optional specific episode to load

        Returns:
            Loaded Keras model
        """
        try:
            if episode is not None:
                filename = f"{name}_ep_{episode:06d}.h5"
            else:
                filename = f"{name}_latest.h5"

            filepath = os.path.join(self.model_dir, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Model file not found: {filepath}")

            model = keras.models.load_model(filepath)
            model_logger.info(f"Model loaded: {filepath}")
            return model

        except Exception as e:
            model_logger.error(f"Failed to load model {name}: {e}")
            raise

    def load_weights(self, model: keras.Model, name: str, episode: Optional[int] = None):
        """
        Load model weights with enhanced error handling.

        Args:
            model: Model to load weights into
            name: Base name of the weights file
            episode: Optional specific episode to load
        """
        try:
            if episode is not None:
                filename = f"{name}_ep_{episode:06d}_weights.h5"
            else:
                filename = f"{name}_latest_weights.h5"

            filepath = os.path.join(self.model_dir, filename)

            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Weights file not found: {filepath}")

            model.load_weights(filepath)
            model_logger.info(f"Weights loaded: {filepath}")

        except Exception as e:
            model_logger.error(f"Failed to load weights {name}: {e}")
            raise

    def list_saved_models(self, name_pattern: Optional[str] = None) -> List[str]:
        """
        List all saved models in the model directory.

        Args:
            name_pattern: Optional pattern to filter model names

        Returns:
            List of model filenames
        """
        try:
            files = os.listdir(self.model_dir)
            model_files = [f for f in files if f.endswith('.h5') and '_weights' not in f]

            if name_pattern:
                model_files = [f for f in model_files if name_pattern in f]

            return sorted(model_files)

        except Exception as e:
            model_logger.error(f"Failed to list models: {e}")
            return []

    def cleanup_old_models(self, name: str, keep_latest: int = 5):
        """
        Clean up old model files, keeping only the latest N versions.

        Args:
            name: Base name of models to clean up
            keep_latest: Number of latest models to keep
        """
        try:
            model_files = self.list_saved_models(name)
            episode_models = [f for f in model_files if '_ep_' in f and name in f]

            if len(episode_models) <= keep_latest:
                return

            # Sort by episode number and remove oldest
            episode_models.sort()
            to_remove = episode_models[:-keep_latest]

            for filename in to_remove:
                filepath = os.path.join(self.model_dir, filename)
                weights_path = filepath.replace('.h5', '_weights.h5')

                # Remove model file
                if os.path.exists(filepath):
                    os.remove(filepath)
                    model_logger.info(f"Removed old model: {filepath}")

                # Remove weights file
                if os.path.exists(weights_path):
                    os.remove(weights_path)
                    model_logger.info(f"Removed old weights: {weights_path}")

        except Exception as e:
            model_logger.error(f"Failed to cleanup models: {e}")


# Enhanced training callbacks
def get_enhanced_callbacks(patience: int = 20,
                          min_delta: float = 0.001,
                          reduce_lr_patience: int = 10) -> List[keras.callbacks.Callback]:
    """
    Get enhanced training callbacks for improved training.

    Args:
        patience: Patience for early stopping
        min_delta: Minimum change to qualify as improvement
        reduce_lr_patience: Patience for learning rate reduction

    Returns:
        List of Keras callbacks
    """
    callbacks = [
        EarlyStopping(
            monitor='loss',
            patience=patience,
            min_delta=min_delta,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='loss',
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=1e-7,
            verbose=1
        )
    ]

    return callbacks


# Utility functions
def get_model_summary_info(model: keras.Model) -> Dict:
    """
    Get comprehensive model information.

    Args:
        model: Keras model to analyze

    Returns:
        Dictionary with model information
    """
    try:
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        non_trainable_params = total_params - trainable_params

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'non_trainable_parameters': non_trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'input_shape': model.input_shape,
            'output_shape': model.output_shape,
            'num_layers': len(model.layers)
        }

    except Exception as e:
        model_logger.error(f"Failed to get model info: {e}")
        return {}


def compare_models(model1: keras.Model, model2: keras.Model) -> Dict:
    """
    Compare two Keras models.

    Args:
        model1: First model
        model2: Second model

    Returns:
        Comparison dictionary
    """
    info1 = get_model_summary_info(model1)
    info2 = get_model_summary_info(model2)

    return {
        'model1_info': info1,
        'model2_info': info2,
        'parameter_difference': info2.get('total_parameters', 0) - info1.get('total_parameters', 0),
        'size_difference_mb': info2.get('model_size_mb', 0) - info1.get('model_size_mb', 0)
    }
