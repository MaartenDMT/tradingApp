"""
Context7 Enhanced Neural Network Utilities for Trading RL.

This module provides professional neural network architectures and utilities
following Context7 best practices:
- Enhanced network architectures with proper regularization
- Professional initialization strategies
- Advanced activation functions and normalization
- Trading-specific network components
"""

from typing import Optional, Tuple

import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers
from tensorflow.keras.optimizers import Adam

import util.loggers as loggers

logger = loggers.setup_loggers()
rl_logger = logger['rl']

# Context7 Network Constants
CONTEXT7_HIDDEN_UNITS = (256, 256)
CONTEXT7_DROPOUT_RATE = 0.1
CONTEXT7_L2_REGULARIZATION = 1e-6
CONTEXT7_BATCH_NORM = True
CONTEXT7_ACTIVATION = 'swish'
CONTEXT7_KERNEL_INITIALIZER = 'he_normal'
CONTEXT7_LEARNING_RATE = 3e-4


class Context7NetworkBuilder:
    """
    Professional network builder following Context7 patterns.
    """

    @staticmethod
    def create_dense_block(
        inputs: tf.Tensor,
        units: int,
        activation: str = CONTEXT7_ACTIVATION,
        dropout_rate: float = CONTEXT7_DROPOUT_RATE,
        l2_reg: float = CONTEXT7_L2_REGULARIZATION,
        batch_norm: bool = CONTEXT7_BATCH_NORM,
        name_prefix: str = "dense"
    ) -> tf.Tensor:
        """
        Create a Context7 dense block with regularization.

        Args:
            inputs: Input tensor
            units: Number of units
            activation: Activation function
            dropout_rate: Dropout rate
            l2_reg: L2 regularization strength
            batch_norm: Whether to use batch normalization
            name_prefix: Name prefix for layers

        Returns:
            Output tensor
        """
        x = layers.Dense(
            units,
            activation=None,
            kernel_initializer=CONTEXT7_KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(l2_reg),
            name=f"{name_prefix}_dense"
        )(inputs)

        if batch_norm:
            x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)

        x = layers.Activation(activation, name=f"{name_prefix}_activation")(x)

        if dropout_rate > 0:
            x = layers.Dropout(dropout_rate, name=f"{name_prefix}_dropout")(x)

        return x

    @staticmethod
    def create_value_head(
        inputs: tf.Tensor,
        name: str = "value_head"
    ) -> tf.Tensor:
        """Create Context7 value estimation head."""
        return layers.Dense(
            1,
            activation='linear',
            kernel_initializer='zeros',
            name=name
        )(inputs)

    @staticmethod
    def create_action_head(
        inputs: tf.Tensor,
        num_actions: int,
        name: str = "action_head"
    ) -> tf.Tensor:
        """Create Context7 action selection head."""
        return layers.Dense(
            num_actions,
            activation='linear',
            kernel_initializer='zeros',
            name=name
        )(inputs)

    @staticmethod
    def create_advantage_head(
        inputs: tf.Tensor,
        num_actions: int,
        name: str = "advantage_head"
    ) -> tf.Tensor:
        """Create Context7 advantage estimation head."""
        return layers.Dense(
            num_actions,
            activation='linear',
            kernel_initializer='zeros',
            name=name
        )(inputs)


class Context7TradingNetwork(Model):
    """
    Professional trading network following Context7 patterns.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_units: Tuple[int, ...] = CONTEXT7_HIDDEN_UNITS,
                 dropout_rate: float = CONTEXT7_DROPOUT_RATE,
                 l2_reg: float = CONTEXT7_L2_REGULARIZATION,
                 network_type: str = "dqn",
                 **kwargs):

        super().__init__(**kwargs)

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.network_type = network_type

        # Build network layers
        self._build_network()

        rl_logger.info(f"Context7 {network_type.upper()} Network created:")
        rl_logger.info(f"  State size: {state_size}")
        rl_logger.info(f"  Action size: {action_size}")
        rl_logger.info(f"  Hidden units: {hidden_units}")
        rl_logger.info(f"  Dropout: {dropout_rate}")
        rl_logger.info(f"  L2 regularization: {l2_reg}")

    def _build_network(self):
        """Build the network architecture."""
        # Hidden layers
        self.hidden_layers = []
        for i, units in enumerate(self.hidden_units):
            self.hidden_layers.append(
                Context7NetworkBuilder.create_dense_block
            )

        # Output heads based on network type
        if self.network_type == "dqn":
            self.q_head = layers.Dense(
                self.action_size,
                activation='linear',
                kernel_initializer='zeros',
                name="q_values"
            )
        elif self.network_type == "dueling_dqn":
            self.value_head = Context7NetworkBuilder.create_value_head
            self.advantage_head = Context7NetworkBuilder.create_advantage_head
        elif self.network_type == "actor":
            self.action_head = layers.Dense(
                self.action_size,
                activation='tanh',
                kernel_initializer='zeros',
                name="actions"
            )
        elif self.network_type == "critic":
            self.value_head = Context7NetworkBuilder.create_value_head

    def call(self, inputs, training=None, mask=None):
        """Forward pass through the network."""
        x = inputs

        # Hidden layers
        for i, units in enumerate(self.hidden_units):
            x = Context7NetworkBuilder.create_dense_block(
                x, units,
                dropout_rate=self.dropout_rate if training else 0.0,
                l2_reg=self.l2_reg,
                name_prefix=f"hidden_{i}"
            )

        # Output based on network type
        if self.network_type == "dqn":
            return self.q_head(x)

        elif self.network_type == "dueling_dqn":
            value = self.value_head(x, name="value")
            advantage = self.advantage_head(x, self.action_size, name="advantage")

            # Dueling architecture: Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
            advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
            q_values = value + advantage - advantage_mean
            return q_values

        elif self.network_type == "actor":
            return self.action_head(x)

        elif self.network_type == "critic":
            return self.value_head(x, name="critic_value")

        else:
            raise ValueError(f"Unknown network type: {self.network_type}")


class Context7ContinuousCritic(Model):
    """
    Professional continuous action critic network for TD3/SAC.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 hidden_units: Tuple[int, ...] = CONTEXT7_HIDDEN_UNITS,
                 dropout_rate: float = CONTEXT7_DROPOUT_RATE,
                 l2_reg: float = CONTEXT7_L2_REGULARIZATION,
                 **kwargs):

        super().__init__(**kwargs)

        self.state_size = state_size
        self.action_size = action_size
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Build network
        self._build_network()

    def _build_network(self):
        """Build critic network for continuous actions."""
        # State processing layers
        self.state_layers = []
        for i, units in enumerate(self.hidden_units[:-1]):
            self.state_layers.append(
                layers.Dense(
                    units,
                    activation=CONTEXT7_ACTIVATION,
                    kernel_initializer=CONTEXT7_KERNEL_INITIALIZER,
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name=f"state_dense_{i}"
                )
            )

        # Action processing
        self.action_dense = layers.Dense(
            self.hidden_units[-1] // 2,
            activation=CONTEXT7_ACTIVATION,
            kernel_initializer=CONTEXT7_KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="action_dense"
        )

        # Combined processing
        self.combined_dense = layers.Dense(
            self.hidden_units[-1],
            activation=CONTEXT7_ACTIVATION,
            kernel_initializer=CONTEXT7_KERNEL_INITIALIZER,
            kernel_regularizer=regularizers.l2(self.l2_reg),
            name="combined_dense"
        )

        # Output
        self.value_output = layers.Dense(
            1,
            activation='linear',
            kernel_initializer='zeros',
            name="q_value"
        )

        # Regularization
        self.dropout = layers.Dropout(self.dropout_rate)
        self.batch_norm = layers.BatchNormalization()

    def call(self, state_action_inputs, training=None, mask=None):
        """Forward pass through critic network."""
        states, actions = state_action_inputs

        # Process state
        x_state = states
        for layer in self.state_layers:
            x_state = layer(x_state)
            if training:
                x_state = self.dropout(x_state, training=training)

        # Process action
        x_action = self.action_dense(actions)

        # Combine state and action
        x_combined = tf.concat([x_state, x_action], axis=1)
        x_combined = self.combined_dense(x_combined)
        x_combined = self.batch_norm(x_combined, training=training)

        if training:
            x_combined = self.dropout(x_combined, training=training)

        # Output Q-value
        q_value = self.value_output(x_combined)
        return q_value


class Context7ContinuousActor(Model):
    """
    Professional continuous action actor network for TD3/SAC.
    """

    def __init__(self,
                 state_size: int,
                 action_size: int,
                 action_bound: float = 1.0,
                 hidden_units: Tuple[int, ...] = CONTEXT7_HIDDEN_UNITS,
                 dropout_rate: float = CONTEXT7_DROPOUT_RATE,
                 l2_reg: float = CONTEXT7_L2_REGULARIZATION,
                 **kwargs):

        super().__init__(**kwargs)

        self.state_size = state_size
        self.action_size = action_size
        self.action_bound = action_bound
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg

        # Build network
        self._build_network()

    def _build_network(self):
        """Build actor network for continuous actions."""
        # Hidden layers
        self.hidden_layers = []
        for i, units in enumerate(self.hidden_units):
            self.hidden_layers.append(
                layers.Dense(
                    units,
                    activation=CONTEXT7_ACTIVATION,
                    kernel_initializer=CONTEXT7_KERNEL_INITIALIZER,
                    kernel_regularizer=regularizers.l2(self.l2_reg),
                    name=f"hidden_{i}"
                )
            )

        # Output layer
        self.action_output = layers.Dense(
            self.action_size,
            activation='tanh',
            kernel_initializer='zeros',
            name="actions"
        )

        # Regularization
        self.dropout = layers.Dropout(self.dropout_rate)
        self.batch_norm_layers = [
            layers.BatchNormalization(name=f"bn_{i}")
            for i in range(len(self.hidden_units))
        ]

    def call(self, inputs, training=None, mask=None):
        """Forward pass through actor network."""
        x = inputs

        # Hidden layers with batch norm and dropout
        for i, (layer, bn) in enumerate(zip(self.hidden_layers, self.batch_norm_layers)):
            x = layer(x)
            x = bn(x, training=training)
            if training:
                x = self.dropout(x, training=training)

        # Output actions
        actions = self.action_output(x)

        # Scale actions to bounds
        scaled_actions = actions * self.action_bound

        return scaled_actions


def create_context7_optimizer(learning_rate: float = CONTEXT7_LEARNING_RATE) -> tf.keras.optimizers.Optimizer:
    """
    Create Context7 optimized Adam optimizer.

    Args:
        learning_rate: Learning rate

    Returns:
        Configured Adam optimizer
    """
    return Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        clipnorm=1.0  # Gradient clipping
    )


def context7_compile_model(model: Model,
                          learning_rate: float = CONTEXT7_LEARNING_RATE,
                          loss: str = 'mse',
                          metrics: Optional[list] = None) -> None:
    """
    Compile model with Context7 professional settings.

    Args:
        model: Keras model to compile
        learning_rate: Learning rate
        loss: Loss function
        metrics: Additional metrics to track
    """
    optimizer = create_context7_optimizer(learning_rate)

    if metrics is None:
        metrics = ['mae']

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=metrics
    )

    rl_logger.info("Model compiled with Context7 settings:")
    rl_logger.info(f"  Optimizer: Adam (lr={learning_rate})")
    rl_logger.info(f"  Loss: {loss}")
    rl_logger.info(f"  Metrics: {metrics}")


# Export main classes and functions
__all__ = [
    'Context7NetworkBuilder',
    'Context7TradingNetwork',
    'Context7ContinuousCritic',
    'Context7ContinuousActor',
    'create_context7_optimizer',
    'context7_compile_model'
]
