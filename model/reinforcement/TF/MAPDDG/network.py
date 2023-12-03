import tensorflow.keras as keras
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l2
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (LSTM, Add, Concatenate, Conv1D, Dense,
                                     Flatten, GlobalAveragePooling1D,
                                     MaxPool1D)

from util.agent_utils import transformer_block


class CentralizedCriticNetwork(keras.Model):
    def __init__(self, num_agents, fc1_dims=128, fc2_dims=128, l2_reg=0.01):
        super(CentralizedCriticNetwork, self).__init__()

        # Define layers for the Centralized Critic Network with L2 regularization
        self.fc1 = Dense(fc1_dims, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))
        self.fc2 = Dense(fc2_dims, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))
        self.output_layer = Dense(num_agents, activation=None)

    def call(self, inputs):
        joint_state, joint_action = inputs

        # Flatten inputs if they're not already flat
        joint_state_flattened = Flatten()(joint_state)
        joint_action_flattened = Flatten()(joint_action)

        # Concatenate joint state and joint action inputs
        concatenated = Concatenate()(
            [joint_state_flattened, joint_action_flattened])

        # Pass through dense layers
        x = self.fc1(concatenated)
        x = self.fc2(x)

        return self.output_layer(x)


class ActorNetwork(keras.Model):
    def __init__(self, action_dimension, fc1_dims=128, fc2_dims=128, l2_reg=0.01):
        super(ActorNetwork, self).__init__()
        # First fully connected layer with L2 regularization
        self.fc1 = Dense(fc1_dims, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))
        # Second fully connected layer with L2 regularization
        self.fc2 = Dense(fc2_dims, activation='relu',
                         kernel_regularizer=regularizers.l2(l2_reg))
        # Output layer - replace 'softmax' with 'linear' for continuous action spaces
        self.output_layer = Dense(
            action_dimension, activation='softmax')

    def call(self, state_input):
        # Flatten the state input if it's not already 1D
        state_flattened = Flatten()(state_input)
        x = self.fc1(state_flattened)
        x = self.fc2(x)
        return self.output_layer(x)


class StandardActorModel(keras.Model):
    def __init__(self, state_dimension, action_dimension, hidden_units, dropout):
        super(StandardActorModel, self).__init__()
        self.d1 = Dense(64, input_dim=state_dimension,
                        activation='relu', kernel_regularizer=l2(0.01))
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(dropout)
        self.d2 = Dense(hidden_units, activation='relu',
                        kernel_regularizer=l2(0.01))
        self.batch_norm2 = BatchNormalization()
        self.output_layer = Dense(action_dimension, activation='softmax')

    def call(self, state_input):
        x = self.d1(state_input)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.d2(x)
        x = self.batch_norm2(x)
        return self.output_layer(x)


class ActorDenseModel(keras.Model):
    def __init__(self, action_dimension):
        super(ActorDenseModel, self).__init__()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(64, activation='relu')
        self.d3 = Dense(32, activation='relu')
        self.out = Dense(
            action_dimension, activation='softmax')

    def call(self, state):
        x = self.d1(state)
        x = self.d2(x)
        x = self.d3(x)
        return self.out(x)


class LSTMActorModel(keras.Model):
    def __init__(self, state_dimension, action_dimension, dropout):
        super(LSTMActorModel, self).__init__()
        self.lstm1 = LSTM(100, activation='relu', input_shape=(
            state_dimension, 1), return_sequences=True)
        self.dropout1 = Dropout(dropout)
        self.lstm2 = LSTM(100, activation='relu')
        self.output_layer = Dense(action_dimension, activation='softmax')

    def call(self, state_input):
        x = self.lstm1(state_input)
        x = self.dropout1(x)
        x = self.lstm2(x)
        return self.output_layer(x)


class Conv1DLSTMActorModel(keras.Model):
    def __init__(self, state_dimension, action_dimension, dropout):
        super(Conv1DLSTMActorModel, self).__init__()
        self.conv1d = Conv1D(filters=64, kernel_size=2,
                             activation='relu', input_shape=(state_dimension, 1))
        self.max_pool = MaxPool1D(pool_size=1)
        self.lstm1 = LSTM(100, activation='relu', return_sequences=True)
        self.dropout1 = Dropout(dropout)
        self.lstm2 = LSTM(100, activation='relu')
        self.output_layer = Dense(action_dimension, activation='softmax')

    def call(self, state_input):
        x = self.conv1d(state_input)
        x = self.max_pool(x)
        x = self.lstm1(x)
        x = self.dropout1(x)
        x = self.lstm2(x)
        return self.output_layer(x)


class SimpleConv1DActorModel(keras.Model):
    def __init__(self, state_dimension, action_dimension, dropout):
        super(SimpleConv1DActorModel, self).__init__()
        self.conv1d = Conv1D(filters=64, kernel_size=2,
                             activation='relu', input_shape=(state_dimension, 1))
        self.max_pool = MaxPool1D(pool_size=1)
        self.flatten = Flatten()
        self.d1 = Dense(100, activation='relu')
        self.dropout1 = Dropout(dropout)
        self.output_layer = Dense(action_dimension, activation='softmax')

    def call(self, state_input):
        x = self.conv1d(state_input)
        x = self.max_pool(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.dropout1(x)
        return self.output_layer(x)


class ResNetActorModel(keras.Model):
    def __init__(self, action_dimension, dropout):
        super(ResNetActorModel, self).__init__()
        self.d1 = Dense(128, activation='relu')
        self.batch_norm1 = BatchNormalization()
        self.dropout1 = Dropout(dropout)
        self.residual_block = self.create_residual_block(dropout)
        self.output_layer = Dense(action_dimension, activation='softmax')

    def call(self, state_input):
        x = self.d1(state_input)
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.residual_block(x)
        return self.output_layer(x)

    def create_residual_block(self, dropout):
        def block(x):
            residual = x
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
            x = Dense(128, activation='relu')(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout)(x)
            x = Add()([x, residual])
            x = BatchNormalization()(x)
            return x
        return block


class TransformerActorModel(keras.Model):
    def __init__(self, action_dimension, dropout=0.1, d_model=128, num_heads=4):
        super(TransformerActorModel, self).__init__()
        self.d1 = Dense(d_model)
        self.transformer_block = transformer_block(d_model, num_heads, dropout)
        self.global_pool = GlobalAveragePooling1D()
        self.output_layer = Dense(action_dimension, activation='softmax')

    def call(self, state_input):
        x = self.d1(state_input)
        x = self.transformer_block(x)
        x = self.global_pool(x)
        return self.output_layer(x)
