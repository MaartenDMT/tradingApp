from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l2
from tensorflow.keras import Add, Input, Model, Sequential
from tensorflow.keras.layers import (LSTM, Conv1D, Dense, Flatten,
                                     GlobalAveragePooling1D, MaxPool1D)

import util.loggers as loggers
from model.reinforcement.agents.agent_utils import transformer_block

logger = loggers.setup_loggers()
rl_logger = logger['rl']


class ModelBuilder:
    def __init__(self, env, opt, loss, learning_rate):
        self.env = env
        self.osn = env.observation_space.shape[0]
        self.opt = opt
        self.loss = loss
        self.learning_rate = learning_rate

    def get_model(self, hidden_units, dropout, m_activation, chosen_model):
        models = {
            "Standard_Model": self._build_model(hidden_units, dropout, m_activation),
            "Dense_Model": self.base_dense_model(m_activation, dropout),
            "LSTM_Model": self.base_lstm_model(m_activation, dropout),
            "CONV1D_LSTM_Model": self.base_conv1d_lstm_model(m_activation, dropout),
            "build_resnet_model": self.build_resnet_model(m_activation, dropout),
            "base_conv1d_model": self.base_conv1d_model(m_activation, dropout),
            "base_transformer_model": self.base_transformer_model(m_activation, dropout),
        }

        return models[chosen_model]

    def _build_model(self, hidden_units, dropout, m_activation):
        rl_logger.info("Standard Model loaded.")
        model = Sequential()
        model.add(Dense(64, input_dim=self.osn,
                  activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(Dense(
            hidden_units, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dense(
            self.env_actions, activation=m_activation))
        model.compile(loss=self.loss, optimizer=self.opt(
            learning_rate=self.learning_rate), metrics=['accuracy'])
        rl_logger.info(
            f"Model built with action space {self.env.action_space.n}.")
        return model

    def base_dense_model(self, m_activation, dropout):
        rl_logger.info("Dense Model loaded.")
        base_model = Sequential()
        base_model.add(Dense(
            128, input_dim=self.osn, activation='relu'))
        base_model.add(BatchNormalization())
        base_model.add(Dropout(dropout))
        base_model.add(Dense(64, activation='relu'))
        base_model.add(BatchNormalization())
        base_model.add(Dense(32, activation='relu'))
        base_model.add(BatchNormalization())
        base_model.add(Dense(
            self.env_actions, activation=m_activation))
        base_model.compile(
            optimizer=self.opt(learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])

        return base_model

    def base_lstm_model(self, m_activation, dropout):
        rl_logger.info("LSTM Model loaded.")
        base_model = Sequential()
        base_model.add(LSTM(100, activation='relu', input_shape=(
            self.osn, 1), return_sequences=True))
        base_model.add(Dropout(dropout))
        base_model.add(LSTM(100, activation='relu'))
        base_model.add(Dense(
            self.env_actions, activation=m_activation))
        base_model.compile(
            optimizer=self.opt(learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])

        return base_model

    def base_conv1d_lstm_model(self, m_activation, dropout):
        rl_logger.info("CONV1D Model loaded.")
        base_model = Sequential()
        base_model.add(Conv1D(filters=64, kernel_size=2,
                       activation='relu', input_shape=(self.osn, 1)))
        base_model.add(MaxPool1D(pool_size=1))
        base_model.add(LSTM(
            100, activation='relu', return_sequences=True))
        base_model.add(Dropout(dropout))
        base_model.add(LSTM(100, activation='relu'))
        base_model.add(Dense(
            self.env_actions, activation=m_activation))
        base_model.compile(
            optimizer=self.opt(learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])

        return base_model

    def build_resnet_model(self, m_activation, dropout):
        rl_logger.info("ResNet Model loaded.")

        inputs = Input(shape=(self.osn,))
        x = Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        # Residual block
        residual = x
        x = Dense(128, activation='relu')(
            x)  # Change this to 128 units
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = Dense(128, activation='relu')(
            x)  # Change this to 128 units
        x = Add()([x, residual])

        x = BatchNormalization()(x)
        outputs = Dense(
            self.env_actions, activation=m_activation)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.opt(
            learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])
        return model

    def base_conv1d_model(self, m_activation, dropout):
        rl_logger.info("Simple CONV1D Model loaded.")

        base_model = Sequential()
        base_model.add(Conv1D(
            filters=64, kernel_size=2, activation='relu', input_shape=(self.osn, 1)))
        base_model.add(MaxPool1D(pool_size=2))
        base_model.add(Flatten())
        base_model.add(Dense(100, activation='relu'))
        base_model.add(Dropout(dropout))
        base_model.add(Dense(
            self.env_actions, activation=m_activation))

        base_model.compile(optimizer=self.opt(
            learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])
        return base_model

    def base_transformer_model(self, m_activation, dropout=0.1, d_model=128, num_heads=4):
        rl_logger.info("Transformer Model loaded.")

        inputs = Input(shape=(self.osn, 1))
        x = Dense(d_model)(inputs)

        # Embed the sequence into the d_model space
        x = Dense(d_model)(x)

        # Add positional encoding if needed. Skipped in this example for brevity.
        # x = positional_encoding(x, d_model)

        x = transformer_block(d_model, num_heads, dropout)(x)
        x = GlobalAveragePooling1D()(x)
        outputs = Dense(
            self.env_actions, activation=m_activation)(x)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.opt(
            learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])
        return model
