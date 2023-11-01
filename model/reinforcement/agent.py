import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import BatchNormalization, Dropout
from keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam

import util.loggers as loggers
from model.reinforcement.agent_extra import (is_new_record,
                                             multi_head_attention, save_model,
                                             save_to_csv, transformer_block)
from util.rl_util import next_available_filename

logger = loggers.setup_loggers()
agent_logger = logger['agent']
rl_logger = logger['rl']

MODEL_PATH = 'data/saved_model'


class DQLAgent:
    def __init__(self, gamma=0.95, hidden_units=24, opt=Adam, learning_rate=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, dropout=0.5,
                 finish=False, env=None, act='argmax', m_activation='linear', actions: int = 3, loss='mse', modelname='Standard_Model'):
        self.env = env
        self.finish = finish
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.hidden_units = hidden_units
        self.gamma = gamma
        self.batch_size = batch_size
        self.opt = opt
        self.learning_rate = learning_rate
        self.loss = loss
        self.max_treward = 0
        self.averages = []
        self.best_treward = 0
        self.act = act
        self.env_actions = actions
        self.modelname = modelname
        self.dropout = dropout
        self.m_activation = m_activation

        # Initialize lists to keep track of states and rewards
        self.state_history = []
        self.reward_history = []
        self.train_action_history = []
        self.test_action_history = []

        self.memory = deque(maxlen=1_000)
        self.osn = env.observation_space.shape[0]
        self.model = self.get_model(
            hidden_units, dropout, m_activation, self.modelname)

        for layer in self.model.layers:
            if hasattr(layer, 'get_weights'):
                rl_logger.info(f'Layer: {layer.name}')

        agent_logger.info(f"Agent initialized with parameters:\n"
                          f"gamma: {gamma}, hidden_units: {hidden_units}, opt: {opt}, learning_rate: {learning_rate}, epsilon: {epsilon},\n "
                          f"epsilon_min: {epsilon_min}, epsilon_decay: {epsilon_decay}, ")

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
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(64, input_dim=self.osn,
                  activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(tf.keras.layers.Dense(
            hidden_units, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Dense(
            self.env_actions, activation=m_activation))
        model.compile(loss=self.loss, optimizer=self.opt(
            learning_rate=self.learning_rate), metrics=['accuracy'])
        agent_logger.info(
            f"Model built with action space {self.env.action_space.n}.")
        return model

    def base_dense_model(self, m_activation, dropout):
        agent_logger.info("Dense Model loaded.")
        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.Dense(
            128, input_dim=self.osn, activation='relu'))
        base_model.add(BatchNormalization())
        base_model.add(Dropout(dropout))
        base_model.add(tf.keras.layers.Dense(64, activation='relu'))
        base_model.add(BatchNormalization())
        base_model.add(tf.keras.layers.Dense(32, activation='relu'))
        base_model.add(BatchNormalization())
        base_model.add(tf.keras.layers.Dense(
            self.env_actions, activation=m_activation))
        base_model.compile(
            optimizer=self.opt(learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])

        return base_model

    def base_lstm_model(self, m_activation, dropout):
        rl_logger.info("LSTM Model loaded.")
        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(
            self.osn, 1), return_sequences=True))
        base_model.add(tf.keras.layers.Dropout(dropout))
        base_model.add(tf.keras.layers.LSTM(100, activation='relu'))
        base_model.add(tf.keras.layers.Dense(
            self.env_actions, activation=m_activation))
        base_model.compile(
            optimizer=self.opt(learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])

        return base_model

    def base_conv1d_lstm_model(self, m_activation, dropout):
        rl_logger.info("CONV1D Model loaded.")
        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2,
                       activation='relu', input_shape=(self.osn, 1)))
        base_model.add(tf.keras.layers.MaxPool1D(pool_size=1))
        base_model.add(tf.keras.layers.LSTM(
            100, activation='relu', return_sequences=True))
        base_model.add(tf.keras.layers.Dropout(dropout))
        base_model.add(tf.keras.layers.LSTM(100, activation='relu'))
        base_model.add(tf.keras.layers.Dense(
            self.env_actions, activation=m_activation))
        base_model.compile(
            optimizer=self.opt(learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])

        return base_model

    def build_resnet_model(self, m_activation, dropout):
        rl_logger.info("ResNet Model loaded.")

        inputs = tf.keras.layers.Input(shape=(self.osn,))
        x = tf.keras.layers.Dense(128, activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)

        # Residual block
        residual = x
        x = tf.keras.layers.Dense(128, activation='relu')(
            x)  # Change this to 128 units
        x = BatchNormalization()(x)
        x = Dropout(dropout)(x)
        x = tf.keras.layers.Dense(128, activation='relu')(
            x)  # Change this to 128 units
        x = tf.keras.layers.Add()([x, residual])

        x = BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(
            self.env_actions, activation=m_activation)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.opt(
            learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])
        return model

    def base_conv1d_model(self, m_activation, dropout):
        rl_logger.info("Simple CONV1D Model loaded.")

        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.Conv1D(
            filters=64, kernel_size=2, activation='relu', input_shape=(self.osn, 1)))
        base_model.add(tf.keras.layers.MaxPool1D(pool_size=2))
        base_model.add(tf.keras.layers.Flatten())
        base_model.add(tf.keras.layers.Dense(100, activation='relu'))
        base_model.add(tf.keras.layers.Dropout(dropout))
        base_model.add(tf.keras.layers.Dense(
            self.env_actions, activation=m_activation))

        base_model.compile(optimizer=self.opt(
            learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])
        return base_model

    def base_transformer_model(self, m_activation, dropout=0.1, d_model=128, num_heads=4):
        rl_logger.info("Transformer Model loaded.")

        inputs = tf.keras.layers.Input(shape=(self.osn, 1))
        x = tf.keras.layers.Dense(d_model)(inputs)

        # Embed the sequence into the d_model space
        x = tf.keras.layers.Dense(d_model)(x)

        # Add positional encoding if needed. Skipped in this example for brevity.
        # x = positional_encoding(x, d_model)

        x = transformer_block(d_model, num_heads, dropout)(x)
        x = tf.keras.layers.GlobalAveragePooling1D()(x)
        outputs = tf.keras.layers.Dense(
            self.env_actions, activation=m_activation)(x)

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=self.opt(
            learning_rate=self.learning_rate), loss=self.loss, metrics=['accuracy'])
        return model

    def get_model_parameters(self):
        """
        Retrieve the parameters used to construct the model.
        """
        return {
            "model_name": self.modelname,
            "gamma": self.gamma,
            "hidden_units": self.hidden_units,
            "learning_rate": self.learning_rate,
            "epsilon": self.epsilon,
            "dropout": self.dropout,
            "act": self.act,
            "m_activation": self.m_activation,
            "input_dim": self.osn,
            "action_space_n": self.env.action_space.n,
            "env_actions": self.env_actions,
            "loss": self.loss,
            "learning_rate": self.learning_rate,
            "metrics": ['accuracy']
        }

    def save(self, b_reward, acc):
        # Check if new record needs to be saved
        should_save = is_new_record(b_reward, acc, self.modelname)

        # Save new record if conditions met
        if should_save:
            model_params = self.get_model_parameters()
            save_to_csv(b_reward, acc, model_params)
            save_model(self)

    def load(self):
        if os.path.exists(MODEL_PATH):
            self.model = tf.keras.models.load_model(
                f"{MODEL_PATH}/{self.modelname}_model.keras")
            agent_logger.info("Model loaded.")

    def select_action(self, state, method='argmax'):
        action_methods = {
            'argmax': self.argmax_action,
            'softmax': self.softmax_action,
            'default': self.default_action
        }
        return action_methods[method](state)

    def default_action(self, state):
        action = self.model.predict(state)[0][0]
        agent_logger.info(f'ACT: prediction is {action}')
        return action

    def argmax_action(self, state):
        q_values = self.model.predict(state)[0]
        agent_logger.info(f'ACT:the prediction is {q_values}')
        action = np.argmax(q_values)
        agent_logger.info(f'ACT:argmax of this prediction is {action}')
        return action

    def softmax_action(self, state):
        q_values = self.model.predict(state)[0]
        agent_logger.info(f'ACT:the prediction is {q_values}')
        max_q = np.max(q_values)
        action_probs = np.exp(q_values - max_q) / \
            np.sum(np.exp(q_values - max_q))

        if np.isnan(action_probs).any():
            return np.random.choice(len(action_probs))
        action = np.random.choice(len(action_probs), p=action_probs)
        agent_logger.info(f'ACT:softmax of this prediction is {action}')
        return action

    def replay(self, method='default'):
        replay_targets = {
            'argmax': lambda next_state: self.gamma * np.amax(self.model.predict(next_state)[0]),
            'softmax': lambda next_state: self.gamma * np.sum(np.exp(q_values) / np.sum(np.exp(q_values)) * self.model.predict(next_state)[0]),
            'default': lambda next_state: self.gamma * self.model.predict(next_state)[0][0]
        }

        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        for state, action, reward, next_state, done in batch:
            if not done:
                q_values = self.model.predict(next_state)[0]
                reward += replay_targets[method](next_state)

            target = self.model.predict(state)
            agent_logger.info(
                f"REPLAY -state shape: {state.shape}, action: {action}, target shape: {target.shape}")
            target[0, action] = reward
            states.append(state)
            targets.append(target)

        states = np.vstack(states)
        targets = np.vstack(targets)
        self.model.fit(states, targets, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        agent_logger.info(f"REPLAY: done. Epsilon is now {self.epsilon}")

    def save_model(self):
        base_filename = MODEL_PATH
        model_count = 1
        acc = self.env.accuracy
        b_reward = self.best_treward

        while os.path.exists(f'{base_filename}/{self.modelname}{model_count}.keras'):
            model_count += 1

        filename = f'{base_filename}/{self.modelname}{model_count}.keras'
        self.model.save(filename)
        agent_logger.info(f"Model saved as {filename}")

        # Save model score to CSV
        score_df = pd.DataFrame({'accuracy': [acc], 'reward': [b_reward]})
        score_filename = "data/best_model/best_model_scores.csv"

        if os.path.isfile(score_filename):
            df = pd.read_csv(score_filename)
            df = df.append(score_df, ignore_index=True)
            df.to_csv(score_filename, index=False)
        else:
            score_df.to_csv(score_filename, index=False)

    def learn(self, episodes):
        trewards = []
        # Initialize some variables for early stopping

        for e in range(1, episodes + 1):
            agent_logger.info(f"LEARN: Currently at episode {e}/{episodes}")
            state = self.env.reset()

            if self.modelname in ["LSTM_Model", "CONV1D_LSTM_Model"]:
                state = np.reshape(state, [1, self.osn, 1])
            else:
                state = np.reshape(state, [1, self.osn])

            sum_of_rewards = 0  # Initialize the sum_of_rewards for this episode

            while not self.env.is_episode_done():
                action = self.select_action(state, self.act)
                next_state, reward, done, _ = self.env.step(action)

                if self.modelname in ["LSTM_Model", "CONV1D_LSTM_Model"]:
                    next_state = np.reshape(state, [1, self.osn, 1])
                else:
                    next_state = np.reshape(next_state, [1, self.osn])

                sum_of_rewards += reward   # Update the sum_of_rewards for this episode
                agent_logger.info(
                    f"LEARN:the rewards is {reward} and the sum of rewards is {sum_of_rewards}")

                self.memory.append([state, action, reward, next_state, done])
                # Append state and reward to their respective histories
                self.state_history.append(state[0])
                self.reward_history.append(reward)
                # In the learn method
                self.train_action_history.append(action)

                state = next_state
                agent_logger.info(
                    f"LEARN:episode {e} ===========================================")

            if sum_of_rewards > self.best_treward:
                self.best_treward = sum_of_rewards
                self.save(self.best_treward, self.env.accuracy)

            av = sum(trewards[-25:]) / \
                25 if len(trewards) > 25 else sum_of_rewards
            self.averages.append(av)
            self.max_treward = max(self.max_treward, sum_of_rewards)
            agent_logger.info(
                f"LEARN:the max rewards for this episode {e} is {self.max_treward}")

            agent_logger.info(
                f"LEARN:===========================================")

            if av > 195 and self.finish:
                break

            if len(self.memory) > self.batch_size:
                self.replay(self.act)   # Adjusted this line

        self.save(self.best_treward, self.env.accuracy)
        agent_logger.info(f"LEARN: Replay done. Epsilon is now {self.epsilon}")

    def test(self, episodes):
        self.load()
        self.epsilon = 0
        test_rewards = []

        for episode in range(1, episodes + 1):
            agent_logger.info(
                f"TEST: Currently at episode {episode}/{episodes}")
            episode_reward = self.run_single_test_episode()
            test_rewards.append(episode_reward)

        average_reward = sum(test_rewards) / episodes
        agent_logger.info(
            f"TEST: Average reward over {episodes} episodes: {average_reward}")
        return test_rewards

    def run_single_test_episode(self):
        state = self.env.reset()
        episode_reward = 0

        while not self.env.is_episode_done():

            if self.modelname in ["LSTM_Model", "CONV1D_LSTM_Model"]:
                state = np.reshape(state, [1, self.osn, 1])
            else:
                state = np.reshape(state, [1, self.osn])

            agent_logger.info(
                f"TEST - state shape: {state.shape}, state: {state}")

            action = self.select_action(state, self.act)

            # Depending on the level of logging you need, you may or may not need the following lines.
            # The action selection process is alearning_rateeady logged within select_action.
            q_values = self.model.predict(state)[0]
            agent_logger.info(f"TEST: model predicted {q_values}")

            # In the test method
            self.test_action_history.append(action)

            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            state = next_state  # Update the state

        agent_logger.info(f"TEST: Episode reward: {episode_reward}")
        return episode_reward

    def calculate_correlations(self, fn=None, ax=None):
        # If feature_names is None, use generic names
        if fn is None:
            fn = [f'feature_{i}' for i in range(self.osn)]

        # Convert state history and reward history to a DataFrame
        agent_logger.info("Length of fn:", len(fn))
        agent_logger.info("Number of columns in state_history:", len(
            self.state_history[0]) if self.state_history else "state_history is empty")

        if self.modelname in ["LSTM_Model", "CONV1D_LSTM_Model"]:
            state_df = pd.DataFrame(self.state_history.squeeze(), columns=fn)
        else:
            state_df = pd.DataFrame(self.state_history, columns=fn)

        state_df = pd.DataFrame(self.state_history, columns=fn)
        reward_df = pd.DataFrame(self.reward_history, columns=['reward'])

        # Combine them into one DataFrame
        combined_df = pd.concat([state_df, reward_df], axis=1)

        # Compute correlation
        correlation_matrix = combined_df.corr()
        reward_correlations = correlation_matrix['reward'].drop('reward')

        agent_logger.info("Feature-Reward Correlations:")
        agent_logger.info(reward_correlations)

        # Save correlations to a CSV
        csv_filename = next_available_filename("reward_corr", "csv")
        reward_correlations.to_csv(csv_filename)
        agent_logger.info(f"Saved correlations to {csv_filename}")

        # Create a list of colors (same length as reward_correlations)
        colors = plt.cm.viridis(np.linspace(0, 1, len(reward_correlations)))

        # Sort correlations
        sorted_correlations = reward_correlations.sort_values()

        # Plot correlations
        if ax is None:
            plt.figure(figsize=(10, 6))
            ax = plt.gca()  # Get current axis

        sorted_correlations.plot(kind='barh', color=colors)
        plt.xlabel('Correlation Coefficient')
        plt.ylabel('Feature')
        plt.title('Feature-Reward Correlations')
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax,
                     orientation='vertical', label='Color Scale')
