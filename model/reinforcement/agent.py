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
from util.rl_util import next_available_filename

pd.set_option('mode.chained_assignment', None)

logger = loggers.setup_loggers()
rl_logger = logger['agent']


class DQLAgent:
    def __init__(self, gamma=0.95, hu=24, opt=Adam, lr=0.001,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, dropout=0.5,
                 finish=False, env=None, act='argmax', m_activation='linear'):
        self.env = env
        self.finish = finish
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_treward = 0
        self.averages = []
        self.best_treward = 0
        self.act = act if m_activation != "tanh"else None

        # Initialize lists to keep track of states and rewards
        self.state_history = []
        self.reward_history = []
        self.train_action_history = []
        self.test_action_history = []

        self.memory = deque(maxlen=200)
        self.osn = env.observation_space.shape[0]
        self.model = self._build_model(
            hu, opt, lr, dropout, m_activation=m_activation)
        rl_logger.info(f"Agent initialized with parameters:\n"
                       f"gamma: {gamma}, hu: {hu}, opt: {opt}, lr: {lr}, epsilon: {epsilon},\n "
                       f"epsilon_min: {epsilon_min}, epsilon_decay: {epsilon_decay}, ")

    def _build_model(self, hu, opt, lr, dropout, m_activation):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(hu, input_dim=self.osn,
                  activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout))
        model.add(tf.keras.layers.Dense(hu, activation='relu', kernel_regularizer=l2(0.01)))
        model.add(BatchNormalization())
        model.add(tf.keras.layers.Dense(self.env.action_space.n, activation=m_activation))
        model.add(tf.keras.layers.Dense(1, activation=m_activation))
        model.compile(loss='mse', optimizer=opt(learning_rate=lr))
        rl_logger.info(
            f"Model built with action space {self.env.action_space.n}.")
        return model
    
    def base_dense_model(self):
        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.Dense(128, activation='relu'))
        base_model.add(tf.keras.layers.Dense(64, activation='relu'))
        base_model.add(tf.keras.layers.Dense(32, activation='relu'))
        base_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        base_model.fit(X_train, y_train, epochs=500, batch_size=16, validation_data=(X_test, y_test), verbose=0)
        return base_model

    def base_lstm_model(self):
        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.LSTM(100, activation='relu', input_shape=(X_train.shape[1], 1), return_sequences=True))
        base_model.add(tf.keras.layers.Dropout(0.2))
        base_model.add(tf.keras.layers.LSTM(100, activation='relu'))
        base_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        base_model.fit(X_train, y_train, epochs=500, batch_size=16, validation_data=(X_test, y_test), verbose=0)
        return base_model

    def base_conv1d_lstm_model(self):
        base_model = tf.keras.Sequential()
        base_model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
        base_model.add(tf.keras.layers.MaxPool1D(pool_size=2))
        base_model.add(tf.keras.layers.LSTM(100, activation='relu', return_sequences=True))
        base_model.add(tf.keras.layers.Dropout(0.2))
        base_model.add(tf.keras.layers.LSTM(100, activation='relu'))
        base_model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        base_model.fit(X_train, y_train, epochs=500, batch_size=16, validation_data=(X_test, y_test), verbose=0)
        return base_model

    def save(self, b_reward, acc):
        # Define the file path for the CSV
        filename = "data/best_model/best_model.csv"

        if os.path.isfile(filename):
            # If the CSV file already exists, read its content
            df = pd.read_csv(filename)
            best_reward = df['reward'].max()
            best_accuracy = df['accuracy'].max()

            if b_reward > best_reward and acc > best_accuracy:
                # If the current reward and accuracy are better, update the CSV and save the model
                df.loc[df['reward'].idxmax()] = [acc, b_reward]
                df.to_csv(filename, index=False)
                self.model.save('data/saved_model/dql_model')

        else:
            # If the CSV file does not exist, create it and save the initial values
            df = pd.DataFrame({'accuracy': [acc], 'reward': [b_reward]})
            df.to_csv(filename, index=False)
            self.model.save('data/saved_model/dql_model')

    def load(self):
        self.model = tf.keras.models.load_model('data/saved_model/dql_model')
        rl_logger.info("Model loaded.")

    def select_action(self, state, method='argmax'):
        if method == 'argmax':
            return self.argmax_action(state)
        elif method == 'softmax':
            return self.softmax_action(state)
        else:
            return self.action(state)

    def select_replay(self, method='argmax'):
        if method == 'argmax':
            self.arg_replay()
        elif method == 'softmax':
            self.soft_replay()
        else:
            self.replay()

    def action(self, state):
        rl_logger.info(f'ACT:getting a random action')
        if random.random() <= self.epsilon:
            a = self.env.action_space.sample()
            rl_logger.info(f'ACT:getting a sampled action {a}')
            return a
        action = self.model.predict(state)[0]
        rl_logger.info(f'ACT: of this prediction is {action}')
        return action

    def argmax_action(self, state):
        rl_logger.info(f'ACT:getting a random action')
        if random.random() <= self.epsilon:
            a = self.env.action_space.sample()
            rl_logger.info(f'ACT:getting a sampled action {a}')
            return a
        a = self.model.predict(state)[0]
        rl_logger.info(f'ACT:model has predicted {a}')
        action = np.argmax(a)
        rl_logger.info(f'ACT:argmax of this prediction is {action}')
        return action

    def softmax_action(self, state):
        if random.random() <= self.epsilon:
            a = self.env.action_space.sample()
            return a
        q_values = self.model.predict(state)[0]
        action_probs = np.exp(q_values) / np.sum(np.exp(q_values))
        action = np.random.choice(len(action_probs), p=action_probs)
        rl_logger.info(f'ACT:softmax of this prediction is {action}')
        return action

    def arg_replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                pred = np.amax(self.model.predict(next_state)[0])
                reward += self.gamma * pred
            target = self.model.predict(state)
            target[0, action] = reward
            self.model.fit(state, target, epochs=1, verbose=False)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        rl_logger.info(f"REPLAY: done. Epsilon is now {self.epsilon}")

    def soft_replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                q_values = self.model.predict(next_state)[0]
                action_probs = np.exp(q_values) / np.sum(np.exp(q_values))
                # Use the expected value based on softmax
                expected_reward = np.sum(action_probs * q_values)
                reward += self.gamma * expected_reward
            target = self.model.predict(state)
            target[0, action] = reward
            self.model.fit(state, target, epochs=1, verbose=False)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        rl_logger.info(f"REPLAY: done. Epsilon is now {self.epsilon}")

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            if not done:
                # Get the single continuous value
                pred = self.model.predict(next_state)[0][0]
                reward += self.gamma * pred
            target = self.model.predict(state)
            # Update the single continuous value in the target
            target[0, 0] = reward
            self.model.fit(state, target, epochs=1, verbose=False)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        rl_logger.info(f"REPLAY: done. Epsilon is now {self.epsilon}")

    def learn(self, episodes):
        trewards = []
        # Initialize some variables for early stopping

        for e in range(1, episodes + 1):
            rl_logger.info(f"LEARN: Currently at episode {e}/{episodes}")
            state = self.env.reset()
            state = np.reshape(state, [1, self.osn])

            sum_of_rewards = 0  # Initialize the sum_of_rewards for this episode

            while not self.env.is_episode_done():
                action = self.select_action(state, self.act)
                next_state, reward, done, _ = self.env.step(action)
                next_state = np.reshape(next_state, [1, self.osn])

                sum_of_rewards += reward   # Initialize the sum_of_rewards for this episode

                self.memory.append([state, action, reward, next_state, done])
                # Append state and reward to their respective histories
                self.state_history.append(state[0])
                self.reward_history.append(reward)
                # In the learn method
                self.train_action_history.append(action)

                state = next_state

            if sum_of_rewards > self.best_treward:
                self.best_treward = sum_of_rewards
                self.save(self.best_treward, self.env.accuracy)

            av = sum(trewards[-25:]) / 25
            self.averages.append(av)
            self.max_treward = max(self.max_treward, sum_of_rewards)

            rl_logger.info(
                f"LEARN:===========================================")

            if av > 195 and self.finish:
                break

            if len(self.memory) > self.batch_size:
                self.select_replay(self.act)

        self.save(self.best_treward, self.env.accuracy)
        rl_logger.info(f"LEARN: Replay done. Epsilon is now {self.epsilon}")

    def test(self, episodes):
        self.load()
        self.epsilon = 0
        test_rewards = []

        reverse_mapping = {0: -1, 1: 0, 2: 1}  # Reverse of action_mapping

        for episode in range(1, episodes + 1):
            rl_logger.info(f"TEST: Currently at episode {episode}/{episodes}")
            episode_reward = self.run_single_test_episode(reverse_mapping)
            test_rewards.append(episode_reward)

        average_reward = sum(test_rewards) / episodes
        rl_logger.info(
            f"TEST: Average reward over {episodes} episodes: {average_reward}")
        return test_rewards

    def run_single_test_episode(self, reverse_mapping):
        state = self.env.reset()
        episode_reward = 0

        while not self.env.is_episode_done():
            state = np.reshape(state, [1, self.osn])
            q_values = self.model.predict(state)[0]

            if self.act == "argmax":
                action_index = np.argmax(q_values)
                # Convert back to your action space of -1, 0, 1
                action = reverse_mapping[action_index]
            elif self.act == "softmax":
                action_probs = np.exp(q_values) / np.sum(np.exp(q_values))
                action_index = np.random.choice(
                    len(action_probs), p=action_probs)
                # Convert back to your action space of -1, 0, 1
                action = reverse_mapping[action_index]
            else:
                action = q_values

            # In the test method
            self.test_action_history.append(action)

            next_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            state = next_state

        rl_logger.info(f"TEST: Episode reward: {episode_reward}")
        return episode_reward
    
    def train_base_dense_model(self, X_train, y_train, X_test, y_test):
        # ... (code for training base dense model)
        accuracy = 0.85  # Replace this with your actual accuracy
        self.save(accuracy, accuracy)  # Save the model
        return accuracy

    def train_base_lstm_model(self, X_train, y_train, X_test, y_test):
        # ... (code for training base LSTM model)
        accuracy = 0.90  # Replace this with your actual accuracy
        self.save(accuracy, accuracy)  # Save the model
        return accuracy

    def train_base_conv1d_lstm_model(self, X_train, y_train, X_test, y_test):
        # ... (code for training base Conv1D LSTM model)
        accuracy = 0.88  # Replace this with your actual accuracy
        self.save(accuracy, accuracy)  # Save the model
        return accuracy

    def calculate_correlations(self, fn=None, ax=None):
        # If feature_names is None, use generic names
        if fn is None:
            fn = [f'feature_{i}' for i in range(self.osn)]

        # Convert state history and reward history to a DataFrame
        rl_logger.info("Length of fn:", len(fn))
        rl_logger.info("Number of columns in state_history:", len(
            self.state_history[0]) if self.state_history else "state_history is empty")
        state_df = pd.DataFrame(self.state_history, columns=fn)
        reward_df = pd.DataFrame(self.reward_history, columns=['reward'])

        # Combine them into one DataFrame
        combined_df = pd.concat([state_df, reward_df], axis=1)

        # Compute correlation
        correlation_matrix = combined_df.corr()
        reward_correlations = correlation_matrix['reward'].drop('reward')

        rl_logger.info("Feature-Reward Correlations:")
        rl_logger.info(reward_correlations)

        # Save correlations to a CSV
        csv_filename = next_available_filename("reward_corr", "csv")
        reward_correlations.to_csv(csv_filename)
        rl_logger.info(f"Saved correlations to {csv_filename}")

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
        plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis),
                     orientation='vertical', label='Color Scale')
