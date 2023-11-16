import os
import random
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from model_selector import ModelBuilder
from tensorflow.keras.models import load_model

import util.loggers as loggers
from util.agent_utils import is_new_record, save_model, save_to_csv
from util.rl_util import next_available_filename

logger = loggers.setup_loggers()
agent_logger = logger['agent']

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
        self.act = act
        self.env_actions = actions
        self.modelname = modelname
        self.dropout = dropout
        self.m_activation = m_activation

        self.memory = deque(maxlen=100_000)
        self.state_history = []
        self.reward_history = []
        self.train_action_history = []
        self.test_action_history = []
        self.max_treward = 0
        self.averages = []
        self.best_treward = 0

        self.model = ModelBuilder(env, opt, loss, learning_rate).get_model(
            hidden_units, dropout, m_activation, modelname)
        self.state_manager = StateManager()

        self.action_selector = ActionSelector(self.model)
        self.replay_logic = ReplayLogic(self.model, self.memory)

    def learn(self, episodes):
        Learner(self.env, self.model, self.memory, self.state_history, self.reward_history,
                self.train_action_history, self.test_action_history, self.action_selector, self.replay_logic).run(episodes)

    def test(self, episodes):
        return Tester(self.env, self.model).run(episodes)

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
            self.model = load_model(
                f"{MODEL_PATH}/{self.modelname}_model.keras")
            agent_logger.info("Model loaded.")

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


class Learner:
    def __init__(self, env, model, memory, state_h, reward_h, train_h, test_h, action_s, replay_l):
        self.env = env
        self.model = model
        self.memory = memory
        self.state_history = state_h
        self.reward_history = reward_h
        self.train_action_history = train_h
        self.test_action_history = test_h
        self.action_selector = action_s
        self.replay_logic = replay_l

    def run(self, episodes):
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
                action = self.action_selector.select(state, self.act)
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
                self.replay_logic.execute(self.act)  # Adjusted this line

        self.save(self.best_treward, self.env.accuracy)
        agent_logger.info(f"LEARN: Replay done. Epsilon is now {self.epsilon}")


class Tester:
    def __init__(self, env, model):
        self.env = env
        self.model = model

    def run(self, episodes):
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


class ReplayLogic:
    def __init__(self, model, memory):
        self.model = model
        self.memory = memory

    def execute(self, method='argmax'):
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
            # Ensure action is an integer and within the valid range
            if isinstance(action, int) and 0 <= action < target.shape[1]:
                target[0, action] = reward
            else:
                agent_logger.error(f"Invalid action: {action}")
            states.append(state)
            targets.append(target)

        states = np.vstack(states)
        targets = np.vstack(targets)
        self.model.fit(states, targets, epochs=1, verbose=False)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        agent_logger.info(f"REPLAY: done. Epsilon is now {self.epsilon}")


class StateManager:
    def __init__(self, config):
        self.memory = deque(maxlen=config['memory_size'])
        # Other state management logic


class ActionSelector:
    def __init__(self, model):
        self.model = model

    def select(self, state, method='argmax'):
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
