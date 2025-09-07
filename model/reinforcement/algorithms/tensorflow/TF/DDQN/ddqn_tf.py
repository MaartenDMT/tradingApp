import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation="relu")
        self.dense2 = keras.layers.Dense(fc2_dims, activation="relu")
        self.V = keras.layers.Dense(1, activation=None)
        self.A = keras.layers.Dense(n_actions, activation=None)

    def call(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = (V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True)))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A


class Replaybuffer():
    def __init__(self, max_size, input_shape) -> None:
        self.mem_size = max_size
        self.mem_cntr = 0

        self.state_memory = np.zeros((self.mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size,  *input_shape),
                                         dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state.reshape(-1)
        self.new_state_memory[index] = state_.reshape(-1)
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size, input_dims, epsilon_dec=1e-3, epsilon_end=0.01, mem_size=100_000,
                 fname='dueling_dqn', fc1_dims=128, fc2_dims=128, replace=100) -> None:

        self.action = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.model_file = fname
        self.replace = replace
        self.batch_size = batch_size

        self.learn_step_counter = 0
        self.memory = Replaybuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions)
        self.q_next = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions)

        self.q_eval.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')

        # just a formatility, won't optimize network
        self.q_next.compile(optimizer=Adam(learning_rate=lr),
                            loss='mean_squared_error')

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_actions(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action)
        else:
            # np.array([observation])
            state = np.array(observation).reshape(1, -1)
            actions = self.q_eval.advantage(state)

            # np.argmax(actions)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size)

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)

        # changing q_pred doesn't matter becazuse we are passing states to the train function anyway
        # tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)

        # improve on my solution !
        for idx, terminal in enumerate(dones):
            q_target[idx, actions[idx]] = rewards[idx] + \
                self.gamma * q_next[idx, max_actions[idx]] * \
                (1 - int(dones[idx]))

        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - \
            self.eps_dec if self.epsilon > self.eps_min else self.eps_min

        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval.save(self.model_file, save_format='tf')

    def load_model(self):
        self.q_eval = load_model(self.model_file)
