import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.optimizers import Adam

from model.reinforcement.TF.PG.network import PolicyGradientNetwork


class Agent:
    def __init__(self, n_actions, alpha=0.003, gamma=0.99, fc1_dims=256, fc2_dims=256, fname='dqn_model'):

        self.action = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []
        self.model_file = fname
        self.policy = PolicyGradientNetwork(
            n_actions, fc1_dims=fc1_dims, fc2_dims=fc2_dims)
        self.policy.compile(optimizer=Adam(learning_rate=self.alpha))

    def choose_action(self, observation):
        state = tf.convert_to_tensor(
            np.array(observation).reshape(1, -1), dtype=tf.float32)
        probs = self.policy(state)
        action_probs = tfp.distributions.Categorical(probs=probs)
        action = action_probs.sample()
        print(f'sampled action is {action}')
        return action.numpy()[0]

    def store_transition(self, observation, action, reward):
        self.state_memory.append(observation.reshape(-1))
        self.reward_memory.append(reward)
        self.action_memory.append(action)

    def learn(self):
        actions = tf.convert_to_tensor(self.action_memory, dtype=tf.float32)
        rewards = tf.convert_to_tensor(self.reward_memory, dtype=tf.float32)

        G = np.zeros_like(rewards)
        for t in range(len(rewards)):
            G_sum = 0
            discount = 1
            for k in range(t, len(rewards)):
                G_sum += rewards[k]*discount
                discount *= self.gamma
            G[t] = G_sum

        with tf.GradientTape() as tape:
            loss = 0
            for idx, (g, state) in enumerate(zip(G, self.state_memory)):
                state = tf.convert_to_tensor([state], dtype=tf.float32)
                probs = self.policy(state)
                actions_probs = tfp.distributions.Categorical(probs=probs)
                log_prob = actions_probs.log_prob(actions[idx])
                loss += -g * tf.squeeze(log_prob)

        gradient = tape.gradient(loss, self.policy.trainable_variables)
        self.policy.optimizer.apply_gradients(
            zip(gradient, self.policy.trainable_variables))

        self.state_memory = []
        self.reward_memory = []
        self.action_memory = []

    def save_models(self):
        print("... saving modesl ...")
        self.policy.save_weights(self.policy.checkpoint_file)

    def load_models(self):
        print("... loading modesl ...")
        self.policy.load_weights(self.policy.checkpoint_file)
