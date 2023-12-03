import os

import tensorflow.keras as keras
from tensorflow.keras.layers import Dense


class PolicyGradientNetwork(keras.Model):
    def __init__(self, n_actions, fc1_dims=256, fc2_dims=256, name='actor_critic', chkpt_dir='tmp/actor_critic'):
        super(PolicyGradientNetwork, self).__init__()
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.model_name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+"_pg")

        self.fc1 = Dense(fc1_dims, activation="relu")
        self.fc2 = Dense(fc2_dims, activation="relu")
        self.pi = Dense(self.n_actions, activation="softmax")

    def call(self, state):
        value = self.fc1(state)
        value = self.fc2(value)

        pi = self.pi(value)

        return pi
