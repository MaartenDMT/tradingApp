import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import util.loggers as loggers
from model.reinforcement.MAPDDG import buffer, network
from model.reinforcement.rl_visual import plot_and_save_metrics
from util.agent_utils import is_new_record, save_to_csv

logger = loggers.setup_loggers()
agent_logger = logger['agent']
rl_logger = logger['rl']

MODEL_PATH = 'data/reinforcement/'


class MAPDDGAgent:
    def __init__(self, env, gamma=0.95, learning_rate=0.001, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, replay_buffer=None, num_agents=2):

        self.env = env
        self.batch_size = batch_size
        self.memory = replay_buffer
        self.env_actions = env.action_space.n
        self.osn = env.observation_space.shape

        # replay buffer for memory
        self.memory = buffer.Replaybuffer(
            100_000, num_agents, state_shape=self.osn, action_dim=self.env_actions)

        self.agents = [Agent(self.env_actions, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, self.env_actions, self.memory, i)
                       for i in range(num_agents)]

    def store_transitions(self, observations, actions, rewards, next_observations, dones):
        # Ensure that rewards and dones are lists of the appropriate length
        if np.isscalar(rewards):
            rewards = [rewards] * self.num_agents
        if np.isscalar(dones):
            dones = [dones] * self.num_agents

        # Store transitions for each agent
        for i, agent in enumerate(self.agents):

            agent.store_transition(
                observations, actions, rewards, next_observations, dones
            )

    def choose_actions(self, observations):
        actions = []
        for agent, obs in zip(self.agents, observations):
            # Implement this method in Agent class
            action = agent.choose_action(obs)
            actions.append(action)
        return actions

    def learn(self):
        # Make sure there are enough samples in the buffer
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(
            self.batch_size)

        for agent in self.agents:
            # Learning logic for each agent
            # You will need to adapt this based on your specific requirements
            agent.learn(states, actions, rewards, next_states, dones)
            agent.decay_epsilon()


class Agent:
    def __init__(self, action_dim, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, env_actions, shared_memory, agent_number):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env_actions = env_actions
        self.agent_idx = agent_number
        self.memory = shared_memory

        self.actor = network.ActorNetwork(action_dim)
        # Adjust the constructor based on your state and action dimensions
        self.critic = network.CriticNetwork()
        self.target_actor = tf.keras.models.clone_model(self.actor)
        self.target_critic = tf.keras.models.clone_model(self.critic)
        self.actor.compile(optimizer=Adam(
            learning_rate=learning_rate), loss='mean_squared_error')
        self.critic.compile(optimizer=Adam(
            learning_rate=learning_rate), loss='mean_squared_error')
        self.target_actor.compile(optimizer=Adam(
            learning_rate=learning_rate), loss='mean_squared_error')
        self.target_critic.compile(optimizer=Adam(
            learning_rate=learning_rate), loss='mean_squared_error')

        # Update target networks
        self.update_target_networks(tau=1)

    def update_target_networks(self, tau=0.005):
        # Update weights with a factor of tau for soft updates
        new_actor_weights = [(tau * aw + (1 - tau) * tw) for aw, tw in zip(
            self.actor.get_weights(), self.target_actor.get_weights())]
        new_critic_weights = [(tau * cw + (1 - tau) * tw) for cw, tw in zip(
            self.critic.get_weights(), self.target_critic.get_weights())]

        self.target_actor.set_weights(new_actor_weights)
        self.target_critic.set_weights(new_critic_weights)

        # Log the soft update of target networks
        agent_logger.info("Soft update of target networks")

    # Add methods for action selection, learning, storing transitions, etc.
    def choose_action(self, observation):
        """
        Choose an action based on epsilon-greedy strategy.
        """
        if np.random.rand() < self.epsilon:
            # Exploration: choose a random action
            action = np.random.choice(self.env_actions)
        else:
            # Exploitation: choose the best action based on actor's policy
            state = np.array(observation).reshape(1, -1)
            action_probs = self.actor.predict(state)[0]
            action = np.argmax(action_probs)

        # Log the chosen action
        agent_logger.info(f"Chose action: {action}")
        return action

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)
        # Log the stored transition
        agent_logger.info(
            f"Stored transition agent: state={state}, action={action}, reward={reward}, new_state={new_state}, done={done}")

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(
            self.batch_size)

        # Agent-specific states, actions, rewards, and dones
        agent_states = states[:, self.agent_idx, :]
        agent_next_states = next_states[:, self.agent_idx, :]
        agent_actions = actions[:, self.agent_idx, :]
        agent_rewards = rewards[:, self.agent_idx]
        agent_dones = dones[:, self.agent_idx]

        # Reshape agent-specific states and next_states based on their dimensionality
        if agent_states.ndim == 3:
            agent_states = agent_states.reshape(self.batch_size, -1)
            agent_next_states = agent_next_states.reshape(self.batch_size, -1)
        elif agent_states.ndim == 2 and agent_states.shape[1] != -1:
            agent_states = agent_states.reshape(self.batch_size, -1)
            agent_next_states = agent_next_states.reshape(self.batch_size, -1)
        elif agent_states.ndim == 1:
            agent_states = agent_states.reshape(self.batch_size, 1)
            agent_next_states = agent_next_states.reshape(self.batch_size, 1)

        # Predict target actions for the next states using all agents' information
        target_actions = self.target_actor.predict(next_states)

        # Predict the target critic value for next state-action pairs using all agents' information
        target_critic_values = self.target_critic.predict(
            [next_states, target_actions])
        target_critic_values = target_critic_values.reshape(
            self.batch_size, -1)

        # Compute the critic target using agent-specific rewards and dones
        critic_targets = agent_rewards + self.gamma * \
            (1 - agent_dones) * target_critic_values

        # Update critic network using all agents' states and actions
        with tf.GradientTape() as tape:
            critic_value = self.critic([states, actions], training=True)
            critic_loss = tf.keras.losses.mean_squared_error(
                critic_targets, critic_value)
        critic_grad = tape.gradient(
            critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(
            zip(critic_grad, self.critic.trainable_variables))

        # Update actor network using agent-specific states
        with tf.GradientTape() as tape:
            new_actions = self.actor(agent_states, training=True)
            critic_value = self.critic([states, new_actions], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

        # Soft update target networks
        self.update_target_networks()

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        # Log updates and process
        agent_logger.info(f"Updated epsilon: {self.epsilon}")
        agent_logger.info("Learning process")

    def save(self, directory=MODEL_PATH):
        if not os.path.exists(directory):
            os.makedirs(directory)

        self.actor.save(os.path.join(
            directory, f"{self.agent_idx}/actor_saved_model"))
        self.critic.save(os.path.join(
            directory, f"{self.agent_idx}/critic_saved_model"))
        self.target_actor.save(os.path.join(
            directory, f"{self.agent_idx}/target_actor_saved_model"))
        self.target_critic.save(os.path.join(
            directory, f"{self.agent_idx}/target_critic_saved_model"))

        # Optionally save other attributes like epsilon, etc.
        with open(os.path.join(directory, "agent_state.txt"), 'w') as f:
            f.write(f"epsilon:{self.epsilon}\n")

            # Log the successful save operation
        agent_logger.info("Models and state saved successfully.")

    def load(self, directory=f'{MODEL_PATH}'):
        self.actor = tf.keras.models.load_model(
            os.path.join(directory, f"{self.agent_idx}/actor_saved_model"))
        self.critic = tf.keras.models.load_model(
            os.path.join(directory, f"{self.agent_idx}/critic_saved_model"))
        self.target_actor = tf.keras.models.load_model(
            os.path.join(directory, f"{self.agent_idx}/target_actor_saved_model"))
        self.target_critic = tf.keras.models.load_model(
            os.path.join(directory, f"{self.agent_idx}/target_critic_saved_model"))

        # Optionally load other attributes like epsilon, etc.
        with open(os.path.join(directory, f"{self.agent_idx}/agent_state.txt"), 'r') as f:
            for line in f.readlines():
                if line.startswith("epsilon:"):
                    self.epsilon = float(line.split(":")[1])

         # Log the successful load operation
        agent_logger.info("Models and state loaded successfully.")
