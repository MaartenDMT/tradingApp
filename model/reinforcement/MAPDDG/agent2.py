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

        self.centralized_critic = network.CentralizedCriticNetwork(
            num_agents)
        self.target_critic = tf.keras.models.clone_model(
            self.centralized_critic)

        # Initialize optimizers
        self.centralized_critic_optimizer = Adam(learning_rate=learning_rate)
        # Assuming you also have an actor optimizer
        self.actor_optimizer = Adam(learning_rate=learning_rate)

        # Create a replay buffer if not provided
        if replay_buffer is None:
            self.memory = buffer.ReplayBuffer(
                100_000, num_agents, state_shape=self.osn, action_dim=self.env_actions)

        # Create agents and pass optimizers
        self.agents = [Agent(self.env_actions, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, self.memory, batch_size, self.osn, i, self.actor_optimizer, self.centralized_critic_optimizer)
                       for i in range(num_agents)]

        # Update target networks
        self.update_target_networks(tau=1)

    def store_transitions(self, observations, actions, rewards, next_observations, dones):
        # Store transitions for each agent
        for i, agent in enumerate(self.agents):
            agent.store_transition(
                observations[i], actions[i], rewards[i], next_observations[i], dones[i], i)

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

        # Flatten the second and third dimensions for states and actions
        states_flattened = states.reshape(
            self.batch_size, -1)  # Reshaping to (32, 2*6)
        actions_flattened = actions.reshape(
            self.batch_size, -1)  # Reshaping to (32, 2*5)

        # Now concatenate along the last dimension
        joint_state = np.concatenate([states_flattened], axis=-1)
        joint_action = np.concatenate([actions_flattened], axis=-1)

        agent_logger.info(
            f"Joint state shape after reshaping: {joint_state.shape}")
        agent_logger.info(
            f"Joint action shape after reshaping: {joint_action.shape}")

        for agent in self.agents:
            agent.learn(states, actions, rewards, next_states,
                        dones, joint_state, joint_action, self.centralized_critic, self.target_critic)
            agent.decay_epsilon()
        # Soft update target networks
        self.update_target_networks()

    def save(self, directory=MODEL_PATH):
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, agent in enumerate(self.agents):
            agent.save(os.path.join(directory, str(i)))

        # Log the successful save operation
        agent_logger.info("Models and state saved successfully.")

    def load(self, directory=MODEL_PATH):
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(directory, str(i)))

        # Log the successful load operation
        agent_logger.info("Models and state loaded successfully.")

    def update_target_networks(self, tau=0.005):
        # Update weights with a factor of tau for soft updates
        for i, agent in enumerate(self.agents):
            new_actor_weights = [(tau * aw + (1 - tau) * tw) for aw, tw in zip(
                agent.actor.get_weights(), agent.target_actor.get_weights())]
            agent.target_actor.set_weights(new_actor_weights)

        new_critic_weights = [(tau * cw + (1 - tau) * tw) for cw, tw in zip(
            self.centralized_critic.get_weights(), self.target_critic.get_weights())]
        self.target_critic.set_weights(new_critic_weights)

        # Log the soft update of target networks
        agent_logger.info("Soft update of target networks")


class Agent():
    def __init__(self, action_dim, learning_rate, gamma, epsilon, epsilon_min, epsilon_decay, shared_memory, batch_size, osn, agent_number, actor_optimizer, centralized_critic_optimizer):

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env_actions = action_dim
        self.agent_idx = agent_number
        self.memory = shared_memory
        self.batch_size = batch_size
        self.osn = osn

        self.actor = network.ActorNetwork(action_dim)
        self.target_actor = tf.keras.models.clone_model(self.actor)

        # Store the optimizers
        self.actor_optimizer = actor_optimizer
        self.centralized_critic_optimizer = centralized_critic_optimizer

        self.actor.compile(optimizer=Adam(learning_rate=learning_rate))
        self.target_actor.compile(optimizer=Adam(learning_rate=learning_rate))

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

    def store_transition(self, state, action, reward, new_state, done, i):
        self.memory.store_transition(state, action, reward, new_state, done, i)
        # Log the stored transition
        agent_logger.info(
            f"Stored transition agent {i}: action={action}, reward={reward}, done={done}")

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

    def learn(self, states, actions, rewards, next_states, dones, joint_state, joint_action, centralized_critic, target_critic):
        if self.memory.mem_cntr < self.batch_size:
            return
        self.centralized_critic = centralized_critic
        self.target_critic = target_critic

        agent_logger.info(f"origenal agent shape: {rewards.shape}")
        # Agent-specific states, actions, rewards, and dones
        agent_states = states[:, self.agent_idx, :]
        agent_next_states = next_states[:, self.agent_idx, :]
        agent_actions = actions[:, self.agent_idx]
        agent_rewards = rewards[:, self.agent_idx]
        agent_dones = dones[:, self.agent_idx]

        # Update the centralized critic
        with tf.GradientTape() as tape:
            critic_value = self.centralized_critic(
                [joint_state, joint_action], training=True)
            agent_logger.info(f"agent shape: {rewards.shape}")
            agent_logger.info(f"critic shape: {critic_value.shape}")

            critic_loss = tf.keras.losses.mean_squared_error(
                rewards, critic_value)
        critic_grad = tape.gradient(
            critic_loss, self.centralized_critic.trainable_variables)
        self.centralized_critic_optimizer.apply_gradients(
            zip(critic_grad, self.centralized_critic.trainable_variables))

        # Update the actor network of the current agent
        with tf.GradientTape() as tape:
            # Predict new actions based on the current agent's state
            new_actions = self.actor(agent_states, training=True)

            # Assuming self.agent_idx corresponds to the index of the current agent
            # and each agent has an action dimension of 5 (since new_actions shape is (32, 5))
            # Calculate total number of agents
            num_agents = joint_action.shape[1] // 5
            action_dim_per_agent = 5

            # Calculate the start and end indices for slicing
            start_idx = self.agent_idx * action_dim_per_agent
            end_idx = start_idx + action_dim_per_agent

            # Replace the actions of the current agent in joint_action with new_actions
            new_joint_action = np.concatenate([
                # Actions before the current agent
                joint_action[:, :start_idx],
                new_actions,      # New actions for the current agent
                # Actions after the current agent
                joint_action[:, end_idx:]
            ], axis=-1)
            agent_logger.info(f"new agent actions shape: {new_actions.shape}")
            agent_logger.info(
                f"the joint action shape : {new_joint_action.shape}")
            agent_logger.info(f"the joint state shape : {joint_state.shape}")

            critic_value = self.centralized_critic(
                [joint_state, new_joint_action], training=True)
            actor_loss = -tf.math.reduce_mean(critic_value)
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

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
