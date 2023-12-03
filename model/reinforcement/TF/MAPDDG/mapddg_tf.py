import os

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import util.loggers as loggers
from model.reinforcement.MAPDDG import buffer, network
from model.reinforcement.rl_visual import plot_and_save_metrics
from util.agent_utils import is_new_record, save_to_csv

from model.reinforcement

logger = loggers.setup_loggers()
agent_logger = logger['agent']
rl_logger = logger['rl']

MODEL_PATH = 'data/reinforcement/'


class MAPDDGAgent:
    def __init__(self, env, gamma=0.95, learning_rate=0.001, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=64, replay_buffer=None, num_agents=2):

        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory = replay_buffer if replay_buffer is not None else buffer.ReplayBuffer(
            100_000, num_agents, state_shape=env.observation_space.shape, action_dim=env.action_space.n)

        # Initialize the centralized critic and its target
        self.centralized_critic = network.CentralizedCriticNetwork(num_agents)
        self.target_critic = tf.keras.models.clone_model(
            self.centralized_critic)
        self.centralized_critic_optimizer = Adam(learning_rate=learning_rate)

        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent = Agent(env.action_space.n, gamma, epsilon, epsilon_min,
                          epsilon_decay, self.memory, batch_size, env.observation_space.shape, learning_rate, i)
            self.agents.append(agent)

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
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample_buffer(
            self.batch_size)

        # Check for NaN in data
        if np.any(np.isnan(states)) or np.any(np.isnan(actions)) or np.any(np.isnan(rewards)) \
           or np.any(np.isnan(next_states)) or np.any(np.isnan(dones)):
            agent_logger.warning("NaN detected in input data, skipping update")
            return

        # Reshape states and actions for critic input
        joint_state = states.reshape(self.batch_size, -1)
        joint_action = actions.reshape(self.batch_size, -1)

        with tf.GradientTape() as tape:
            # Current critic value
            current_critic_value = self.centralized_critic(
                [joint_state, joint_action], training=True)

            # Calculate target actions for the next state using target actors of each agent
            target_actions = [agent.target_actor(
                next_states[:, i, :], training=False) for i, agent in enumerate(self.agents)]
            target_joint_action = tf.concat(target_actions, axis=1)

            # Target Q-values
            target_joint_state = next_states.reshape(self.batch_size, -1)
            target_critic_value = self.target_critic(
                [target_joint_state, target_joint_action], training=False)
            target_q_value = rewards + self.gamma * \
                (1 - dones) * target_critic_value

            # Critic loss
            critic_loss = tf.keras.losses.mean_squared_error(
                target_q_value, current_critic_value)

        # Compute gradients and update centralized critic
        critic_grad = tape.gradient(
            critic_loss, self.centralized_critic.trainable_variables)

        # Check for NaN in gradients
        if any(np.any(np.isnan(g.numpy())) for g in critic_grad if g is not None):
            agent_logger.warning("NaN detected in gradients, skipping update")
            return

        self.centralized_critic_optimizer.apply_gradients(
            zip(critic_grad, self.centralized_critic.trainable_variables))

        # Update each agent's actor network
        for agent in self.agents:
            agent.learn(states, actions, self.centralized_critic)
            self.epsilon = agent.decay_epsilon()

        # Soft update target networks
        self.update_target_networks()

    def save_models(self, directory=MODEL_PATH):
        if not os.path.exists(directory):
            os.makedirs(directory)

        for i, agent in enumerate(self.agents):
            agent_directory = os.path.join(directory, f"agent_{i}")
            agent.save(agent_directory)

        # Save centralized critic models
        self.centralized_critic.save(os.path.join(
            directory, "centralized_critic_saved_model"))
        self.target_critic.save(os.path.join(
            directory, "target_critic_saved_model"))

        agent_logger.info("All models saved successfully.")

    def load_models(self, directory=MODEL_PATH):
        for i, agent in enumerate(self.agents):
            agent_directory = os.path.join(directory, f"agent_{i}")
            agent.load(agent_directory)

        # Load centralized critic models
        self.centralized_critic = tf.keras.models.load_model(
            os.path.join(directory, "centralized_critic_saved_model"))
        self.target_critic = tf.keras.models.load_model(
            os.path.join(directory, "target_critic_saved_model"))

        agent_logger.info("All models loaded successfully.")

    def update_target_networks(self, tau=0.005):
        # Soft update the target networks
        for agent in self.agents:
            new_actor_weights = [(tau * aw + (1 - tau) * tw) for aw, tw in zip(
                agent.actor.get_weights(), agent.target_actor.get_weights())]
            agent.target_actor.set_weights(new_actor_weights)

        new_critic_weights = [(tau * cw + (1 - tau) * tw) for cw, tw in zip(
            self.centralized_critic.get_weights(), self.target_critic.get_weights())]
        self.target_critic.set_weights(new_critic_weights)
        # Log the soft update of target networks
        agent_logger.info("Soft update of target networks")


class Agent():
    def __init__(self, action_dim, gamma, epsilon, epsilon_min, epsilon_decay, shared_memory, batch_size, osn, learning_rate, agent_number):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.env_actions = action_dim
        self.agent_idx = agent_number
        self.memory = shared_memory
        self.batch_size = batch_size
        self.osn = osn

        # Actor network for this agent
        self.actor = network.ActorNetwork(action_dim)

        # Target actor network for this agent
        self.target_actor = tf.keras.models.clone_model(self.actor)

        self.actor_optimizer = Adam(learning_rate=learning_rate)

        # If you plan to use separate optimizers for target networks
        self.target_actor_optimizer = Adam(learning_rate=learning_rate)

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
        return self.epsilon

    def learn(self, states, actions, centralized_critic):
        agent_states = states[:, self.agent_idx, :]

        if np.any(np.isnan(agent_states)):
            agent_logger.warning(
                f"NaN detected in Agent {self.agent_idx} data, skipping update")
            return

        with tf.GradientTape() as tape:
            new_actions = self.actor(agent_states, training=True)
            agent_logger.info(
                f"the new actions of the actor {new_actions} & shape {new_actions.shape}")

            # Assuming the critic expects flattened states and actions
            batch_size = states.shape[0]
            state_dim = np.prod(states.shape[1:])
            action_dim = np.prod(actions.shape[1:])

            flattened_states = tf.reshape(states, [batch_size, state_dim])
            modified_joint_action = actions.copy()
            modified_joint_action[:, self.agent_idx] = new_actions.numpy()
            flattened_actions = tf.reshape(modified_joint_action, [
                                           batch_size, action_dim])

            agent_logger.info(
                f"the shape of state agent{self.agent_idx}: {flattened_states.shape} & shape of action {flattened_actions.shape}")

            critic_evaluation = centralized_critic(
                [flattened_states, flattened_actions], training=False)
            agent_logger.info(
                f"the critic evaluation {critic_evaluation}")
            actor_loss = -tf.math.reduce_mean(critic_evaluation)

        # Compute gradients and update actor network
        actor_grad = tape.gradient(actor_loss, self.actor.trainable_variables)

        # Print gradients to debug
        for var, grad in zip(self.actor.trainable_variables, actor_grad):
            agent_logger.info(f"ACTOR {var.name}, gradient: {grad}")

        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.trainable_variables))

        agent_logger.info(f"Updated epsilon: {self.epsilon}")
        agent_logger.info("Learning process")

    def save(self, directory=MODEL_PATH):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.actor.save(os.path.join(
            directory, f"{self.agent_idx}/actor_saved_model"))
        self.target_actor.save(os.path.join(
            directory, f"{self.agent_idx}/target_actor_saved_model"))

        # Optionally save other attributes like epsilon, etc.
        with open(os.path.join(directory, "agent_state.txt"), 'w') as f:
            f.write(f"epsilon:{self.epsilon}\n")

            # Log the successful save operation
        agent_logger.info("Models and state saved successfully.")

    def load(self, directory=f'{MODEL_PATH}'):
        self.actor = tf.keras.models.load_model(
            os.path.join(directory, f"{self.agent_idx}/actor_saved_model"))
        self.target_actor = tf.keras.models.load_model(
            os.path.join(directory, f"{self.agent_idx}/target_actor_saved_model"))

        # Optionally load other attributes like epsilon, etc.
        with open(os.path.join(directory, f"{self.agent_idx}/agent_state.txt"), 'r') as f:
            for line in f.readlines():
                if line.startswith("epsilon:"):
                    self.epsilon = float(line.split(":")[1])

         # Log the successful load operation
        agent_logger.info("Models and state loaded successfully.")
