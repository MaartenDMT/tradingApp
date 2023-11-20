import numpy as np


class Replaybuffer:
    def __init__(self, max_size, num_agents, state_shape, action_dim):
        self.mem_size = max_size
        self.num_agents = num_agents
        self.mem_cntr = 0
        self.state_memory = np.zeros(
            (self.mem_size, num_agents, *state_shape), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.mem_size, num_agents, *state_shape), dtype=np.float32)
        self.action_memory = np.zeros(
            (self.mem_size, num_agents, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(
            (self.mem_size, num_agents), dtype=np.float32)
        self.terminal_memory = np.zeros(
            (self.mem_size, num_agents), dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        for i in range(self.num_agents):
            self.state_memory[index, i] = state[i]
            self.new_state_memory[index, i] = state_[i]
            self.action_memory[index, i] = action[i]
            self.reward_memory[index, i] = reward[i]
            self.terminal_memory[index, i] = done[i]
        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        new_states = self.new_state_memory[batch]
        terminals = self.terminal_memory[batch]

        return states, actions, rewards, new_states, terminals
