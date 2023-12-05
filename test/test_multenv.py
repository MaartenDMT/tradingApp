import unittest
from model.reinforcement.env import MultiAgentEnvironment, Environment


class TestMultiAgentEnvironment(unittest.TestCase):

    def setUp(self):
        # Set up with a specific number of agents
        self.num_agents = 3
        # Dummy arguments for Environment
        args = ('BTCUSDT', ["close", "open"], 10, "30m", 5, 0.5)
        self.env = MultiAgentEnvironment(self.num_agents, *args)

    def test_initialization(self):
        self.assertEqual(self.env.num_agents, self.num_agents)
        self.assertFalse(self.env.single_agent_mode)
        self.assertEqual(len(self.env.agents), self.num_agents)

    def test_reset_single_agent(self):
        single_agent_env = MultiAgentEnvironment(1, *args)
        state = single_agent_env.reset()
        self.assertIsNotNone(state)

    def test_reset_multiple_agents(self):
        states = self.env.reset()
        self.assertEqual(len(states), self.num_agents)

    def test_step_single_agent(self):
        single_agent_env = MultiAgentEnvironment(1, *args)
        action = single_agent_env.get_action_space().sample()
        state, reward, info, done = single_agent_env.step(action)
        self.assertIsNotNone(state)

    def test_step_multiple_agents(self):
        actions = [self.env.get_action_space().sample()
                   for _ in range(self.num_agents)]
        states, rewards, infos, dones = self.env.step(actions)
        self.assertEqual(len(states), self.num_agents)
        self.assertEqual(len(rewards), self.num_agents)
        self.assertEqual(len(infos), self.num_agents)
        self.assertEqual(len(dones), self.num_agents)

    def test_get_action_space(self):
        action_space = self.env.get_action_space()
        self.assertIsNotNone(action_space)

    def test_get_observation_space(self):
        observation_space = self.env.get_observation_space()
        self.assertIsNotNone(observation_space)

    def test_get_look_back(self):
        look_back = self.env.get_look_back()
        self.assertIsNotNone(look_back)

    def test_update_and_adjust_features(self):
        self.env.update_and_adjust_features()
        # Check if this method runs without errors


if __name__ == '__main__':
    unittest.main()
