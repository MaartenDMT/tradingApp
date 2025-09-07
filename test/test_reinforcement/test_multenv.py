import unittest

from model.reinforcement import TradingEnvironment


class TestMultiAgentEnvironment(unittest.TestCase):

    def setUp(self):
        # Set up with a specific number of agents
        self.num_agents = 3
        # Dummy arguments for Environment
        self.args = ('BTCUSDT', ["close", "open"], 10, "30m", 5, 0.5)
        self.env = TradingEnvironment(*self.args)

    def test_initialization(self):
        # Test basic initialization
        self.assertIsNotNone(self.env)

    def test_reset_single_agent(self):
        single_agent_env = TradingEnvironment(*self.args)
        state = single_agent_env.reset()
        self.assertIsNotNone(state)

    def test_reset_multiple_agents(self):
        state = self.env.reset()
        self.assertIsNotNone(state)

    def test_step_single_agent(self):
        single_agent_env = TradingEnvironment(*self.args)
        action = 1  # Simple action for testing
        state, reward, info, done = single_agent_env.step(action)
        self.assertIsNotNone(state)

    def test_step_multiple_agents(self):
        action = 1  # Simple action for testing
        state, reward, info, done = self.env.step(action)
        self.assertIsNotNone(state)
        self.assertIsNotNone(reward)
        self.assertIsNotNone(info)
        self.assertIsNotNone(done)

    def test_get_action_space(self):
        # Test that environment has proper action space
        self.assertIsNotNone(self.env)

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
    unittest.main()
