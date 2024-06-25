import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import A2C, DQN, PPO, TD3

# Tensorflow models
import model.reinforcement.TF.AC.ac_tf as AC_tf
import model.reinforcement.TF.DDQN.ddqn_tf as DDQN_tf
import model.reinforcement.TF.MAPDDG.mapddg_tf as MAPDDG_tf
import model.reinforcement.TF.PG.pg_tf as PG_tf
import model.reinforcement.TF.PPO.ppo_tf as PPO_tf
import model.reinforcement.TF.SAC.sac_tf as SAC_tf
# Pytorch models
import model.reinforcement.TORCH.A3C.a3c_torch as A3C_torch
import model.reinforcement.TORCH.DDG.ddg_torch as DDG_torch
import model.reinforcement.TORCH.PPO.ppo_torch as PPO_torch
from model.reinforcement.env import MultiAgentEnvironment
from model.reinforcement.rl_env.trading_env import TradingEnvironment
from model.reinforcement.rl_util import generate_random_candlestick_data
from model.reinforcement.rl_visual import plot_and_save_metrics, plotting
from model.reinforcement.visual_plot import plot_learning_curve, plotLearning

# import model.reinforcement.TORCH.ICM.parrallel_env as ICM_torch


class TensorflowModel:
    def __init__(self, params, rl_logger, num_agents=1) -> None:
        self.num_agents = num_agents
        self.env = MultiAgentEnvironment(num_agents=num_agents, symbol='BTCUSDT', features=params['features'],
                                         limit=300, time="30m", actions=params['env_actions'], min_acc=params['min_acc'])
        self.rl_logger = rl_logger
        self.params = params

        # rl_logger.info(
        #     f"Performance with data {env.env_data}")
        rl_logger.info(
            f"observation space: {self.env.observation_space.shape}")
        rl_logger.info(
            f"look back: {self.env.look_back}")
        rl_logger.info(params)

    def num_agents(self, num_agents=1):
        self.num_agents = num_agents

    def mappddg(self):
        num_agents = self.num_agents(2)
        # Initialize Environment and Agent with specific hyperparamseters
        env = self.env

        agent = MAPDDG_tf.MAPDDGAgent(
            **self.params, env=env, num_agents=num_agents)

        scores = []
        actions = []
        eps_history = []
        best_scores = [None] * num_agents
        load_checkpoint = False

        if load_checkpoint:
            agent.load()

        for i in range(self.params['episodes']):
            done = [False] * num_agents
            episode_scores = [0] * num_agents
            observation = env.reset()  # Observations for all agents
            while not all(done):
                agent_actions = agent.choose_actions(observation)
                observation_, reward, info, done = env.step(agent_actions)

                # Check if reward is a scalar and convert it to a list if necessary
                if np.isscalar(reward):
                    observation = [observation] * num_agents
                    observation_ = [observation_] * num_agents
                    agent_actions = [agent_actions] * num_agents
                    reward = [reward] * num_agents
                    done = [done] * num_agents

                for j in range(num_agents):
                    episode_scores[j] += reward[j]
                agent.store_transitions(
                    observation, agent_actions, reward, observation_, done)
                observation = observation_

            if not load_checkpoint:
                agent.learn()

            eps_history.append(agent.epsilon)
            scores.append(episode_scores)  # Track scores for each agent

            avg_scores = [np.mean([score[j] for score in scores[-100:]])
                          for j in range(num_agents)]

            # Logging and checking for best scores
            for j in range(num_agents):
                self.rl_logger.info(
                    'episode: %d agent: %d score: %.2f average score %.2f epsilon %.2f',
                    i, j, episode_scores[j], avg_scores[j], agent.epsilon)

                if best_scores[j] is None or avg_scores[j] > best_scores[j]:
                    best_scores[j] = avg_scores[j]
                    if not load_checkpoint:
                        # Consider saving model for each agent separately
                        agent.save()

            if i % 5 == 0:
                plot_and_save_metrics(num_agents, scores, actions, i, "mapddg")

        if not load_checkpoint:
            x = [i+1 for i in range(self.params['episodes'])]
            plotLearning(x, scores, eps_history)

    def sac(self):
        num_agents = 1
        env = self.env

        agent = SAC_tf.Agent(input_dims=env.observation_space.shape,
                             env=env, n_actions=env.action_space.n)

        n_episodes = 200

        filename = 'data/png/reinforcement/SAC_dqn_tf.png'

        best_score = None
        score_history = []
        load_checkpoint = False

        if load_checkpoint:
            agent.load_models()

        for i in range(n_episodes):
            done = False
            scores = 0
            observation = env.reset()
            while not done:
                action = agent.choose_action(observation)
                observation_, reward, info, done = env.step(action)
                scores += reward
                agent.remember(observation, action, reward, observation_, done)
                if not load_checkpoint:
                    agent.learn()
                observation = observation_

            score_history.append(scores)
            avg_score = np.mean(score_history[-100:])

            if best_score == None:
                best_score = avg_score

            if avg_score > best_score:
                best_score = avg_score
                if not load_checkpoint:
                    agent.save_models()

            if i % 5 == 0:
                plot_and_save_metrics(num_agents, scores, action, i, "sac")

            if i % 10 == 0:
                env.update_and_adjust_features()

            self.rl_logger.info(
                'episode: %d, score: %.2f, average score %.2f', i, scores, avg_score)

        if not load_checkpoint:
            x = [i + 1 for i in range(n_episodes)]
            plot_learning_curve(x, score_history, filename)


class TorchModel:
    def __init__(self, params, rl_logger, num_agents=1):
        self.num_agents = num_agents
        self.env = MultiAgentEnvironment(num_agents=num_agents, symbol='BTCUSDT', features=params['features'],
                                         limit=300, time="30m", actions=params['env_actions'], min_acc=params['min_acc'])
        self.rl_logger = rl_logger
        self.params = params

        # rl_logger.info(
        #     f"Performance with data {env.env_data}")
        rl_logger.info(
            f"observation space: {self.env.observation_space.shape}")
        rl_logger.info(
            f"look back: {self.env.look_back}")
        rl_logger.info(params)

    def num_agents(self, num_agents=1):
        self.num_agents = num_agents

    def a3c(self):
        lr = 1e-4
        env_id = self.env
        n_actions = env_id.action_space.n
        input_dims = env_id.observation_space.shape

        global_actor_critic = A3C_torch.ActorCritic(input_dims, n_actions)
        global_actor_critic.share_memory()
        optim = A3C_torch.SharedAdam(global_actor_critic.parameters(), lr=lr,
                                     betas=(0.92, 0.999))
        global_ep = A3C_torch.global_ep()

        workers = [A3C_torch.Agent(global_actor_critic,
                                   optim,
                                   input_dims,
                                   n_actions,
                                   gamma=0.99,
                                   lr=lr,
                                   name=i,
                                   global_ep_idx=global_ep,
                                   env_id=env_id) for i in range(A3C_torch.cpu_count())]
        [w.start() for w in workers]
        [w.join() for w in workers]

    def ppo(self):
        env = self.env
        N = 20
        batch_size = 5
        n_epochs = 4
        alpha = 0.0003
        agent = PPO_torch.Agent(n_actions=env.action_space.n, batch_size=batch_size,
                                alpha=alpha, n_epochs=n_epochs,
                                input_dims=env.observation_space.shape)
        n_games = 300

        best_score = env.reward_range[0]
        score_history = []

        learn_iters = 0
        avg_score = 0
        n_steps = 0

        for i in range(n_games):
            observation = env.reset()
            done = False
            score = 0
            while not done:
                action, prob, val = agent.choose_action(observation)
                observation_, reward, done, info = env.step(action)
                n_steps += 1
                score += reward
                agent.remember(observation, action, prob, val, reward, done)
                if n_steps % N == 0:
                    agent.learn()
                    learn_iters += 1
                observation = observation_
            score_history.append(score)
            avg_score = np.mean(score_history[-100:])

            if avg_score > best_score:
                best_score = avg_score
                agent.save_models()

            self.rl_logger.info('episode {} score {:.1f} avg score {:.1f} time_steps {} learning_steps {}'.format(
                i, score, avg_score, n_steps, learn_iters))

        x = [i+1 for i in range(len(score_history))]
        figure_file = 'data/png/reinforcement/ppo_torch.png'
        plot_learning_curve(x, score_history, figure_file)

    def ddg(self):
        env = self.env
        agent = DDG_torch.Agent(alpha=0.000025, beta=0.00025, input_dims=env.observation.shape, tau=0.001, env=env,
                                batch_size=64,  layer1_size=400, layer2_size=300, n_actions=env.action_space.n)

        # agent.load_models()
        np.random.seed(0)

        score_history = []
        for i in range(1000):
            obs = env.reset()
            done = False
            score = 0
            while not done:
                act = agent.choose_action(obs)
                new_state, reward, done, info = env.step(act)
                agent.remember(obs, act, reward, new_state, int(done))
                agent.learn()
                score += reward
                obs = new_state
                # env.render()
            score_history.append(score)

            # if i % 25 == 0:
            #    agent.save_models()

            self.rl_logger.info('episode ', i, 'score %.2f' % score,
                                'trailing 100 games avg %.3f' % np.mean(score_history[-100:]))

        filename = 'data/png/reinforcement/ddg_torch.png'
        plotLearning(score_history, filename, window=100)

    # def icm(self):
    #     ICM_torch.start_method()
    #     env_id = self.env
    #     n_threads = 12
    #     n_actions = env.action_space.n
    #     input_shape = env.observation.shape
    #     env = ICM_torch.ParallelEnv(env_id=env_id, n_threads=n_threads, n_actions=n_actions,
    #                                 input_shape=input_shape, icm=True, rl_logger=self.rl_logger)


class StablebaselineModel:
    def __init__(self, params, rl_logger, num_agents=1):
        data, time = generate_random_candlestick_data(
            6000, initial_price=100, min_volatility=0.005, max_volatility=0.1, max_shadow_amplitude=0.7, max_volume_multiplier=10.0, seed=42, bias_factor=0.002)
        self.env = TradingEnvironment(None, initial_balance=1000)
        self.model = A2C("MlpPolicy", self.env, verbose=1)
        self.model.learn(total_timesteps=20_000, log_interval=100)
        self.rl_logger = rl_logger
        self.params = params
        env_rec = self.model.get_env()
        obs = env_rec.reset()
        done = False
        infos = []

        while not done:
            # action = env_rec.action_space.sample()  # random agent action
            action, _state = self.model.predict(obs)
            self.rl_logger.info(
                f"sampled Action: {action[0]} and type of {type(action)}")
            obs, reward, terminated, done, info = self.env.step(action)
            infos.append(info)
            env_rec.render("human")
            self.rl_logger.info(
                f"action: {action}, Balance: {info['balance']}, shares: {info['current_position']}, Price:{info['current_price']}, Total Worth: {info['current_total_worth']}, reward: {reward}")


        time_steps = range(len(infos))
        returns = [info['balance'] for info in infos]
        share_prices = [info['current_price'] for info in infos]
        total_worth = [info['current_total_worth'] for info in infos]

        # create a figure with two y-axes
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # plot the first data series(returns) on the left axis
        ax1.plot(time_steps, returns, label='Cumulative Returns', color='blue')
        ax1.set_xlabel('Time steps')
        ax1.set_ylabel("Cumulative Return", color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')

        # create a second y-axis on the right for share prices and total worth
        ax2 = ax1.twinx()

        ax2.plot(time_steps, share_prices, label='Share Prices', color='green')
        ax2.plot(time_steps, total_worth, label='Total Worth', color='red')
        ax2.set_ylabel('Price/Worth', color='green')
        ax2.tick_params(axis='y', labelcolor='green')

        # Collect all the handles and labels from both axes
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()

        # Combine the handles and labels and create a single legend
        handles = handles1 + handles2
        labels = labels1 + labels2

        # Add the legend to the plot
        fig.legend(handles, labels, loc='upper left',
                   bbox_to_anchor=(0.1, 0.9))

        plt.title('financial date over time')
        plt.grid(True)

        plt.show()

        # rl_logger.info(
        #     f"Performance with data {env.env_data}")
        self.rl_logger.info(
            f"observation space: {env_rec.observation_space.shape}")
        rl_logger.info(params)
