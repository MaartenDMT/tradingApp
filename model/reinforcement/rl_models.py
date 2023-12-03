import model.reinforcement.TF.MAPDDG as MAPDDG
import model.reinforcement.TF.DDQN as DDQN
import model.reinforcement.TF.SAC as SAC
import model.reinforcement.TF.AC as AC
import model.reinforcement.TF.PG as PG
import model.reinforcement.TF.PPO as PPO

from model.reinforcement.env import MultiAgentEnvironment
from model.reinforcement.rl_visual import plot_and_save_metrics, plotting
from model.reinforcement.visual_plot import plotLearning
import numpy as np


def MAPDDG(params, rl_logger):
    num_agents = 2
    # Initialize Environment and Agent with specific hyperparamseters
    env = MultiAgentEnvironment(num_agents=num_agents, symbol='BTCUSDT', features=params['features'],
                                limit=300, time="30m", actions=params['env_actions'], min_acc=params['min_acc'])

    # rl_logger.info(
    #     f"Performance with data {env.env_data}")
    rl_logger.info(
        f"observation space: {env.observation_space.shape}")
    rl_logger.info(
        f"look back: {env.look_back}")
    rl_logger.info(params)

    agent = MAPDDG.mapddg_tf.MAPDDGAgent(
        **params, env=env, num_agents=num_agents)

    scores = []
    actions = []
    eps_history = []
    best_scores = [None] * num_agents
    load_checkpoint = False

    if load_checkpoint:
        agent.load()

    for i in range(params['episodes']):
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
            rl_logger.info(
                'episode: %d agent: %d score: %.2f average score %.2f epsilon %.2f',
                i, j, episode_scores[j], avg_scores[j], agent.epsilon)

            if best_scores[j] is None or avg_scores[j] > best_scores[j]:
                best_scores[j] = avg_scores[j]
                if not load_checkpoint:
                    # Consider saving model for each agent separately
                    agent.save()

        if i % 5 == 0:
            plot_and_save_metrics(scores, actions, i, "standard")

    if not load_checkpoint:
        x = [i+1 for i in range(params['episodes'])]
        plotLearning(x, scores, eps_history)


def SAC(params, rl_logger):
    num_agents = 1
    env = MultiAgentEnvironment(num_agents=num_agents, symbol='BTCUSDT', features=params['features'],
                                limit=300, time="30m", actions=params['env_actions'], min_acc=params['min_acc'])

    agent = SAC.sac_tf.Agent(input_dims=env.observation_space.shape,
                             env=env, n_actions=env.action_space.n)

    n_episodes = 200

    filename = 'SAC_dqn_tf.png'

    best_score = None
    score_history = []
    load_checkpoint = False
    eps_history = []

    if load_checkpoint:
        agent.load_models()

    for i in range(n_episodes):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, info, done = env.step(action)
            score += reward
            agent.remember(observation, action, reward, observation_, done)
            if not load_checkpoint:
                agent.learn()
            observation = observation_

        score_history.append(score)
        avg_score = np.mean(score_history[-100:])
        eps_history.append(agent.epsilon)

        if best_score == None:
            best_score = avg_score

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

        rl_logger('episode: ', i, 'score: %.2f' % score,
                  ' average score %.2f' % avg_score)

    if not load_checkpoint:
        x = [i+1 for i in range(params['episodes'])]
        plotLearning(x, score_history, eps_history)
