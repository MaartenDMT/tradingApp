import numpy as np

import model.reinforcement.TF.AC.ac_tf as AC
import model.reinforcement.TF.DDQN.ddqn_tf as DDQN
import model.reinforcement.TF.MAPDDG.mapddg_tf as MAPDDG
import model.reinforcement.TF.PG.pg_tf as PG
import model.reinforcement.TF.PPO.ppo_tf as PPO
import model.reinforcement.TF.SAC.sac_tf as SAC
from model.reinforcement.env import MultiAgentEnvironment
from model.reinforcement.rl_visual import plot_and_save_metrics, plotting
from model.reinforcement.visual_plot import plot_learning_curve, plotLearning


def mappddg(params, rl_logger):
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

    agent = MAPDDG.MAPDDGAgent(
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
            plot_and_save_metrics(num_agents, scores, actions, i, "mapddg")

    if not load_checkpoint:
        x = [i+1 for i in range(params['episodes'])]
        plotLearning(x, scores, eps_history)


def sac(params, rl_logger):
    num_agents = 1
    env = MultiAgentEnvironment(num_agents=num_agents, symbol='BTCUSDT', features=params['features'],
                                limit=300, time="30m", actions=params['env_actions'], min_acc=params['min_acc'])

    agent = SAC.Agent(input_dims=env.observation_space.shape,
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

        rl_logger.info(
            'episode: %d, score: %.2f, average score %.2f', i, scores, avg_score)

    if not load_checkpoint:
        x = [i + 1 for i in range(n_episodes)]
        plot_learning_curve(x, score_history, filename)
