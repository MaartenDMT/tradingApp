import numpy as np

from model.reinforcement.env import Environment
from model.reinforcement.MAPDDG.agent2 import MAPDDGAgent
from model.reinforcement.visual_plot import plotLearning
from util.utils import load_config

config = load_config()
param_grid = {
    'modelname': str(config['Params']['modelname']),
    'gamma': float(config['Params']['gamma']),
    'hidden_units': int(config['Params']['hidden_units']),
    'learning_rate': float(config['Params']['learning_rate']),
    'batch_size': int(config['Params']['batch_size']),
    'epsilon_min': float(config['Params']['epsilon_min']),
    'epsilon_decay': float(config['Params']['epsilon_decay']),
    'dropout': float(config['Params']['dropout']),
    'act': str(config['Params']['act']),  # softmax, argmax
    # linear, tanh, sigmoid
    # 'm_activation': str(config['Params']['m_activation']),

    # 'loss': str(config['Params']['loss'])
}

params = {
    'episodes': int(config['Params']['episodes']),
    'env_actions': int(config['Params']['env_actions']),
    'test_episodes': int(config['Params']['test_episodes']),
    'min_acc': float(config['Params']['min_acc']),
}

if __name__ == '__main__':
    # tf.compat.v1.disable_eager_execution()
    # env = gym.make("LunarLander-v2")
    env = Environment(symbol='BTCUSDT', features=[
                      'open', 'high', 'low', 'close', 'volume'], limit=300, time="30m", actions=3, min_acc=55)
    n_games = 400

    agent = MAPDDGAgent(**param_grid, env=env)

    scores = []
    eps_history = []
    best_score = None
    load_checkpoint = False

    if load_checkpoint:
        agent.load_models()

    for i in range(n_games):
        done = False
        score = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, info, done = env.step(action)
            score += reward
            agent.store_transition(observation, action,
                                   reward, observation_, done)
            observation = observation_
            if not load_checkpoint:
                agent.learn()

        eps_history.append(agent.epsilon)
        scores.append(score)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: %.2f' % score,
              ' average score %.2f' % avg_score,
              'epsilon %.2f' % agent.epsilon)

        if best_score == None:
            best_score = avg_score

        if avg_score > best_score:
            best_score = avg_score
            if not load_checkpoint:
                agent.save_models()

    if not load_checkpoint:
        filename = 'dueling_dqn_keras.png'
        x = [i+1 for i in range(n_games)]
        plotLearning(x, scores, eps_history, filename)
