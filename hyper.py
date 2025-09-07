import optuna
import pandas as pd

import util.loggers as loggers
from model.reinforcement import TradingEnvironment
from model.reinforcement.agents.agent_manager import DQLAgent
from util.utils import load_config

# Logger and Configuration Setup
logger = loggers.setup_loggers()
rl_logger = logger['rl']
config = load_config()

# Environment Setup
env = TradingEnvironment(symbol='BTCUSDT', features=['open', 'high', 'low', 'close', 'volume'],
                  limit=300, time="30m", actions=3, min_acc=float(config['Params']['min_acc']))

# Objective Function for Optuna


def objective(trial):
    hyperparameters = {
        'modelname': trial.suggest_categorical('modelname', ["Standard_Model", "Dense_Model", "LSTM_Model", "CONV1D_LSTM_Model", "build_resnet_model", "base_conv1d_model", "base_transformer_model"]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99),
        'hidden_units': trial.suggest_int('hidden_units', 16, 32, 64),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.9, 1.1),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'act': trial.suggest_categorical('act', ['argmax', 'softmax']),
        'm_activation': trial.suggest_categorical('m_activation', ['linear', 'tanh', 'sigmoid'])
    }

    agent = DQLAgent(env=env, **hyperparameters)
    agent.learn(episodes=5)
    test_rewards = agent.test(episodes=5)
    return sum(test_rewards) / len(test_rewards)


# Study Creation and Optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

# Result Processing
best_params = study.best_params
top_10_trials = sorted(study.get_trials(states=[
                       optuna.trial.TrialState.COMPLETE]), key=lambda trial: trial.value, reverse=True)[:10]
top_10_params = [trial.params for trial in top_10_trials]
top_10_params_df = pd.DataFrame(top_10_params)
top_10_params_df.to_csv('top_5_params.csv', index=False)

# Testing the Agent
agent = DQLAgent(env=env, **best_params)
agent.learn(episodes=50)
agent.save_model()
test_rewards = agent.test(episodes=20)
avg_test_reward = sum(test_rewards) / len(test_rewards)
print(avg_test_reward)
