import optuna
import pandas as pd

import util.loggers as loggers
from model.reinforcement.agent import DQLAgent
from model.reinforcement.env import Environment
from util.utils import load_config

logger = loggers.setup_loggers()
rl_logger = logger['rl']
config = load_config()

CSV_PATH = "data/best_model/best_params.csv"

params = {key: config['Params'][key] for key in config['Params']}

env = Environment(symbol='BTCUSDT', features=['close'],
                  limit=300, time="30m", actions=3, min_acc=float(params['min_acc']))


def objective(trial):
    hyperparameters = {
        'modelname': trial.suggest_categorical('modelname', ["Standard_Model", "Dense_Model", "LSTM_Model", "CONV1D_LSTM_Model"]),
        'gamma': trial.suggest_float('gamma', 0.9, 0.99),
        'hu': trial.suggest_int('hu', 16, 32),
        'lr': trial.suggest_float('lr', 0.001, 0.1, log=True),
        'epsilon': trial.suggest_float('epsilon', 0.9, 1.1),
        'dropout': trial.suggest_float('dropout', 0.1, 0.3),
        'act': trial.suggest_categorical('act', ['argmax', 'softmax']),
        'm_activation': trial.suggest_categorical('m_activation', ['linear', 'tanh', 'sigmoid'])
    }

    agent = DQLAgent(env=env, **hyperparameters)
    agent.learn(episodes=10)
    test_rewards = agent.test(episodes=10)
    avg_test_reward = sum(test_rewards) / 10

    return avg_test_reward


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=2)

# Get the best parameters and use them
best_params = study.best_params

# Fetch all completed trials and sort them based on value
completed_trials = study.get_trials(
    deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])
top_10_trials = sorted(
    completed_trials, key=lambda trial: trial.value, reverse=True)[:10]

# Extract parameters from the top 10 trials
top_10_params = [trial.params for trial in top_10_trials]

# Save top_10_params to CSV
top_10_params_df = pd.DataFrame(top_10_params)
# 'top_10_params.csv' is the filename. You can change it if needed.
top_10_params_df.to_csv('top_5_params.csv', index=False)

agent = DQLAgent(env=env, **best_params)
agent.learn(episodes=100)
agent.save_model()
test = agent.test(episodes=20)
avg_test = sum(test) / 20
print(avg_test)

# TODO: add matplotlib and money function to the programm
