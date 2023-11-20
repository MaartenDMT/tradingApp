import pickle
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util.loggers as loggers
from util.rl_util import next_available_filename

logger = loggers.setup_loggers()
logger = logger['rl']


def plotting(agent, env, fn):
    try:
        # Create a single figure
        plt.figure(figsize=(18, 18))

        x = range(len(agent.averages))
        y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
        z = agent.averages

        # Subplot 1: Moving Averages and Regression
        plt.subplot(4, 2, 1)  # 4 rows, 2 columns, 1st plot
        plt.plot(z, label='Moving averages')
        plt.plot(x, y, 'r--', label='Regression')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.legend()
        plt.title('Moving Averages and Regression')

        # Strategy Calculation
        agent.env.data_['returns'] = env.data_['close'].pct_change()
        agent.env.data_['strategy'] = env.data['action'] * env.data_['returns']
        logger.info(f"action:{env.data['action']}")

        # Subplot 2: Strategy
        plt.subplot(4, 2, 2)  # 4 rows, 2 columns, 2nd plot
        agent.env.data_['strategy'].cumsum().plot(label='Strategy')
        plt.xlabel('Time')
        plt.ylabel('Cumulative Return')
        plt.legend()
        plt.title('Strategy Over Time')

        # Subplot 3: Actions during Learning

        logger.info(f"Time steps: {len(agent.env.data)}")
        logger.info(
            f"agent training history shape: {agent.train_action_history}")
        logger.info(
            f"agent.train_action_history[:5]: {agent.train_action_history[:5]}")
        logger.info(
            f"agent.train_action_history[5]: {agent.train_action_history[5]}")

        plt.subplot(4, 2, 3)  # 4 rows, 2 columns, 3rd plot
        actions = np.array(agent.train_action_history[:len(agent.env.data)])
        logger.info(f"Actions shape: {actions.shape}")

        time_steps = np.arange(len(actions))
        for action in np.unique(actions):
            plt.scatter(time_steps[actions == action],
                        actions[actions == action], label=f'Action {action}')
        plt.xlabel('Time step')
        plt.ylabel('Action')
        plt.legend()
        plt.title('Actions during Learning')

        # Subplot 4: Actions during Testing
        plt.subplot(4, 2, 4)  # 4 rows, 2 columns, 4th plot
        actions = np.array(agent.test_action_history[:len(agent.env.data)])
        time_steps = np.arange(len(actions))
        logger.info(f'full action: {len(actions)}')
        logger.info(f'full time steps: {len(time_steps)}')
        for action in np.unique(actions):
            logger.info(f'action: {action}')
            plt.scatter(time_steps[actions == action],
                        actions[actions == action], label=f'Action {action}')
        plt.xlabel('Time step')
        plt.ylabel('Action')
        plt.legend()
        plt.title('Actions during Testing')

        # Subplot 5: Feature-Reward Correlations
        ax5 = plt.subplot(4, 2, 5)  # 4 rows, 2 columns, 5th plot
        calculate_correlations(agent.state_history,
                               agent.reward_history, agent.osn, agent.modelname, fn, ax5)
        # Note: This assumes calculate_correlations plots within the same active figure

        # Save the figure with multiple subplots
        plot_filename = next_available_filename("combined_plot", "png")
        plt.savefig(plot_filename)

        # Example usage for saving DataFrame
        csv_filename = next_available_filename("dataframe", "csv")
        data_df = pd.DataFrame(agent.env.data)
        data_df.to_csv(csv_filename)

        # Example usage for saving agent
        agent_filename = next_available_filename("agent", "pickle")
        with open(agent_filename, 'wb') as f:
            pickle.dump(agent, f)

        # Show the entire figure with subplots
        # plt.show()
    except Exception as e:
        logger.error(f"{e}\n{traceback.format_exc()}")


def calculate_correlations(state_history, reward_history, osn, modelname, fn=None, ax=None):
    # If feature_names is None, use generic names
    if fn is None:
        fn = [f'feature_{i}' for i in range(osn)]

    # Convert state history and reward history to a DataFrame
    logger.info("Length of fn:", len(fn))
    logger.info("Number of columns in state_history:", len(
        state_history[0]) if state_history else "state_history is empty")

    if modelname in ["LSTM_Model", "CONV1D_LSTM_Model"]:
        state_df = pd.DataFrame(state_history.squeeze(), columns=fn)
    else:
        state_df = pd.DataFrame(state_history, columns=fn)

    state_df = pd.DataFrame(state_history, columns=fn)
    reward_df = pd.DataFrame(reward_history, columns=['reward'])

    # Combine them into one DataFrame
    combined_df = pd.concat([state_df, reward_df], axis=1)

    # Compute correlation
    correlation_matrix = combined_df.corr()
    reward_correlations = correlation_matrix['reward'].drop('reward')

    logger.info("Feature-Reward Correlations:")
    logger.info(reward_correlations)

    # Save correlations to a CSV
    csv_filename = next_available_filename("reward_corr", "csv")
    reward_correlations.to_csv(csv_filename)
    logger.info(f"Saved correlations to {csv_filename}")

    # Create a list of colors (same length as reward_correlations)
    colors = plt.cm.viridis(np.linspace(0, 1, len(reward_correlations)))

    # Sort correlations
    sorted_correlations = reward_correlations.sort_values()

    # Plot correlations
    if ax is None:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()  # Get current axis

    sorted_correlations.plot(kind='barh', color=colors)
    plt.xlabel('Correlation Coefficient')
    plt.ylabel('Feature')
    plt.title('Feature-Reward Correlations')
    plt.colorbar(plt.cm.ScalarMappable(cmap=plt.cm.viridis), ax=ax,
                 orientation='vertical', label='Color Scale')


def plot_and_save_metrics(rewards, actions, episode, name):
    """
    Plot and save graphs for rewards and action distributions.

    :param rewards: List of accumulated rewards for each agent.
    :param actions: List of actions taken for each agent.
    :param episode: Current episode number.
    """

    num_agents = len(rewards[0])

    # Create a figure and a set of subplots for each agent
    fig, axs = plt.subplots(num_agents, 2, figsize=(10, 8 * num_agents))

    for i in range(num_agents):
        # Plot rewards for each agent
        axs[i, 0].plot([r[i] for r in rewards], label=f'Agent {i} Rewards')
        axs[i, 0].set_title(f'Agent {i} Reward Trend')
        axs[i, 0].set_xlabel('Step')
        axs[i, 0].set_ylabel('Reward')
        axs[i, 0].legend()

        # Plot action distribution for each agent
        axs[i, 1].hist([a[i] for a in actions], bins=len(set([a[i] for a in actions])),
                       density=True, alpha=1, color='g')
        axs[i, 1].set_title(f'Agent {i} Action Distribution')
        axs[i, 1].set_xlabel('Action')
        axs[i, 1].set_ylabel('Frequency')

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(
        f'data/png/metrics/{num_agents}_rl_{episode}_metrics_{name}.png')
    plt.close()
