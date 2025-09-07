import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import util.loggers as loggers
from model.reinforcement.utils.rl_utilities import next_available_filename

logger = loggers.setup_loggers()
logger = logger['rl']


def plotting(agents, env, fn):
    if not isinstance(agents, list):
        agents = [agents]  # Convert to list for uniform processing

    try:
        # Create a single figure
        plt.figure(figsize=(18, 18))

        for i, agent in enumerate(agents):
            x = range(len(agent.averages))
            y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
            z = agent.averages

            # Subplot for Moving Averages and Regression
            # Adjust subplot position based on the agent index
            plt.subplot(len(agents), 4, i * 4 + 1)
            plt.plot(z, label='Moving averages')
            plt.plot(x, y, 'r--', label='Regression')
            plt.xlabel('Episodes')
            plt.ylabel('Total Reward')
            plt.legend()
            plt.title(f'Agent {i+1} - Moving Averages and Regression')

            # Other subplots can be similarly created for each agent...
            # Strategy, Actions during Learning, Actions during Testing, Feature-Reward Correlations

            # Example for Actions during Learning
            plt.subplot(len(agents), 4, i * 4 + 3)
            actions = np.array(
                agent.train_action_history[:len(agent.env.data)])
            time_steps = np.arange(len(actions))
            for action in np.unique(actions):
                plt.scatter(time_steps[actions == action],
                            actions[actions == action], label=f'Action {action}')
            plt.xlabel('Time step')
            plt.ylabel('Action')
            plt.legend()
            plt.title(f'Agent {i+1} - Actions during Learning')

            # Similarly, create other subplots for each agent

        # Save the figure with multiple subplots
        plot_filename = next_available_filename("combined_plot", "png")
        plt.savefig(plot_filename)

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


def plot_and_save_metrics(nr, rewards, actions, episode, name):
    """
    Plot and save graphs for rewards and action distributions.
    """

    # Determine if it's a single agent or multi-agent scenario
    is_multi_agent = nr > 1

    # Create a figure and a set of subplots based on the number of agents
    fig, axs = plt.subplots(nr, 2, figsize=(10, 8 * max(nr, 1)))

    for i in range(nr):
        # Adjust indexing based on whether it's single or multi-agent
        agent_rewards = rewards[i] if is_multi_agent else rewards
        agent_actions = actions[i] if is_multi_agent else actions

        # Ensure that agent_actions is a list
        if not isinstance(agent_actions, list):
            agent_actions = [agent_actions]

        # Plot rewards
        ax = axs[i, 0] if nr > 1 else axs[0]
        ax.plot(agent_rewards, label=f'Agent {i} Rewards')
        ax.set_title(f'Agent {i} Reward Trend')
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.legend()

        # Plot action distribution
        ax = axs[i, 1] if nr > 1 else axs[1]
        unique_actions = set(agent_actions)
        ax.hist(agent_actions, bins=len(unique_actions),
                density=True, alpha=1, color='g')
        ax.set_title(f'Agent {i} Action Distribution')
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(
        f'data/png/metrics/{name}/{nr}_rl_{episode}_metrics_{name}.png')
    plt.close()
