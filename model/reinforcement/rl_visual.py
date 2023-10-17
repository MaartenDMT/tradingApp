import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from util.rl_util import next_available_filename


def plotting(agent, env, fn):
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
    print(env.data['action'])

    # Subplot 2: Strategy
    plt.subplot(4, 2, 2)  # 4 rows, 2 columns, 2nd plot
    agent.env.data_['strategy'].cumsum().plot(label='Strategy')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.title('Strategy Over Time')

    # Subplot 3: Actions during Learning
    plt.subplot(4, 2, 3)  # 4 rows, 2 columns, 3rd plot
    actions = np.array(agent.train_action_history[:len(agent.env.data)])
    print(f"Actions shape: {actions.shape}")
    print(f"Time steps: {len(agent.env.data)}")
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
    print(f'full action: {len(actions)}')
    print(f'full time steps: {len(time_steps)}')
    for action in np.unique(actions):
        print(f'action: {action}')
        plt.scatter(time_steps[actions == action],
                    actions[actions == action], label=f'Action {action}')
    plt.xlabel('Time step')
    plt.ylabel('Action')
    plt.legend()
    plt.title('Actions during Testing')

    # Subplot 5: Feature-Reward Correlations
    ax5 = plt.subplot(4, 2, 5)  # 4 rows, 2 columns, 5th plot
    agent.calculate_correlations(fn, ax5)
    # Note: This assumes calculate_correlations plots within the same active figure

    # Save the figure with multiple subplots
    plot_filename = next_available_filename("combined_plot", "png")
    plt.savefig(plot_filename)

    # Example usage for saving DataFrame
    csv_filename = next_available_filename("dataframe", "csv")
    data_df = pd.DataFrame(agent.env.data)
    data_df.to_csv(csv_filename)

    # Example usage for saving agent
    agent_filename = next_available_filename("agent", "pickle/pkl")
    with open(agent_filename, 'wb') as f:
        pickle.dump(agent, f)

    # Show the entire figure with subplots
    # plt.show()
