import matplotlib.pyplot as plt
import numpy as np


def plotLearning(x, scores, epsilons, lines=None):
    fig, ax = plt.subplots()

    num_agents = len(scores[0])
    # Add more colors if more than 6 agents
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']

    for i in range(num_agents):
        running_avg = np.empty(len(scores))
        for t in range(len(scores)):
            running_avg[t] = np.mean([score[i]
                                     for score in scores[max(0, t-20):(t+1)]])

        ax.plot(x, running_avg, color=colors[i % len(
            colors)], label=f'Agent {i} Score')

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.legend(loc='upper left')

    ax2 = ax.twinx()
    ax2.plot(x, epsilons, color="C7", label="Epsilon")
    ax2.set_ylabel('Epsilon', color="C7")
    ax2.tick_params(axis='y', colors="C7")
    ax2.legend(loc='upper right')

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    filename = f'data/png/reinforcement/{num_agents}_mapddg.png'
    plt.savefig(filename)
    plt.close()
