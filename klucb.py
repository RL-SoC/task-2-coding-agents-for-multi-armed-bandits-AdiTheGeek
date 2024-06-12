import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class KLUCBAgent(Agent):
    # Add fields 

    def __init__(self, time_horizon, bandit:MultiArmedBandit,): 
        # Add fields
        super().__init__(time_horizon, bandit)

    def give_pull(self):
        raise NotImplementedError 

    def reinforce(self, reward, arm):
        raise NotImplementedError
 
    def plot_arm_graph(self):
        raise NotImplementedError


# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = None ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
