import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class UCBAgent(Agent):
    ucb_memory : np.ndarray # An array of the ucb of the different arms
    reward_memory : np.ndarray # A per arm value of how much reward was gathered
    count_memory : np.ndarray # An array of the number of times an arm is pulled 

    def __init__(self, time_horizon, bandit:MultiArmedBandit): 
        super().__init__(time_horizon, bandit)
        self.ucb_memory = np.zeros(len(bandit.arms))
        self.bandit : MultiArmedBandit = bandit
        self.reward_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.time_step = 0

    def give_pull(self):
        if self.time_step == 0:
            self.initiate()
        else:
            chosen_arm = np.argmax(self.ucb_memory)
            reward = self.bandit.pull(chosen_arm)
            self.reinforce(reward,chosen_arm)
        self.update_ucbs()

    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)
    
    def initiate(self):
        for arm in range(len(self.bandit.arms)):
            reward = self.bandit.pull(arm)
            self.count_memory[arm] += 1
            self.reward_memory[arm] += reward
            self.time_step += 1
            self.update_ucbs()
        self.rewards.append(reward)
        self.bandit.cumulative_regret_array = [0]
        reward = 1 if np.random.random() < self.bandit.arms[arm] else 0
        self.bandit.cumulative_regret_array.append(self.bandit.cumulative_regret_array[-1] + self.bandit.best_arm - reward)

    def calc_ucb(self,arm):
        self.ucb_memory[arm] = (self.reward_memory[arm]/
                                self.count_memory[arm]) + np.sqrt((2*np.log(self.time_step))/
                                                                  self.count_memory[arm])
    
    def update_ucbs(self):
        for arm in range(len(self.bandit.arms)):
            self.calc_ucb(arm)
    
    def plot_arm_graph(self):
        counts = self.count_memory
        indices = np.arange(len(counts))

        # Plot the data
        plt.figure(figsize=(12, 6))
        plt.bar(indices, counts, color='skyblue', edgecolor='black')

        # Formatting
        plt.title('Counts per Category', fontsize=16)
        plt.xlabel('Arm', fontsize=14)
        plt.ylabel('Pull Count', fontsize=14)
        plt.grid(axis='y', linestyle='-')  # Add grid lines for the y-axis
        plt.xticks(indices, [f'Category {i+1}' for i in indices], rotation=45, ha='right')
        # plt.yticks(np.arange(0, max(counts) + 2, step=2))

        # Annotate the bars with the count values
        for i, count in enumerate(counts):
            plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, color='black')

        # Tight layout to ensure there's no clipping of labels
        plt.tight_layout()

        # Show plot
        plt.show()


# Code to test
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 10_000
    bandit = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent = UCBAgent(TIME_HORIZON,bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
