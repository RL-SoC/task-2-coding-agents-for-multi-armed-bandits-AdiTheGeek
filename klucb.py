import numpy as np
from base import Agent, MultiArmedBandit
from scipy.special import rel_entr
import matplotlib.pyplot as plt


class KLUCBAgent(Agent):
    c : float
    ucb_memory : np.ndarray # An array of the ucb of the different arms
    reward_memory : np.ndarray # A per arm value of how much reward was gathered
    count_memory : np.ndarray # An array of the number of times an arm is pulled 

    def __init__(self, time_horizon, bandit:MultiArmedBandit, c = 3): 
        super().__init__(time_horizon, bandit)
        self.c = c
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
        self.update_klucbs() 
    
    def calc_klucb(self,arm):
        emp_mean = self.reward_memory[arm]/self.count_memory[arm]
        y_at_klucb = ( np.log(self.time_step) + self.c * np.log(np.log(self.time_step)) )/ self.count_memory[arm]
        self.ucb_memory[arm] = self.binary_search(emp_mean,1, y_at_klucb)
    
    def update_klucbs(self):
        for arm in range(len(self.bandit.arms)):
            self.calc_klucb(arm)

    def initiate(self):
        for arm in range(len(self.bandit.arms)):
            reward = self.bandit.pull(arm)
            self.count_memory[arm] += 1
            self.reward_memory[arm] += reward
            self.time_step += 1
            self.update_klucbs()
        self.rewards.append(reward)
        self.bandit.cumulative_regret_array = [0]
        reward = 1 if np.random.random() < self.bandit.arms[arm] else 0
        self.bandit.cumulative_regret_array.append(self.bandit.cumulative_regret_array[-1] + self.bandit.best_arm - reward)

    def kl_divergence(self,p, q):
        if p == 0:
            return 0 if q == 0 else np.inf
        if p == 1:
            return 0 if q == 1 else np.inf
        if q == 0 or q == 1:
            return np.inf
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def binary_search(self,a,b,y):
        while b-a > 10e-6:
            mid = (a+b)/2
            if self.kl_divergence(a, mid) > y:
                b = mid
            else:
                a = mid
        return (a+b)/2
    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        self.reward_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)
 
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
    agent = KLUCBAgent(TIME_HORIZON,bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
