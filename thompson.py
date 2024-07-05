import numpy as np
from base import Agent, MultiArmedBandit
import matplotlib.pyplot as plt


class ThompsonSamplingAgent(Agent):
    success_memory : np.ndarray # A per arm value of how many successes it got
    failure_memory : np.ndarray # A per arm value of how many failures it got
    sample_memory : np.ndarray # An array of samples drawn for each arm for their unique Beta distribution
    count_memory : np.ndarray # An array of the number of times an arm is pulled 

    def __init__(self, time_horizon, bandit:MultiArmedBandit): 
        super().__init__(time_horizon, bandit)
        self.bandit : MultiArmedBandit = bandit
        self.success_memory = np.zeros(len(bandit.arms))
        self.failure_memory = np.zeros(len(bandit.arms))
        self.sample_memory = np.zeros(len(bandit.arms))
        self.count_memory = np.zeros(len(bandit.arms))
        self.time_step = 0

    def give_pull(self):
        self.draw_samples()
        chosen_arm = np.argmax(self.sample_memory)
        reward = self.bandit.pull(chosen_arm)
        self.reinforce(reward,chosen_arm)


    def reinforce(self, reward, arm):
        self.count_memory[arm] += 1
        if reward == 1:
            self.success_memory[arm] += reward
        else:
            self.failure_memory[arm] += reward
        self.time_step += 1
        self.rewards.append(reward)

    def draw_samples (self):
        for arm in range(len(self.bandit.arms)):
            self.sample_memory[arm] = np.random.beta(self.success_memory[arm]+1,self.failure_memory[arm]+1)
 
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
    agent = ThompsonSamplingAgent(TIME_HORIZON,bandit) ## Fill with correct constructor

    # Loop
    for i in range(TIME_HORIZON):
        agent.give_pull()

    # Plot curves
    agent.plot_reward_vs_time_curve()
    agent.plot_arm_graph()
    bandit.plot_cumulative_regret()
