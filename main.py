import numpy as np
from base import Agent, MultiArmedBandit
from klucb import KLUCBAgent
from ucb import UCBAgent
from thompson import ThompsonSamplingAgent
from epsilon_greedy import EpsilonGreedyAgent
import matplotlib.pyplot as plt

# S1
if __name__ == "__main__":
    # Init Bandit
    TIME_HORIZON = 30_000
    bandit_1 = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    bandit_2 = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    bandit_3 = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    bandit_4 = MultiArmedBandit(np.array([0.23,0.55,0.76,0.44]))
    agent_1 = EpsilonGreedyAgent(TIME_HORIZON,bandit_1)
    agent_2 = UCBAgent(TIME_HORIZON,bandit_2)
    agent_3 = KLUCBAgent(TIME_HORIZON,bandit_3)
    agent_4 = ThompsonSamplingAgent(TIME_HORIZON,bandit_4)

    # Loop
    for i in range(TIME_HORIZON):
        agent_1.give_pull()
    
    for i in range(TIME_HORIZON):
        agent_2.give_pull()

    for i in range(TIME_HORIZON):
        agent_3.give_pull()

    for i in range(TIME_HORIZON):
        agent_4.give_pull()

    #Plotting cummulative rewards for all the agents together
        # Create an index for timesteps
    timesteps = np.arange(1, len(agent_1.rewards) + 1)

    # Average out rewards for each agent
    avg_rewards_1 = np.cumsum(agent_1.rewards) / timesteps
    avg_rewards_2 = np.cumsum(agent_2.rewards) / timesteps
    avg_rewards_3 = np.cumsum(agent_3.rewards) / timesteps
    avg_rewards_4 = np.cumsum(agent_4.rewards) / timesteps

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(timesteps, avg_rewards_1, linestyle='-', color='g', label='Epsilon Greedy')
    plt.plot(timesteps, avg_rewards_2, linestyle='-', color='b', label='UCB')
    plt.plot(timesteps, avg_rewards_3, linestyle='-', color='r', label='KLUCB')
    plt.plot(timesteps, avg_rewards_4, linestyle='-', color='m', label='Thompson sampling')

    # Formatting
    plt.title('Average Reward Over Time', fontsize=16)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Mean Reward Value up to Timestep t', fontsize=14)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)

    # Add legend
    plt.legend(loc='upper left', fontsize=12)

    # Tight layout to ensure there's no clipping of labels
    plt.tight_layout()

    # Show plot
    plt.show()

    # Plotting cummulative regret for all the agents together
    timesteps = np.arange(1, len(bandit_1.cumulative_regret_array) + 1)

    # Plot the data
    plt.figure(figsize=(8,4))
    plt.plot(timesteps, bandit_1.cumulative_regret_array, linestyle='-', color='g', label='Epsilon Greedy')
    plt.plot(timesteps, bandit_2.cumulative_regret_array, linestyle='-', color='r', label='UCB')
    plt.plot(timesteps, bandit_3.cumulative_regret_array, linestyle='-', color='b', label='KLUCB')
    plt.plot(timesteps, bandit_4.cumulative_regret_array, linestyle='-', color='m', label='Thompson Sampling')

    # Formatting
    plt.title('Cumulative Regret Over Time', fontsize=16)
    plt.xlabel('Timesteps', fontsize=14)
    plt.ylabel('Cumulative Regret', fontsize=14)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.yticks(np.arange(0, max(bandit_1.cumulative_regret_array) + 5, step=5))

    # Add legend
    plt.legend(loc='upper left', fontsize=12)

    # Tight layout to ensure there's no clipping of labels
    plt.tight_layout()

    # Show plot
    plt.show()

    #plotting the bar graph
    opt_arm_1 = np.argmax(agent_1.count_memory)
    opt_arm_2 = np.argmax(agent_2.count_memory)
    opt_arm_3 = np.argmax(agent_3.count_memory)
    opt_arm_4 = np.argmax(agent_4.count_memory)
    indices = range(1,5)

    opt_arm_counts = np.array([agent_1.count_memory[opt_arm_1],
                               agent_2.count_memory[opt_arm_2],
                               agent_3.count_memory[opt_arm_3],
                               agent_4.count_memory[opt_arm_4],])
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.bar(indices, opt_arm_counts, color='skyblue', edgecolor='black')

    # Formatting
    plt.title('Counts per Category', fontsize=16)
    plt.xlabel('Opt_Arm', fontsize=14)
    plt.ylabel('Pull Count', fontsize=14)
    plt.grid(axis='y', linestyle='-')  # Add grid lines for the y-axis
    plt.xticks(indices, [f'Agent {i}' for i in indices], rotation=45, ha='right')
    # plt.yticks(np.arange(0, max(counts) + 2, step=2))

    # Annotate the bars with the count values
    for i, count in enumerate(opt_arm_counts):
        plt.text(i, count + 0.5, str(count), ha='center', va='bottom', fontsize=12, color='black')

    # Tight layout to ensure there's no clipping of labels
    plt.tight_layout()

    # Show plot
    plt.show()


    #S2
    Final_regret_bandit_1_agent_1 = np.zeros(17)
    Final_regret_bandit_1_agent_2 = np.zeros(17)
    Final_regret_bandit_1_agent_3 = np.zeros(17)
    Final_regret_bandit_1_agent_4 = np.zeros(17)

    Final_regret_bandit_2_agent_1 = np.zeros(17)
    Final_regret_bandit_2_agent_2 = np.zeros(17)
    Final_regret_bandit_2_agent_3 = np.zeros(17)
    Final_regret_bandit_2_agent_4 = np.zeros(17)

    for j in range(17):
        TIME_HORIZON = 30_000
        bandit_1 = MultiArmedBandit(np.array([0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]))
        bandit_2 = MultiArmedBandit(np.array([0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]))
        
        agent_1 = EpsilonGreedyAgent(TIME_HORIZON,bandit_1)
        for i in range(TIME_HORIZON):
            agent_1.give_pull()
        Final_regret_bandit_1_agent_1[j] = bandit_1.cumulative_regret_array[-1]
        print(f"A1_B1:{j}")

        agent_2 = UCBAgent(TIME_HORIZON,bandit_1)
        for i in range(TIME_HORIZON):
            agent_2.give_pull()
        Final_regret_bandit_1_agent_2[j] = bandit_1.cumulative_regret_array[-1]
        print(f"A2_B1:{j}")

        agent_3 = KLUCBAgent(TIME_HORIZON,bandit_1)
        for i in range(TIME_HORIZON):
            agent_3.give_pull()
        Final_regret_bandit_1_agent_3[j] = bandit_1.cumulative_regret_array[-1]
        print(f"A3_B1:{j}")

        agent_4 = ThompsonSamplingAgent(TIME_HORIZON,bandit_1)
        for i in range(TIME_HORIZON):
            agent_4.give_pull()
        Final_regret_bandit_1_agent_4[j] = bandit_1.cumulative_regret_array[-1]
        print(f"A4_B1:{j}")

        agent_1 = EpsilonGreedyAgent(TIME_HORIZON,bandit_2)
        for i in range(TIME_HORIZON):
            agent_1.give_pull()
        Final_regret_bandit_2_agent_1[j] = bandit_2.cumulative_regret_array[-1]
        print(f"A1_B2:{j}")

        agent_2 = UCBAgent(TIME_HORIZON,bandit_2)
        for i in range(TIME_HORIZON):
            agent_2.give_pull()
        Final_regret_bandit_2_agent_2[j] = bandit_2.cumulative_regret_array[-1]
        print(f"A2_B2:{j}")

        agent_3 = KLUCBAgent(TIME_HORIZON,bandit_2)
        for i in range(TIME_HORIZON):
            agent_3.give_pull()
        Final_regret_bandit_2_agent_3[j] = bandit_2.cumulative_regret_array[-1]
        print(f"A3_B2:{j}")

        agent_4 = ThompsonSamplingAgent(TIME_HORIZON,bandit_2)
        for i in range(TIME_HORIZON):
            agent_4.give_pull()
        Final_regret_bandit_2_agent_4[j] = bandit_2.cumulative_regret_array[-1]
        print(f"A4_B2:{j}")

    indices = np.arange(len(Final_regret_bandit_2_agent_4))
    # Plot the data
    plt.figure(figsize=(8,4))
    plt.plot(indices, Final_regret_bandit_1_agent_1, linestyle='-', color='g', label='Epsilon_Greedy')
    plt.plot(indices, Final_regret_bandit_1_agent_2, linestyle='-', color='r', label='UCB')
    plt.plot(indices, Final_regret_bandit_1_agent_3, linestyle='-', color='b', label='KLUCB')
    plt.plot(indices, Final_regret_bandit_1_agent_4, linestyle='-', color='m', label='Thompson_sampling')

    # Formatting
    plt.title('Final regret for bandit_1', fontsize=16)
    plt.xlabel('Game index', fontsize=14)
    plt.ylabel('Final_regret', fontsize=14)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.xticks(indices,[f'Game{i+1}' for i in indices],rotation=45, ha='right')

    # Add legend
    plt.legend(loc='upper left', fontsize=12)

    # Tight layout to ensure there's no clipping of labels
    plt.tight_layout()

    # Show plot
    plt.show()

    # Bandit 2
    # Plot the data
    plt.figure(figsize=(8,4))
    plt.plot(indices, Final_regret_bandit_2_agent_1, linestyle='-', color='g', label='Epsilon_Greedy')
    plt.plot(indices, Final_regret_bandit_2_agent_2, linestyle='-', color='r', label='UCB')
    plt.plot(indices, Final_regret_bandit_2_agent_3, linestyle='-', color='b', label='KLUCB')
    plt.plot(indices, Final_regret_bandit_2_agent_4, linestyle='-', color='m', label='Thompson_sampling')

    # Formatting
    plt.title('Final regret for bandit_2', fontsize=16)
    plt.xlabel('Game index', fontsize=14)
    plt.ylabel('Final_regret', fontsize=14)
    plt.grid(True, which='both', linestyle='-', linewidth=0.5)
    plt.xticks(indices,[f'Game{i+1}' for i in indices],rotation=45, ha='right')

    # Add legend
    plt.legend(loc='upper left', fontsize=12)

    # Tight layout to ensure there's no clipping of labels
    plt.tight_layout()

    # Show plot
    plt.show()
