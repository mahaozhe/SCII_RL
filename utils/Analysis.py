"""
To analyze and visualize the training results.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

model_name = "MoveToBeacon-DDPG-SG2"
tokens = ["final", "best", "2000"]
main_path = "./../Saves/"

colors = ['blue', 'red', 'green', 'cyan', 'magenta']
labels = ["actor", "critic"]

save_path = os.path.join(main_path, model_name, tokens[0])

# * for the epoch rewards
epoch_rewards = np.load(os.path.join(save_path, "epoch_rewards.npy")).tolist()
epochs = list(range(1, len(epoch_rewards) + 1))

plt.plot(epochs, epoch_rewards)
plt.title("Rewards per Epoch")
plt.show()

# * for the cumulative rewards
cumulative_rewards = []
cum_reward = 0

for reward in epoch_rewards:
    cum_reward += reward
    cumulative_rewards.append(cum_reward)

plt.plot(epochs, cumulative_rewards)
plt.title("Cumulative Rewards over Training")
plt.show()

# * for the epoch steps
epoch_steps = np.load(os.path.join(save_path, "epoch_steps.npy")).tolist()

plt.plot(epochs, epoch_steps)
plt.title("Steps per Epoch")
plt.show()
