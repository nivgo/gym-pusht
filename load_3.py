import numpy as np
import matplotlib.pyplot as plt

data = np.load('mouse_demo_1749333380.npz', allow_pickle=True)
obs = data['observations']
rewards = data['rewards']

ball_x = obs[:, 2]
ball_y = obs[:, 3]
paddle_x = obs[:, 0]
paddle_y = obs[:, 1]

fig, axs = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [3, 1]})

# Trajectories
axs[0].plot(paddle_x, paddle_y, label='Paddle', color='blue', alpha=0.6)
sc = axs[0].scatter(ball_x, ball_y, c=rewards, cmap='plasma', label='Ball (reward)', s=30)
axs[0].legend()
axs[0].set_xlim([0, 512])
axs[0].set_ylim([0, 512])
axs[0].invert_yaxis()
axs[0].set_title('Demo Trajectories with Reward Coloring')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
plt.colorbar(sc, ax=axs[0], label='Reward')

# Reward over time
axs[1].plot(rewards, label='Reward per timestep', color='green')
axs[1].set_title('Reward Over Time')
axs[1].set_xlabel('Timestep')
axs[1].set_ylabel('Reward')
axs[1].legend()

plt.tight_layout()
plt.show()
