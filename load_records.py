import numpy as np
import matplotlib.pyplot as plt

data = np.load('mouse_demo_1749333380.npz', allow_pickle=True)
obs = data['observations']
rewards = data['rewards']
acts = data['actions']

paddle_x = obs[:, 0]
paddle_y = obs[:, 1]
ball_x = obs[:, 2]
ball_y = obs[:, 3]

plt.figure(figsize=(8, 6))
plt.plot(paddle_x, paddle_y, label='Paddle', color='blue', alpha=0.6)

# Plot ball trajectory with color mapped to reward
sc = plt.scatter(ball_x, ball_y, c=rewards, cmap='plasma', label='Ball (reward)', s=30)
plt.colorbar(sc, label='Reward')

plt.xlim([0, 512])
plt.ylim([0, 512])
plt.gca().invert_yaxis()
plt.title('Demo Trajectories with Reward Coloring')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
