import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

data = np.load('mouse_demo_1749333380.npz', allow_pickle=True)
obs = data['observations']
rewards = data['rewards']

paddle_x = obs[:, 0]
paddle_y = obs[:, 1]
ball_x = obs[:, 2]
ball_y = obs[:, 3]

cum_rewards = np.cumsum(rewards)

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_xlim(0, 512)
ax.set_ylim(0, 512)
ax.invert_yaxis()
paddle_dot, = ax.plot([], [], 'bo', ms=10, label='Paddle')
ball_dot, = ax.plot([], [], 'ro', ms=10, label='Ball')
text = ax.text(10, 20, '', fontsize=12, color='darkgreen')
ax.legend()

def init():
    paddle_dot.set_data([], [])
    ball_dot.set_data([], [])
    text.set_text('')
    return paddle_dot, ball_dot, text

def update(frame):
    paddle_dot.set_data(paddle_x[frame], paddle_y[frame])
    ball_dot.set_data(ball_x[frame], ball_y[frame])
    text.set_text(f"Cumulative Reward: {cum_rewards[frame]:.1f}  |  Instant Reward: {rewards[frame]:.2f}")
    return paddle_dot, ball_dot, text

ani = FuncAnimation(fig, update, frames=len(paddle_x), init_func=init, blit=True, interval=20)
plt.show()
