import gymnasium as gym
import gym_pusht
import pygame
import numpy as np
import time

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
obs, info = env.reset()

recorded_obs = []
recorded_actions = []
recorded_rewards = []
recorded_dones = []
recorded_infos = []

print("Move your mouse in the simulation window to control the paddle.")
print("Close the window or press ESC to stop and save the demo.")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False

    mouse_pos = pygame.mouse.get_pos()  # (x, y) in window coordinates

    # Make sure coordinates are within the allowed action space [0, 512]
    x = float(np.clip(mouse_pos[0], 0, 512))
    y = float(np.clip(mouse_pos[1], 0, 512))
    action = [x, y]

    # Record data
    recorded_obs.append(np.copy(obs))
    recorded_actions.append(np.copy(action))

    obs, reward, terminated, truncated, info = env.step(action)
    recorded_rewards.append(reward)
    recorded_dones.append(terminated or truncated)
    recorded_infos.append(info)

    env.render()
    time.sleep(0.01)

    if terminated or truncated:
        obs, info = env.reset()

env.close()

# Save all recorded data to a file
filename = f"mouse_demo_{int(time.time())}.npz"
np.savez_compressed(
    filename,
    observations=np.array(recorded_obs),
    actions=np.array(recorded_actions),
    rewards=np.array(recorded_rewards),
    dones=np.array(recorded_dones),
    infos=np.array(recorded_infos, dtype=object)
)
print(f"Demo saved to {filename}")
