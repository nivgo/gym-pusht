import gymnasium as gym
import gym_pusht
import torch
import numpy as np

# Load policy and stats
checkpoint = torch.load('imitation_policy.pt')
obs_mean = checkpoint['obs_mean']
obs_std = checkpoint['obs_std']
act_mean = checkpoint['act_mean']
act_std = checkpoint['act_std']

class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, 64), torch.nn.ReLU(),
            torch.nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.model(x)

policy = PolicyNet(5, 2)
policy.load_state_dict(checkpoint['policy'])
policy.eval()

env = gym.make("gym_pusht/PushT-v0", render_mode="human")
obs, info = env.reset()
done = False

while True:
    # Normalize observation
    obs_in = (np.array(obs) - obs_mean) / obs_std
    with torch.no_grad():
        act_out = policy(torch.from_numpy(obs_in).float().unsqueeze(0)).squeeze(0).numpy()
    # De-normalize action
    action = act_out * act_std + act_mean
    action = np.clip(action, 0, 512)

    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        obs, info = env.reset()
