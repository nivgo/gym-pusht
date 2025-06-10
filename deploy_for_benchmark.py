import gymnasium as gym
import gym_pusht
import torch
import numpy as np
import csv
import os
from datetime import datetime
import argparse

# ----------- Argument Parsing -----------
parser = argparse.ArgumentParser()
parser.add_argument('--run-name', type=str, default="", help="Descriptive name for this run (e.g., 'IL_policy_v2_aug')")
args = parser.parse_args()

# ----------- Policy Loading -----------
checkpoint = torch.load("imitation_policy_augmented.pt", weights_only=False)
obs_mean = checkpoint['obs_mean']
obs_std = checkpoint['obs_std']
act_mean = checkpoint['act_mean']
act_std = checkpoint['act_std']

class PolicyNet(torch.nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(obs_dim, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, 128), torch.nn.ReLU(),
            torch.nn.Linear(128, act_dim)
        )
    def forward(self, x):
        return self.model(x)

policy = PolicyNet(5, 2)
policy.load_state_dict(checkpoint['policy'])
policy.eval()

env = gym.make("gym_pusht/PushT-v0", render_mode=None)

num_episodes = 100
results = []
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SEED_BASE = 1000   # Use a fixed base

for episode in range(num_episodes):
    seed = SEED_BASE + episode
    obs, info = env.reset(seed=seed)
    total_reward = 0
    steps = 0

    while True:
        obs_in = (np.array(obs) - obs_mean) / obs_std
        with torch.no_grad():
            act_out = policy(torch.from_numpy(obs_in).float().unsqueeze(0)).squeeze(0).numpy()
        action = act_out * act_std + act_mean
        action = np.clip(action, 0, 512)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            break

    success = info.get("is_success", False)
    results.append({
        "timestamp": timestamp,
        "run_name": args.run_name,
        "episode": episode+1,
        "seed": seed,
        "total_reward": total_reward,
        "steps": steps,
        "success": int(success)
    })

# Write to CSV (append if exists, else create)
csv_file = "benchmark_results.csv"
write_header = not os.path.isfile(csv_file)
with open(csv_file, "a", newline="") as f:
    writer = csv.DictWriter(
        f, 
        fieldnames=["timestamp", "run_name", "episode", "seed", "total_reward", "steps", "success"]
    )
    if write_header:
        writer.writeheader()
    writer.writerows(results)

print(f"Done! Results for {num_episodes} episodes appended to {csv_file}")
