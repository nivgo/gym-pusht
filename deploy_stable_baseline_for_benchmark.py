from stable_baselines3 import SAC
import gym_pusht
import gymnasium as gym
import numpy as np
import csv, os
from datetime import datetime
import argparse

# ----------- Argument Parsing -----------
parser = argparse.ArgumentParser()
parser.add_argument('--run-name', type=str, default="", help="Descriptive name for this run (e.g., 'SAC_finetuned')")
args = parser.parse_args()

# ----------- Load SAC agent -----------
model = SAC.load("sac_finetuned_agent")
env = gym.make("gym_pusht/PushT-v0", render_mode=None)

num_episodes = 100
results = []
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
SEED_BASE = 1000

for episode in range(num_episodes):
    seed = SEED_BASE + episode
    obs, info = env.reset(seed=seed)
    total_reward = 0
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
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
