import gymnasium as gym
import gym_pusht
import torch
import numpy as np
import argparse
import time

#How to run??
#python3 visual_compare.py --mode pt --episodes 5
#python3 visual_compare.py --mode sac --episodes 5 --sac-path sac_finetuned_agent.zip

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="pt", choices=['pt', 'sac'],
                    help="Policy mode: 'pt' for PolicyNet, 'sac' for SB3 SAC model")
parser.add_argument('--pt-path', type=str, default="imitation_policy.pt", help="Path to PT model checkpoint")
parser.add_argument('--sac-path', type=str, default="sac_finetuned_agent.zip", help="Path to SAC .zip file")
parser.add_argument('--episodes', type=int, default=3, help="Number of episodes to run")
args = parser.parse_args()

env = gym.make("gym_pusht/PushT-v0", render_mode="human")

if args.mode == "pt":
    # --- PT Model ---
    checkpoint = torch.load(args.pt_path, weights_only=False)
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

    def get_action(obs):
        obs_in = (np.array(obs) - obs_mean) / obs_std
        with torch.no_grad():
            act_out = policy(torch.from_numpy(obs_in).float().unsqueeze(0)).squeeze(0).numpy()
        action = act_out * act_std + act_mean
        action = np.clip(action, 0, 512)
        return action

elif args.mode == "sac":
    # --- SAC Model ---
    from stable_baselines3 import SAC
    model = SAC.load(args.sac_path)

    def get_action(obs):
        # SB3 expects np.array
        action, _ = model.predict(np.array(obs), deterministic=True)
        return np.clip(action, 0, 512)

else:
    raise ValueError("Unknown mode! Use --mode pt or --mode sac")

for ep in range(args.episodes):
    obs, info = env.reset()
    done = False
    ep_reward = 0
    steps = 0

    print(f"=== Episode {ep+1} ===")
    while True:
        action = get_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        ep_reward += reward
        steps += 1
        env.render()
        time.sleep(0.01)  # Remove or adjust to speed up
        if terminated or truncated:
            print(f"Episode {ep+1} finished: total_reward={ep_reward:.2f}, steps={steps}")
            break

print("Done.")
env.close()
