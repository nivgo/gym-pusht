import gym_pusht
import gymnasium as gym
from stable_baselines3 import SAC
import torch
import numpy as np

from training_agent import PolicyNet

# 1. Make the environment
env = gym.make("gym_pusht/PushT-v0", render_mode=None)

# 2. Load your imitation policy
imitation_ckpt = torch.load('imitation_policy_augmented.pt', weights_only=False)
policy = PolicyNet(obs_dim=5, act_dim=2)
policy.load_state_dict(imitation_ckpt['policy'])

# 3. Create the RL model
model = SAC('MlpPolicy', env, verbose=1)

# 4. Overwrite the actor network weights with imitation policy weights
# This works for SB3 2.x style policies, see https://stable-baselines3.readthedocs.io/
model.policy.actor.mu.load_state_dict(policy.model.state_dict(), strict=False)

# 5. Train further with RL (fine-tune)
model.learn(total_timesteps=10_000)
model.save("sac_finetuned_agent")

print("RL fine-tuning complete and model saved!")
