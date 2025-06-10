import numpy as np

data = np.load('merged_demo_data_cleaned.npz', allow_pickle=True)
obs = data['observations']
actions = data['actions']
rewards = data['rewards']
dones = data['dones']
infos = data['infos']

# Horizontal flip
obs_flipped = obs.copy()
actions_flipped = actions.copy()
# Flip agent_x and ball_x (assume obs shape [agent_x, agent_y, ball_x, ball_y, ball_angle])
obs_flipped[:, 0] = 512 - obs[:, 0]  # agent_x
obs_flipped[:, 2] = 512 - obs[:, 2]  # ball_x
# Flip angle if needed (depends on your convention, here we just negate)
obs_flipped[:, 4] = -obs[:, 4]
actions_flipped[:, 0] = 512 - actions[:, 0]  # action_x

# Optionally flip other keys in infos (skip for now)

# Add Gaussian noise (adjust std as needed)
noise_std = 2.0
obs_noisy = obs + np.random.normal(0, noise_std, obs.shape)
actions_noisy = actions + np.random.normal(0, noise_std, actions.shape)

# Concatenate everything
aug_obs = np.concatenate([obs, obs_flipped, obs_noisy], axis=0)
aug_actions = np.concatenate([actions, actions_flipped, actions_noisy], axis=0)
aug_rewards = np.concatenate([rewards, rewards, rewards], axis=0)
aug_dones = np.concatenate([dones, dones, dones], axis=0)
aug_infos = np.concatenate([infos, infos, infos], axis=0)

np.savez_compressed(
    'merged_demo_data_augmented.npz',
    observations=aug_obs,
    actions=aug_actions,
    rewards=aug_rewards,
    dones=aug_dones,
    infos=aug_infos
)
print(f"Augmented data: {aug_obs.shape[0]} samples (original: {obs.shape[0]})")
