import numpy as np

data = np.load('merged_demo_data.npz', allow_pickle=True)
observations = data['observations']
actions = data['actions']
rewards = data['rewards']
dones = data['dones']
infos = data['infos']

# Indices where episodes end
done_indices = np.where(dones)[0]

# Start indices (0 + after each done)
start_indices = np.concatenate([[0], done_indices[:-1]+1])

# Containers for cleaned data
obs_clean, act_clean, rew_clean, dones_clean, infos_clean = [], [], [], [], []

for start, end in zip(start_indices, done_indices):
    episode_rewards = rewards[start:end+1]
    # You said: failure if reward at end is < 0
    if episode_rewards[-1] >= 0:
        obs_clean.append(observations[start:end+1])
        act_clean.append(actions[start:end+1])
        rew_clean.append(rewards[start:end+1])
        dones_clean.append(dones[start:end+1])
        infos_clean.append(infos[start:end+1])

# Concatenate cleaned episodes
observations_clean = np.concatenate(obs_clean, axis=0)
actions_clean = np.concatenate(act_clean, axis=0)
rewards_clean = np.concatenate(rew_clean, axis=0)
dones_clean = np.concatenate(dones_clean, axis=0)
infos_clean = np.concatenate(infos_clean, axis=0)

# Save cleaned data
np.savez_compressed(
    'merged_demo_data_cleaned.npz',
    observations=observations_clean,
    actions=actions_clean,
    rewards=rewards_clean,
    dones=dones_clean,
    infos=infos_clean
)
print(f"Saved cleaned file with {len(observations_clean)} records (from {len(observations)})")
