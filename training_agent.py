import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load your data
data = np.load('mouse_demo_1749333380.npz', allow_pickle=True)
observations = data['observations'].astype(np.float32)  # (N, 5)
actions = data['actions'].astype(np.float32)            # (N, 2)

# Normalize observations and actions for better training (optional, but helps!)
obs_mean, obs_std = observations.mean(axis=0), observations.std(axis=0) + 1e-6
act_mean, act_std = actions.mean(axis=0), actions.std(axis=0) + 1e-6

obs_norm = (observations - obs_mean) / obs_std
act_norm = (actions - act_mean) / act_std

# Convert to torch tensors
obs_tensor = torch.from_numpy(obs_norm)
act_tensor = torch.from_numpy(act_norm)

# Split into train/validation
N = len(obs_tensor)
train_idx = int(0.9 * N)
train_obs, train_act = obs_tensor[:train_idx], act_tensor[:train_idx]
val_obs, val_act = obs_tensor[train_idx:], act_tensor[train_idx:]

# Define policy network
class PolicyNet(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, act_dim)
        )
    def forward(self, x):
        return self.model(x)

policy = PolicyNet(obs_dim=5, act_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop
for epoch in range(100):
    policy.train()
    optimizer.zero_grad()
    pred = policy(train_obs)
    loss = loss_fn(pred, train_act)
    loss.backward()
    optimizer.step()
    
    # Validation loss
    if (epoch+1) % 10 == 0 or epoch == 0:
        policy.eval()
        with torch.no_grad():
            val_pred = policy(val_obs)
            val_loss = loss_fn(val_pred, val_act).item()
        print(f"Epoch {epoch+1}: Train Loss {loss.item():.4f} | Val Loss {val_loss:.4f}")

# Save policy and normalization stats for deployment
torch.save({'policy': policy.state_dict(), 'obs_mean': obs_mean, 'obs_std': obs_std, 'act_mean': act_mean, 'act_std': act_std}, 'imitation_policy.pt')
