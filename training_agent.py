import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load your data
data = np.load('merged_demo_data_augmented.npz', allow_pickle=True)
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
            nn.Linear(obs_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, act_dim)
        )
    def forward(self, x):
        return self.model(x)

policy = PolicyNet(obs_dim=5, act_dim=2)
optimizer = optim.Adam(policy.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

# Training loop with best checkpoint tracking
best_val_loss = float('inf')
best_epoch = -1
best_state = None

for epoch in range(2000):
    policy.train()
    optimizer.zero_grad()
    pred = policy(train_obs)
    loss = loss_fn(pred, train_act)
    loss.backward()
    optimizer.step()

    # Validation loss
    policy.eval()
    with torch.no_grad():
        val_pred = policy(val_obs)
        val_loss = loss_fn(val_pred, val_act).item()
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        # Save the best model weights
        best_state = policy.state_dict()

    if (epoch+1) % 10 == 0 or epoch == 0:
        print(f"Epoch {epoch+1}: Train Loss {loss.item():.4f} | Val Loss {val_loss:.4f}")

# Save the best policy and normalization stats for deployment
torch.save(
    {
        'policy': best_state,
        'obs_mean': obs_mean,
        'obs_std': obs_std,
        'act_mean': act_mean,
        'act_std': act_std,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
    },
    'imitation_policy_augmented.pt'
)
print(f"\nBest validation loss: {best_val_loss:.4f} at epoch {best_epoch}")
