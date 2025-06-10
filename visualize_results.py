import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

csv_file = "benchmark_results.csv"
assert os.path.isfile(csv_file), f"{csv_file} not found!"

# Read results
df = pd.read_csv(csv_file)
print(f"Loaded {len(df)} results.")

# If no 'run_name', add dummy
if 'run_name' not in df.columns:
    df['run_name'] = "default"

# Group by run_name
grouped = df.groupby('run_name')

# Print summary
for name, group in grouped:
    mean_reward = group['total_reward'].mean()
    mean_steps = group['steps'].mean()
    success_rate = group['success'].mean()
    print(f"\nRun: {name}")
    print(f" Episodes: {len(group)}")
    print(f"  Mean Reward: {mean_reward:.2f}")
    print(f"  Mean Steps: {mean_steps:.1f}")
    print(f"  Success Rate: {success_rate*100:.1f}%")

# --------- Plotting ---------
sns.set(style="whitegrid")
palette = sns.color_palette("tab10", n_colors=len(grouped))

# Reward Distribution
plt.figure(figsize=(10,6))
for i, (name, group) in enumerate(grouped):
    sns.histplot(group['total_reward'], label=name, kde=True, stat="density", bins=20, color=palette[i], alpha=0.5)
plt.title("Reward Distribution by Run")
plt.xlabel("Total Reward")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("reward_distribution.png")
plt.show()

# Success Rate Barplot
plt.figure(figsize=(8,4))
sns.barplot(x=[name for name, _ in grouped], y=[g['success'].mean()*100 for _, g in grouped])
plt.title("Success Rate by Run")
plt.ylabel("Success Rate (%)")
plt.xlabel("Run Name")
plt.tight_layout()
plt.savefig("success_rate.png")
plt.show()

# Steps Distribution
plt.figure(figsize=(10,6))
for i, (name, group) in enumerate(grouped):
    sns.histplot(group['steps'], label=name, kde=True, stat="density", bins=20, color=palette[i], alpha=0.5)
plt.title("Steps per Episode by Run")
plt.xlabel("Steps")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig("steps_distribution.png")
plt.show()
