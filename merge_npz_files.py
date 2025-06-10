import numpy as np
import glob

# Find all .npz files in the current directory
npz_files = glob.glob("*.npz")

# Containers for merged data
merged = {}

for idx, fname in enumerate(npz_files):
    data = np.load(fname, allow_pickle=True)
    if idx == 0:
        # Initialize containers
        for k in data.keys():
            merged[k] = [data[k]]
    else:
        for k in data.keys():
            merged[k].append(data[k])

# Concatenate along first dimension (axis=0)
for k in merged:
    merged[k] = np.concatenate(merged[k], axis=0)

# Save the merged dataset
out_name = "merged_demo_data.npz"
np.savez_compressed(out_name, **merged)
print(f"Merged {len(npz_files)} files into {out_name}")
for k in merged:
    print(f"{k}: shape {merged[k].shape}")
