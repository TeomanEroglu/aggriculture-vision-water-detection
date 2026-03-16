import torch

try:
    stats = torch.load("checkpoints/dataset_stats.pt", weights_only=True)
    print("Stats loaded.")
    print("Mean:", stats['mean'])
    print("Std:", stats['std'])
    
    if (stats['std'] == 0).any():
        print("WARNING: Zero found in std!")
    if torch.isnan(stats['mean']).any() or torch.isnan(stats['std']).any():
        print("WARNING: NaN found in stats!")

except Exception as e:
    print(f"Error loading stats: {e}")
