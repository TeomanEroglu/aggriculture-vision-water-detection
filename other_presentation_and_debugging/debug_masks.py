import os
import torch
import numpy as np
from src.data_loader import AgricultureVisionMultiLabel, BasicTransform

def check_one_batch():
    print("Checking 'val' dataset masks...")
    ds = AgricultureVisionMultiLabel("data", split="val", transform=BasicTransform(), debug=True)
    
    if len(ds) == 0:
        print("Dataset 'val' is empty!")
        return

    # Check first 5 samples
    for i in range(min(5, len(ds))):
        sample = ds[i]
        mask = sample["valid_mask"] # Tensor
        print(f"Sample {i} ({sample['id']}): Valid Mask True Pixels: {mask.sum().item()} / {mask.numel()} ({mask.sum().item()/mask.numel()*100:.2f}%)")
        
        labels = sample["labels"]
        print(f"Sample {i}: Labels Sum: {labels.sum().item()}")

if __name__ == "__main__":
    check_one_batch()
