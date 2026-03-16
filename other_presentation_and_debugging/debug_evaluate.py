import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from PIL import Image

from src.data_loader import AgricultureVisionMultiLabel, BasicTransform, CLASS_NAMES
from src.model import AgrarUNetPlusPlus 

def compute_iou_debug(model, loader, device):
    model.eval()
    num_classes = len(CLASS_NAMES)
    inter = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    print(f"Processing {len(loader)} batches for IoU...")
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i >= 5: break # Only check first 5 batches
            
            imgs, masks, valid = batch["image"].to(device), batch["labels"].to(device), batch["valid_mask"].to(device)
            
            if torch.isnan(imgs).any() or torch.isinf(imgs).any():
                print(f"BATCH {i}: WARN: Input images contain NaN or Inf!")

            # with torch.amp.autocast('cuda'):
            #     outputs = model(imgs)
            # Try without autocast first
            outputs = model(imgs)
            
            # DEBUG PRINTS
            print(f"-- Batch {i} --")
            print(f"Output stats: Min {outputs.min().item():.4f}, Max {outputs.max().item():.4f}, Mean {outputs.mean().item():.4f}")
            
            preds = (torch.sigmoid(outputs) > 0.5)
            preds_np = preds.cpu().numpy()
            print(f"Preds sum: {preds.sum().item()} / {preds.numel()}")
            print(f"Masks sum: {masks.sum().item()} / {masks.numel()}")
            
            masks_np, valid_np = masks.cpu().numpy(), valid.cpu().numpy()

            for c in range(num_classes):
                p_c = preds_np[:, c, :, :] & valid_np
                t_c = (masks_np[:, c, :, :] > 0.5) & valid_np
                inter_c = np.logical_and(p_c, t_c).sum()
                union_c = np.logical_or(p_c, t_c).sum()
                print(f"Class {c}: Inter {inter_c}, Union {union_c}")
                
                inter[c] += inter_c
                union[c] += union_c

    ious = {name: inter[i] / (union[i] + 1e-6) for i, name in enumerate(CLASS_NAMES)}
    return ious

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")
    CHECKPOINT = "checkpoints/best_hybrid_model.pth"

    full_val_dataset = AgricultureVisionMultiLabel("data", "val", transform=BasicTransform())
    val_samples = full_val_dataset.samples
    random.seed(42) 
    random.shuffle(val_samples)
    split_idx = len(val_samples) // 2

    # Use the HOLD-OUT set (Second half)
    test_dataset = AgricultureVisionMultiLabel("data", "val", transform=BasicTransform())
    test_dataset.samples = val_samples[split_idx:]
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0) # num_workers=0 for debug

    model = AgrarUNetPlusPlus(num_classes=len(CLASS_NAMES)).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        try:
            model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
            print(f"Best model loaded from {CHECKPOINT}.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}")
            return
    else:
        print("Checkpoint not found!"); return

    ious = compute_iou_debug(model, test_loader, DEVICE)
    print("\nDEBUG TEST RESULTS:")
    for name, score in ious.items():
        print(f"{name:12}: {score:.4f}")

if __name__ == "__main__":
    main()
