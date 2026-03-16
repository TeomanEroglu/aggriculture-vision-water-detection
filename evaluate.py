import os
import numpy as np
import torch
import random
from torch.utils.data import DataLoader
from PIL import Image

from src.data_loader import AgricultureVisionMultiLabel, BasicTransform, CLASS_NAMES
from src.model import AgrarUNetPlusPlus 

def compute_metrics(model, loader, device):
    model.eval()
    num_classes = len(CLASS_NAMES)
    tp = np.zeros(num_classes, dtype=np.float64)
    fp = np.zeros(num_classes, dtype=np.float64)
    fn = np.zeros(num_classes, dtype=np.float64)
    tn = np.zeros(num_classes, dtype=np.float64)

    print(f"Processing {len(loader)} batches for metrics...")
    with torch.no_grad():
        for batch in loader:
            imgs, masks, valid = batch["image"].to(device), batch["labels"].to(device), batch["valid_mask"].to(device)
            outputs = model(imgs)
            
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            masks_np, valid_np = masks.cpu().numpy(), valid.cpu().numpy()

            for c in range(num_classes):
                p_c = preds[:, c, :, :] & valid_np
                t_c = (masks_np[:, c, :, :] > 0.5) & valid_np
                
                tp[c] += np.logical_and(p_c, t_c).sum()
                fp[c] += np.logical_and(p_c, np.logical_not(t_c)).sum()
                fn[c] += np.logical_and(np.logical_not(p_c), t_c).sum()
                tn[c] += np.logical_and(np.logical_not(p_c), np.logical_not(t_c)).sum()

    results = {}
    for i, name in enumerate(CLASS_NAMES):
        iou = tp[i] / (tp[i] + fp[i] + fn[i] + 1e-6)
        precision = tp[i] / (tp[i] + fp[i] + 1e-6)
        recall = tp[i] / (tp[i] + fn[i] + 1e-6)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        accuracy = (tp[i] + tn[i]) / (tp[i] + tn[i] + fp[i] + fn[i] + 1e-6)
        
        results[name] = {
            "IoU": iou,
            "Accuracy": accuracy,
            "Precision": precision,
            "F1": f1
        }
    return results

def save_visual_comparison(model, loader, device, out_dir="final_test_visuals", max_images=None):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    count = 0
    
    # We need the RGB directory to load original high-quality images
    dataset = loader.dataset
    rgb_dir = dataset.rgb_dir

    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device)
            masks = batch["labels"] 
            ids = batch["id"]
            
            preds = (torch.sigmoid(model(imgs)) > 0.5).cpu().numpy()
            
            for i in range(len(ids)):
                if max_images is not None and count >= max_images: return
                
                sample_id = ids[i]
                # Load ORIGINAL high-quality RGB image
                rgb_path = os.path.join(rgb_dir, f"{sample_id}.jpg")
                if not os.path.exists(rgb_path):
                    rgb_path = os.path.join(rgb_dir, f"{sample_id}.png")
                
                if not os.path.exists(rgb_path):
                    continue
                
                orig_img = Image.open(rgb_path).convert("RGB")
                orig_w, orig_h = orig_img.size
                img_np = np.array(orig_img)
                
                # Overlay canvas
                overlay = np.zeros_like(img_np, dtype=np.uint8)
                
                # Get GT and Pred masks
                gt_masks = masks[i].cpu().numpy()
                
                for c in range(preds.shape[1]):
                    # Resize masks to original image scale
                    gt = np.array(Image.fromarray((gt_masks[c] > 0.5).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.NEAREST)).astype(bool)
                    pr = np.array(Image.fromarray((preds[i, c] > 0.5).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.NEAREST)).astype(bool)
                    
                    # False Negative (GT=1, Pred=0) -> Red
                    overlay[gt & ~pr] = [255, 0, 0]
                    # False Positive (GT=0, Pred=1) -> Blue
                    overlay[pr & ~gt] = [0, 0, 255]
                    # True Positive (GT=1, Pred=1) -> Green
                    overlay[gt & pr] = [0, 255, 0]

                # Alpha blending: combine with original
                alpha = 0.4
                mask_indices = np.any(overlay > 0, axis=-1)
                
                combined = img_np.copy()
                combined[mask_indices] = (alpha * overlay[mask_indices] + (1 - alpha) * img_np[mask_indices]).astype(np.uint8)
                
                Image.fromarray(combined).save(os.path.join(out_dir, f"{sample_id}.png"))
                count += 1
                orig_img.close()

def print_results(split_name, results):
    """Print a formatted metrics table for a given split."""
    print(f"\n{'='*60}")
    print(f"  Split: {split_name}")
    print(f"{'='*60}")
    print(f"{'Class':<15} | {'IoU':<8} | {'Acc':<8} | {'Prec':<8} | {'F1':<8}")
    print("-" * 60)
    all_ious, all_accs, all_precs, all_f1s = [], [], [], []
    for name, m in results.items():
        print(f"{name:<15} | {m['IoU']:8.4f} | {m['Accuracy']:8.4f} | {m['Precision']:8.4f} | {m['F1']:8.4f}")
        all_ious.append(m['IoU'])
        all_accs.append(m['Accuracy'])
        all_precs.append(m['Precision'])
        all_f1s.append(m['F1'])
    print("-" * 60)
    print(f"{'Mean':<15} | {np.mean(all_ious):8.4f} | {np.mean(all_accs):8.4f} | {np.mean(all_precs):8.4f} | {np.mean(all_f1s):8.4f}")
    print("=" * 60)

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "checkpoints/best_hybrid_model.pth"
    BATCH_SIZE = 4

    # Identical seed as main.py for reproducible splits
    random.seed(42)

    # Train split
    train_dataset = AgricultureVisionMultiLabel("data", "train", transform=BasicTransform())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Val split: 50/50 into validation and hold-out test
    full_val_dataset = AgricultureVisionMultiLabel("data", "val", transform=BasicTransform())
    val_samples = full_val_dataset.samples.copy()
    random.shuffle(val_samples)
    split_idx = len(val_samples) // 2

    def make_subset(base_dataset, sample_paths):
        """Create a dataset subset by filtering samples AND cached_metadata."""
        ids = {os.path.splitext(os.path.basename(p))[0] for p in sample_paths}
        subset = AgricultureVisionMultiLabel("data", "val", transform=BasicTransform())
        subset.samples = sample_paths
        subset.cached_metadata = [m for m in base_dataset.cached_metadata if m['id'] in ids]
        return subset

    val_dataset  = make_subset(full_val_dataset, val_samples[:split_idx])
    test_dataset = make_subset(full_val_dataset, val_samples[split_idx:])

    val_loader  = DataLoader(val_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # Load Best Model
    model = AgrarUNetPlusPlus(num_classes=len(CLASS_NAMES)).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"Model loaded from '{CHECKPOINT}'.")
    else:
        print("Checkpoint not found!"); return

    # Evaluate all splits
    splits = {
        "Train":         train_loader,
        "Validation":    val_loader,
        "Hold-Out Test": test_loader,
    }
    for split_name, loader in splits.items():
        print(f"\nEvaluating '{split_name}' ({len(loader.dataset)} samples)...")
        results = compute_metrics(model, loader, DEVICE)
        print_results(split_name, results)

    # Generate visuals for the Hold-Out Test set
    print("\nGenerating visuals for Hold-Out Test set...")
    save_visual_comparison(model, test_loader, DEVICE, out_dir="final_test_visuals", max_images=None)
    print("Visuals saved in 'final_test_visuals'.")

if __name__ == "__main__":
    main()
