import torch
import random
from torch.utils.data import DataLoader
import os
from torch.nn.utils import clip_grad_norm_
import time
from tqdm import tqdm

# Project-specific imports
from src.data_loader import AgricultureVisionMultiLabel, BasicTransform, AugmentTransform, CLASS_NAMES
from src.model import AgrarUNetPlusPlus 
from src.utils import multilabel_loss, calculate_batch_iou

# ---------------------------------------------------------
# 1) Global Configuration
# ---------------------------------------------------------
ROOT_DIR = "data"
BATCH_SIZE = 12  
NUM_WORKERS = 8    
EPOCHS = 20          
LEARNING_RATE = 3e-4 
WEIGHT_DECAY = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

random.seed(42) # Set seed for reproducible validation/test split

class ComposeTransforms:
    """Utility class to compose multiple transformations."""
    def __init__(self, transforms): 
        self.transforms = transforms
    def __call__(self, sample):
        for t in self.transforms: 
            sample = t(sample)
        return sample

def main():
    scaler = torch.amp.GradScaler('cuda')
    os.makedirs("checkpoints", exist_ok=True)

    # 2) Data & Transformations
    train_transform = ComposeTransforms([AugmentTransform(), BasicTransform()])
    val_transform = BasicTransform() 

    train_dataset = AgricultureVisionMultiLabel(ROOT_DIR, "train", transform=train_transform, debug=True)
    full_val_dataset = AgricultureVisionMultiLabel(ROOT_DIR, "val", transform=val_transform, debug=False)

    val_samples = full_val_dataset.samples
    random.shuffle(val_samples) 

    # Split the official validation data 50/50 into validation and test sets
    split_idx = len(val_samples) // 2

    val_dataset = AgricultureVisionMultiLabel(ROOT_DIR, "val", transform=val_transform)
    val_dataset.samples = val_samples[:split_idx]

    test_dataset = AgricultureVisionMultiLabel(ROOT_DIR, "val", transform=val_transform)
    test_dataset.samples = val_samples[split_idx:]

    # DataLoader setup
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

    # 3) Model, Optimizer & Training Setup
    print(f"\nInitializing Model on: {DEVICE}")
    model = AgrarUNetPlusPlus(num_classes=len(CLASS_NAMES)).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience = 5
    early_stop_counter = 0

    print(f"Data: Train={len(train_dataset)}, Val={len(val_dataset)}")
    print(f"Starting training with optimization for 8GB VRAM...\n")

    for epoch in range(EPOCHS):
        start_time = time.time() 
        
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0

        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=True)
        
        for batch in train_pbar:
            imgs = batch["image"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            masks = batch["valid_mask"].to(DEVICE)

            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                outputs = model(imgs)
                loss = multilabel_loss(outputs, labels, masks)
            
            scaler.scale(loss).backward()
            
            # Gradient Clipping
            scaler.unscale_(optimizer)
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        total_inter = torch.zeros(len(CLASS_NAMES), device=DEVICE, dtype=torch.long)
        total_union = torch.zeros(len(CLASS_NAMES), device=DEVICE, dtype=torch.long)

        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False)
        
        with torch.no_grad():
            for v_batch in val_pbar:
                v_imgs = v_batch["image"].to(DEVICE)
                v_labels = v_batch["labels"].to(DEVICE)
                v_masks = v_batch["valid_mask"].to(DEVICE)
                
                with torch.amp.autocast('cuda'):
                    v_out = model(v_imgs)
                    v_l = multilabel_loss(v_out, v_labels, v_masks)
                
                val_loss += v_l.item()

                # Accumulate IoU metrics
                batch_inter, batch_union = calculate_batch_iou(v_out, v_labels, v_masks)
                total_inter += batch_inter
                total_union += batch_union

        # --- EPOCH STATISTICS ---
        epoch_duration = time.time() - start_time
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        iou_per_class = total_inter.float() / (total_union.float() + 1e-6)
        mean_iou = iou_per_class.mean().item()

        print(f"\n>> EPOCH {epoch+1} COMPLETED <<")
        print(f"Duration:      {epoch_duration:.2f}s ({epoch_duration/60:.2f} min)")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Mean IoU:   {mean_iou:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 30)

        # --- CHECKPOINTING LOGIC ---
        
        # 1. Save epoch checkpoint
        epoch_model_name = f"checkpoints/model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), epoch_model_name)

        # 2. Save best loss model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "checkpoints/best_loss_model.pth")
            print(f"--> [SAVE] New best loss! Saved to 'best_loss_model.pth'")

        # 3. Save best IoU model (MAIN MODEL)
        if mean_iou > best_val_iou:
            best_val_iou = mean_iou
            torch.save(model.state_dict(), "checkpoints/best_hybrid_model.pth")
            print(f"--> [SAVE] New best IoU! 'best_hybrid_model.pth' updated.")
            early_stop_counter = 0 
        else:
            early_stop_counter += 1
            print(f"No IoU improvement (Patience: {early_stop_counter}/{patience})")

        scheduler.step()

        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without IoU improvement.")
            break

    print("\nTraining completed. Best model saved at 'checkpoints/best_hybrid_model.pth'.")

if __name__ == '__main__':
    main()
