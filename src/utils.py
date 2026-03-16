import torch
import torch.nn as nn
import torch.nn.functional as F

def multilabel_loss(logits, labels, valid_mask):
    """
    Hybrid Loss: Combines Weighted BCE (for stability) and Dice Loss (for IoU optimization).
    Optimized for multi-label segmentation within a valid field mask.
    """
    # 1. Weighted BCE
    # Positive weight increased for "water" class to handle imbalanced data.
    pos_weight = torch.tensor([10.0], device=logits.device)
    bce = F.binary_cross_entropy_with_logits(logits, labels, reduction="none", pos_weight=pos_weight)
    
    # Masking valid field pixels
    mask = valid_mask.unsqueeze(1)
    masked_bce = (bce * mask).sum() / (mask.sum() + 1e-6)

    # 2. Dice Loss 
    probs = torch.sigmoid(logits)
    p = probs * mask
    t = labels * mask
    
    intersection = (p * t).sum()
    union = p.sum() + t.sum()
    dice_loss = 1 - (2. * intersection + 1e-6) / (union + 1e-6)

    # Combination: 1 part BCE + 1 part Dice
    return masked_bce + dice_loss

def calculate_batch_iou(logits, labels, valid_mask):
    """
    Calculate Intersection and Union for a batch on GPU.
    Returns: (intersection_sum, union_sum) as tensors per class.
    """
    # Threshold probabilities to get binary predictions
    preds = torch.sigmoid(logits) > 0.5
    
    # Adjust valid mask shape to [B, 1, H, W]
    if valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1)
        
    # Mask predictions and labels
    preds_masked = preds & valid_mask.bool()
    labels_masked = (labels > 0.5) & valid_mask.bool()

    # Calculate Intersection & Union per class (dim=1)
    intersection = (preds_masked & labels_masked).long().sum(dim=(0, 2, 3))
    union = (preds_masked | labels_masked).long().sum(dim=(0, 2, 3))

    return intersection, union

def visualize_sample(sample, class_names):
    """
    Visualize a single sample from the DataLoader for debugging purposes.
    """
    import matplotlib.pyplot as plt
    
    # Normalize/clamp values for matplotlib display
    rgb = sample["image"].permute(1, 2, 0).cpu().numpy()
    # Values are clamped to 0-1 for matplotlib display
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-6)
    
    labels = sample["labels"].cpu().numpy()
    
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    plt.title("Original RGB")
    plt.imshow(rgb)
    
    for i, name in enumerate(class_names):
        plt.subplot(3, 4, i + 2)
        plt.title(name)
        plt.imshow(labels[i], cmap='gray')
    
    plt.tight_layout()
    
    plt.show()