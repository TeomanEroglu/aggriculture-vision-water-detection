import os
from glob import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import torchvision.transforms.functional as TF
import random
from tqdm import tqdm

# ---------------------------------------------------------
# Class Configuration
# ---------------------------------------------------------
CLASS_NAMES = ["water"]
SOURCE_MAPPING = {"water": ["water", "waterway"]}
NUM_CLASSES = len(CLASS_NAMES)
STATS_FILE = "checkpoints/dataset_stats.pt" 
GLOBAL_STATS = None  

# ---------------------------------------------------------
# Helper Function: Automatic Statistics Calculation
# ---------------------------------------------------------
def get_dataset_stats(root):
    """
    Load or calculate dataset mean and std for normalization.
    """
    global GLOBAL_STATS
    if GLOBAL_STATS is not None:
        return GLOBAL_STATS
        
    if os.path.exists(STATS_FILE):
        print(f"--> Statistics loaded from: {STATS_FILE}")
        GLOBAL_STATS = torch.load(STATS_FILE, weights_only=True)
        return GLOBAL_STATS

    print("--> No statistics file found. Calculating initial values...")
    
    # Temporary dataset for calculation (no transform to measure raw data)
    base_dataset = AgricultureVisionMultiLabel(root, split="train", transform=None, calculate_stats_mode=True)
    loader = DataLoader(base_dataset, batch_size=64, num_workers=4, shuffle=False)

    cnt = 0
    fst_moment = torch.zeros(4)
    snd_moment = torch.zeros(4)

    for batch in tqdm(loader, desc="Calculating Stats"):
        images = batch["image"]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean = fst_moment.view(4, 1, 1)
    std = torch.sqrt(torch.clamp(snd_moment - fst_moment ** 2, min=1e-6)).view(4, 1, 1)
    
    GLOBAL_STATS = {'mean': mean, 'std': std}
    os.makedirs(os.path.dirname(STATS_FILE), exist_ok=True)
    torch.save(GLOBAL_STATS, STATS_FILE)
    print(f"--> Statistics saved at: {STATS_FILE}")
    return GLOBAL_STATS

# ---------------------------------------------------------
# 1) Dataset Class
# ---------------------------------------------------------
class AgricultureVisionMultiLabel(Dataset):
    """
    Custom Dataset for Agriculture-Vision multi-label segmentation.
    Supports RGB and NIR channels.
    """
    def __init__(self, root, split="train", transform=None, debug=False, calculate_stats_mode=False):
        self.root = root
        self.split = split
        self.transform = transform
        self.debug = debug

        base = os.path.join(root, split)
        self.rgb_dir = os.path.join(base, "images", "rgb")
        self.nir_dir = os.path.join(base, "images", "nir")
        self.boundary_dir = os.path.join(base, "boundaries")
        self.mask_dir = os.path.join(base, "masks")

        self.label_source_dirs = {
            target_cls: [os.path.join(base, "labels", src) for src in SOURCE_MAPPING.get(target_cls, [target_cls])]
            for target_cls in CLASS_NAMES
        }

        self.samples = sorted(glob(os.path.join(self.rgb_dir, "*.*")))
        
        # --- OPTIMIZATION: Cache file paths ---
        self.cached_metadata = []
        if not calculate_stats_mode:
            print(f"--> Caching file paths for {len(self.samples)} samples...")
            for rgb_path in tqdm(self.samples, desc="Caching Paths"):
                sample_id = self._id_from_path(rgb_path)
                
                # NIR check
                nir_path = os.path.join(self.nir_dir, sample_id + ".jpg")
                has_nir = os.path.exists(nir_path)
                
                # Label checks
                valid_label_paths = {}
                for target_cls in CLASS_NAMES:
                    valid_label_paths[target_cls] = []
                    sources = self.label_source_dirs[target_cls]
                    for src_dir in sources:
                        cls_path = os.path.join(src_dir, sample_id + ".png")
                        if os.path.exists(cls_path):
                            valid_label_paths[target_cls].append(cls_path)
                            
                self.cached_metadata.append({
                    "id": sample_id,
                    "rgb_path": rgb_path,
                    "nir_path": nir_path if has_nir else None,
                    "valid_label_paths": valid_label_paths
                })
        else:
            self.cached_metadata = None

        if not calculate_stats_mode:
            get_dataset_stats(root)

    def __len__(self):
        return len(self.samples)

    def _id_from_path(self, p):
        return os.path.splitext(os.path.basename(p))[0]

    def __getitem__(self, idx):
        if self.cached_metadata:
            meta = self.cached_metadata[idx]
            sample_id = meta["id"]
            rgb_path = meta["rgb_path"]
            nir_path_or_none = meta["nir_path"]
            label_paths_map = meta["valid_label_paths"]
        else:
            # Fallback for simple data loading
            rgb_path = self.samples[idx]
            sample_id = self._id_from_path(rgb_path)
            nir_path_or_none = os.path.join(self.nir_dir, sample_id + ".jpg")
            if not os.path.exists(nir_path_or_none):
                nir_path_or_none = None
            label_paths_map = None 
        
        # 1. Load Images
        rgb = np.array(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0

        if nir_path_or_none:
            nir = np.array(Image.open(nir_path_or_none).convert("L"), dtype=np.float32) / 255.0
        else:
            fill_val = GLOBAL_STATS['mean'][3].item() if GLOBAL_STATS else 0.5
            nir = np.full_like(rgb[:,:,0], fill_value=fill_val)
        
        nir = nir[:, :, np.newaxis]
        combined = np.concatenate([rgb, nir], axis=-1)
        image_tensor = torch.from_numpy(combined).permute(2, 0, 1)

        # 2. Load Masks & Labels
        boundary_path = os.path.join(self.boundary_dir, sample_id + ".png")
        mask_path = os.path.join(self.mask_dir, sample_id + ".png")
        
        boundary = np.array(Image.open(boundary_path).convert("L"), dtype=np.uint8)
        farm_mask = np.array(Image.open(mask_path).convert("L"), dtype=np.uint8)
        valid_mask = (boundary > 0) & (farm_mask > 0)

        labels = []
        for target_cls in CLASS_NAMES:
            combined_mask = np.zeros_like(valid_mask, dtype=bool)
            
            if self.cached_metadata:
                # Optimized path
                existing_paths = label_paths_map[target_cls]
                for cls_path in existing_paths:
                    lbl = np.array(Image.open(cls_path).convert("L"), dtype=np.uint8)
                    combined_mask = combined_mask | (lbl > 0)
            else:
                # Fallback path
                sources = self.label_source_dirs[target_cls]
                for src_dir in sources:
                    cls_path = os.path.join(src_dir, sample_id + ".png")
                    if os.path.exists(cls_path):
                        lbl = np.array(Image.open(cls_path).convert("L"), dtype=np.uint8)
                        combined_mask = combined_mask | (lbl > 0)
                        
            labels.append(combined_mask)

        labels = np.stack(labels, axis=0).astype(np.float32)
        labels[:, ~valid_mask] = 0.0

        sample = {
            "image": image_tensor,
            "labels": torch.from_numpy(labels),
            "valid_mask": torch.from_numpy(valid_mask).bool(),
            "id": sample_id
        }

        if self.transform:
            sample = self.transform(sample)
        return sample

# ---------------------------------------------------------
# 2) Transformations
# ---------------------------------------------------------
class BasicTransform:
    """Standard transformations including resize and normalization."""
    def __init__(self, size=(384, 384)):
        self.size = size

    def __call__(self, sample):
        img = sample["image"]
        labels = sample["labels"]
        mask = sample["valid_mask"].unsqueeze(0)

        img = TF.resize(img, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        labels = TF.resize(labels, self.size, interpolation=TF.InterpolationMode.NEAREST)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # Normalization using calculated global stats
        if GLOBAL_STATS:
            img = (img - GLOBAL_STATS['mean']) / GLOBAL_STATS['std']
        
        sample["image"] = img
        sample["labels"] = labels
        sample["valid_mask"] = mask.squeeze(0)
        return sample

class AugmentTransform:
    """Data augmentation for training set."""
    def __init__(self):
        self.color_jitter = torch.nn.Sequential(
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
        )

    def __call__(self, sample):
        img = sample["image"]
        labels = sample["labels"]
        mask = sample["valid_mask"]

        # 1. Random Rotation (-180 to 180 degrees)
        if random.random() < 0.7:
            angle = random.uniform(-180, 180)
            img = TF.rotate(img, angle, interpolation=TF.InterpolationMode.BILINEAR)
            labels = TF.rotate(labels, angle, interpolation=TF.InterpolationMode.NEAREST)
            mask = TF.rotate(mask.unsqueeze(0), angle, interpolation=TF.InterpolationMode.NEAREST).squeeze(0)

        # 2. Horizontal Flip
        if random.random() < 0.5:
            img = TF.hflip(img)
            labels = TF.hflip(labels)
            mask = TF.hflip(mask)
        
        # 3. Vertical Flip
        if random.random() < 0.5:
            img = TF.vflip(img)
            labels = TF.vflip(labels)
            mask = TF.vflip(mask)

        # 4. Color Jitter (RGB channels only)
        if random.random() < 0.8:
            rgb = img[:3, :, :]
            nir = img[3:, :, :]
            
            rgb_jittered = self.color_jitter(rgb)
            img = torch.cat([rgb_jittered, nir], dim=0)
            
        sample["image"] = img
        sample["labels"] = labels
        sample["valid_mask"] = mask
        return sample
