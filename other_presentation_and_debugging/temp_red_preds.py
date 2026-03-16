import os
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

from src.data_loader import AgricultureVisionMultiLabel, BasicTransform, CLASS_NAMES
from src.model import AgrarUNetPlusPlus

# Target field IDs extracted from the image
TARGET_FIELDS = [
    "7Y3ACJ1FZ",
    "9FLI6WXJG",
    "19UH8942B",
    "GJQ4F4C6F",
    "HYA94W99C",
    "J3GVGXF1W",
    "LPEQWANP3",
    "P8WQZAA4H",
    "RJRH1W2AM"
]

def save_red_visuals(model, loader, device, out_dir="temp_red_crops"):
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    
    dataset = loader.dataset
    rgb_dir = dataset.rgb_dir

    print(f"Generating red-only prediction crops in {out_dir}...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Predicting crops"):
            imgs = batch["image"].to(device)
            ids = batch["id"]
            
            # Filter for IDs that belong to our target fields
            indices_to_process = []
            for idx, sample_id in enumerate(ids):
                field_id = sample_id.split('_')[0]
                if field_id in TARGET_FIELDS:
                    indices_to_process.append(idx)
            
            if not indices_to_process:
                continue

            outputs = model(imgs)
            preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
            
            for i in indices_to_process:
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
                
                # We only want predicted classes in RED
                # Assuming CLASS_NAMES[0] is "water"
                for c in range(preds.shape[1]):
                    pr = np.array(Image.fromarray((preds[i, c] > 0.5).astype(np.uint8)).resize((orig_w, orig_h), resample=Image.NEAREST)).astype(bool)
                    overlay[pr] = [255, 0, 0] # Pure RED for prediction

                # Alpha blending
                alpha = 0.5
                mask_indices = np.any(overlay > 0, axis=-1)
                
                combined = img_np.copy()
                combined[mask_indices] = (alpha * overlay[mask_indices] + (1 - alpha) * img_np[mask_indices]).astype(np.uint8)
                
                Image.fromarray(combined).save(os.path.join(out_dir, f"{sample_id}.png"))
                orig_img.close()

def stitch_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    pattern = re.compile(r"(.+)_(\d+)-(\d+)-(\d+)-(\d+)\.(png|jpg|jpeg)", re.IGNORECASE)
    fields = defaultdict(list)
    
    for filename in os.listdir(input_dir):
        match = pattern.match(filename)
        if match:
            field_id = match.group(1)
            x1, y1, x2, y2 = map(int, match.group(2, 3, 4, 5))
            fields[field_id].append({
                'filename': filename,
                'coords': (x1, y1, x2, y2)
            })

    print(f"Stitching {len(fields)} fields into {output_dir}...")
    for field_id in tqdm(fields, desc="Stitching fields"):
        crops = fields[field_id]
        
        min_x = min(c['coords'][0] for c in crops)
        min_y = min(c['coords'][1] for c in crops)
        max_x = max(c['coords'][2] for c in crops)
        max_y = max(c['coords'][3] for c in crops)
        
        width = max_x - min_x
        height = max_y - min_y
        
        full_image = Image.new('RGB', (width, height))
        
        for crop in crops:
            crop_path = os.path.join(input_dir, crop['filename'])
            crop_img = Image.open(crop_path)
            x1, y1, x2, y2 = crop['coords']
            pos = (x1 - min_x, y1 - min_y)
            full_image.paste(crop_img, pos)
            crop_img.close()
            
        output_path = os.path.join(output_dir, f"{field_id}.png")
        full_image.save(output_path)
        full_image.close()

def main():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CHECKPOINT = "checkpoints/best_hybrid_model.pth"
    CROP_DIR = "temp_red_crops"
    RESULT_DIR = "reconstructed_red_preds"

    # Dataset for all validation samples (then we filter in the loop)
    # This is safer to ensure we don't miss patches that might be in different random splits
    val_dataset = AgricultureVisionMultiLabel("data", "val", transform=BasicTransform())
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

    # Load Model
    model = AgrarUNetPlusPlus(num_classes=len(CLASS_NAMES)).to(DEVICE)
    if os.path.exists(CHECKPOINT):
        model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
        print(f"Model loaded from {CHECKPOINT}.")
    else:
        print("Checkpoint not found!"); return

    # 1. Save visuals for crops
    save_red_visuals(model, val_loader, DEVICE, out_dir=CROP_DIR)

    # 2. Stitch crops together
    stitch_images(CROP_DIR, RESULT_DIR)
    
    print(f"\nDone! Reconstructed images are in '{RESULT_DIR}'.")

if __name__ == "__main__":
    main()
