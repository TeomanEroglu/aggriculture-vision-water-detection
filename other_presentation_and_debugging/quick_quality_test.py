import torch
import os
from PIL import Image
from evaluate import save_visual_comparison
from src.data_loader import AgricultureVisionMultiLabel, BasicTransform, CLASS_NAMES
from src.model import AgrarUNetPlusPlus
from torch.utils.data import DataLoader

def test():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = AgricultureVisionMultiLabel("data", "val", transform=BasicTransform())
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    
    model = AgrarUNetPlusPlus(num_classes=len(CLASS_NAMES)).to(DEVICE)
    # Checkpoint is optional for this test, we just want to see if it SAVES a file
    
    print("Testing save_visual_comparison for 1 sample...")
    save_visual_comparison(model, loader, DEVICE, out_dir="test_verify_quality", max_images=1)
    
    if os.path.exists("test_verify_quality"):
        files = os.listdir("test_verify_quality")
        if files:
            print(f"File saved: {files[0]}")
            img = Image.open(os.path.join("test_verify_quality", files[0]))
            img_np = np.array(img)
            print(f"Image shape: {img_np.shape}")
            # Check if there are any non-neutral pixels (not just original image)
            # This is a bit hard without knowing the original, but we can check if we applied the overlay logic correctly
        else:
            print("No files saved in directory.")
    else:
        print("Directory 'test_verify_quality' not created!")

if __name__ == "__main__":
    test()
