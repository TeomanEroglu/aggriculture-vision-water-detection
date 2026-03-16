import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data_loader import AgricultureVisionMultiLabel

def calculate_dataset_stats(root_dir="data", batch_size=32):
    # Load the dataset without normalization (transform=None) to measure raw statistics
    dataset = AgricultureVisionMultiLabel(root_dir, split="train", transform=None)
    loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=False)

    cnt = 0
    fst_moment = torch.empty(4)
    snd_moment = torch.empty(4)

    print(f"Calculating Mean and Std for {len(dataset)} images...")

    for batch in tqdm(loader):
        # image shape: [B, 4, H, W]
        images = batch["image"]
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels

    mean = fst_moment
    std = torch.sqrt(snd_moment - fst_moment ** 2)

    return mean, std

if __name__ == "__main__":
    mean, std = calculate_dataset_stats()
    print("\nCalculated values for your 4 channels (RGB-NIR):")
    print(f"MEAN: {mean.tolist()}")
    print(f"STD:  {std.tolist()}")