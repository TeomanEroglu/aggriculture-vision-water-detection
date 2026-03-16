# Agriculture Vision – Water Segmentation

A deep learning pipeline for semantic segmentation of water/waterway regions in aerial agricultural imagery using the [Agriculture-Vision dataset](https://www.agriculture-vision.com/).

## Model

**U-Net++ with EfficientNet-B4 backbone** (`segmentation_models_pytorch`)
- Input: 4 channels (RGB + NIR)
- Output: Binary segmentation mask (water class)
- Loss: Multi-label binary cross-entropy (masked)
- Metric: IoU (Intersection over Union)

## Project Structure

```
aggriculture-vision/
├── main.py            # Training script
├── evaluate.py        # Evaluation & visualization
├── src/
│   ├── data_loader.py # Dataset class, transforms, augmentation
│   ├── model.py       # AgrarUNetPlusPlus model definition
│   ├── utils.py       # Loss function, IoU calculation
│   └── calculate_stats.py
├── checkpoints/       # Saved model weights
├── data/              # Agriculture-Vision dataset (not included)
└── requirements.txt
```

## Setup

```bash
pip install torch torchvision segmentation-models-pytorch tqdm matplotlib pandas pillow numpy
```

> **Note:** PyTorch must be installed separately with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/).

## Data

Download the [Agriculture-Vision dataset](https://www.agriculture-vision.com/) and place it in the `data/` directory with the following structure:

```
data/
├── train/
│   ├── images/rgb/
│   ├── images/nir/
│   ├── labels/water/
│   ├── boundaries/
│   └── masks/
└── val/
    └── ...
```

## Usage

**Train:**
```bash
python main.py
```

**Evaluate:**
```bash
python evaluate.py
```

Checkpoints are saved automatically to `checkpoints/`:
- `best_hybrid_model.pth` – best IoU model (main)
- `best_loss_model.pth` – best validation loss model
- `model_epoch_N.pth` – per-epoch checkpoint

## Training Config

| Parameter       | Value   |
|----------------|---------|
| Epochs         | 20      |
| Batch Size     | 12      |
| Learning Rate  | 3e-4    |
| Optimizer      | AdamW   |
| Scheduler      | CosineAnnealingLR |
| Early Stopping | 5 epochs (IoU) |
| Input Size     | 384×384 |
