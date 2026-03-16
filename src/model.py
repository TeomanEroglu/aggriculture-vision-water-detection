import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class AgrarUNetPlusPlus(nn.Module):
    """
    Segmentation model based on U-Net++ with an EfficientNet-B4 backbone.
    Configured for 4 input channels (RGB + NIR).
    """
    def __init__(self, num_classes=1): 
        super().__init__()
        
        self.model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",        
            encoder_weights="imagenet",     
            in_channels=4,                  
            classes=num_classes,            
            activation=None                 # Activation (Sigmoid) is applied externally during loss/inference
        )

    def forward(self, x):
        return self.model(x)
