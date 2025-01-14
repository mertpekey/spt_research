import os
from dotenv import load_dotenv

import torch.nn as nn
import timm
from huggingface_hub import login
from torchvision import transforms

class UNIModel(nn.Module):
    """
    UNI Vision Transformer model wrapper
    """
    
    def __init__(self, pretrained=True):
        super().__init__()

        # Login to Hugging Face
        load_dotenv()
        login(token=os.getenv('HF_TOKEN'))
        
        # Initialize model
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/uni",
            pretrained=pretrained,
            init_values=1e-5,
            dynamic_img_size=True
        )
        
        # Get model-specific transforms for the processor
        self.processor = transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        
        # Set hidden size for compatibility with HF models
        self.config = type('Config', (), {'hidden_size': 1024})()  # Create config object with hidden_size
    
    def forward(self, pixel_values):
        """
        Forward pass that matches HF model interface
        
        Args:
            pixel_values: Input tensor of shape [B, C, H, W]
            
        Returns:
            Object with pooler_output containing features [B, hidden_size]
        """
        features = self.model(pixel_values)
        
        # Create a simple namespace object with pooler_output
        return type('ModelOutput', (), {'pooler_output': features})() 