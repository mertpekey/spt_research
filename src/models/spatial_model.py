import torch
import torch.nn as nn

from .hf_model import HF_Model

class SpatialModel(nn.Module):
    def __init__(self, num_genes, num_proteins, config):
        super().__init__()
        
        # Base model
        self.base_model = HF_Model(num_genes, num_proteins, config)
        self.processor = self.base_model.processor
        
        # Feature dimension from base model
        feature_dim = 256 + self.base_model.out_features
        
        # Spatial attention
        self.spatial_attention = SpatialAttention(feature_dim, num_heads=8)
        
        # Final prediction layer
        self.fc = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_proteins)
        )
    
    def forward(self, image, gene_data, coords, main_image_size):
        # Get image features
        img_features = self.base_model.image_model(pixel_values=image).pooler_output
        img_features = img_features.view(img_features.size(0), -1)
        
        # Get gene features
        gene_features = self.base_model.gene_fc(gene_data)
        
        # Combine features
        combined_features = torch.cat((img_features, gene_features), dim=1)
        
        # Apply spatial attention
        attended_features = self.spatial_attention(combined_features, coords, main_image_size)
        
        # Final prediction
        return self.fc(attended_features)
    

class SpatialAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads  # Store num_heads as instance variable
        self.attention_scale = nn.Parameter(torch.FloatTensor([10.0]))
        self.multihead_attn = nn.MultiheadAttention(feature_dim, num_heads, batch_first=True)
        
    def forward(self, features, coords, main_image_size):
        # Normalize by image size (assuming you have access to image_width and image_height)
        normalized_coords = coords.float()
        normalized_coords[:, 0] = normalized_coords[:, 0] / main_image_size[0]  # x coordinates
        normalized_coords[:, 1] = normalized_coords[:, 1] / main_image_size[1] # y coordinates
        distances = torch.cdist(normalized_coords, normalized_coords)  # [batch_size, batch_size]
        
        # Calculate distance-based attention mask
        attention_mask = torch.exp(-distances / self.attention_scale)
        
        # Expand mask for multi-head attention [batch_size, num_heads, seq_len, seq_len]
        attention_mask = attention_mask.unsqueeze(0).expand(self.num_heads, -1, -1)
        
        # Apply multihead attention
        features = features.unsqueeze(0)
        attended_features, _ = self.multihead_attn(
            features, features, features,
            attn_mask=attention_mask  # Now properly shaped for multi-head attention
        )
        
        return attended_features.squeeze(0)
        # Add residual connection
        # return attended_features.squeeze(0) + features.squeeze(0)  # x + attn(x)
