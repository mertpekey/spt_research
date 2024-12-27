import torch
import torch.nn as nn
from src.models.hf_model import HF_Model

class TwoStageModel(nn.Module):
    def __init__(self, num_genes, num_proteins, config):
        super(TwoStageModel, self).__init__()
        
        # First stage model
        self.first_stage = HF_Model(num_genes, num_proteins, config)
        self.processor = self.first_stage.processor
        
        # Second stage layers (only created if using predictions as features)
        self.use_predictions = config['image_model'].get('use_predictions_as_features', False)
        if self.use_predictions:
            self.second_stage = nn.Sequential(
                nn.Linear(num_proteins + 256 + self.first_stage.out_features, 256),
                nn.ReLU(),
                nn.Linear(256, num_proteins)
            )
    
    def forward(self, image, gene_data):
        # First stage prediction
        first_pred = self.first_stage(image, gene_data)
        
        if not self.use_predictions:
            return first_pred
            
        # Get intermediate features from first stage
        img_features = self.first_stage.image_model(pixel_values=image).pooler_output
        img_features = img_features.view(img_features.size(0), -1)
        gene_features = self.first_stage.gene_fc(gene_data)
        
        # Combine all features for second stage
        combined = torch.cat((first_pred, img_features, gene_features), dim=1)
        return self.second_stage(combined) 