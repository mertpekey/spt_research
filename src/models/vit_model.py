import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class VIT_Model(nn.Module):
    def __init__(self, num_genes, num_proteins, config):
        super(VIT_Model, self).__init__()
        
        if config['pretrained']:
            self.vit = ViTModel.from_pretrained(config['vit_model_name'])
            if config['freeze_image_model']:
                for param in self.vit.parameters():
                    param.requires_grad = False
        else:
            self.vit = ViTModel(ViTConfig())

        vit_output_dim = self.vit.config.hidden_size
        self.vit_fc = nn.Linear(vit_output_dim, 256)

        self.gene_fc = nn.Sequential(nn.Linear(num_genes, 256), nn.ReLU())
        
        self.fc = nn.Sequential(
            nn.Linear(256 + 256, 256),
            nn.ReLU(),
            nn.Linear(256, num_proteins)
        )
        
    def forward(self, image, gene_data):
        vit_outputs = self.vit(pixel_values=image)
        img_features = self.vit_fc(vit_outputs.pooler_output)  # [batch_size, 256]
        gene_features = self.gene_fc(gene_data)  # [batch_size, 256]
        combined = torch.cat((img_features, gene_features), dim=1)  # [batch_size, 512]
        return self.fc(combined)