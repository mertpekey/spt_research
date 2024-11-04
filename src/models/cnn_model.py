import torch
import torch.nn as nn
import torchvision.models as models

class SpatialTranscriptomicsModel(nn.Module):
    def __init__(self, num_genes, num_proteins):
        super(SpatialTranscriptomicsModel, self).__init__()
        self.cnn = models.resnet18(pretrained=True)
        self.cnn.fc = nn.Identity()
        
        self.gene_fc = nn.Sequential(nn.Linear(num_genes, 256), nn.ReLU())
        
        self.fc = nn.Sequential(
            nn.Linear(256 + 512, 256),
            nn.ReLU(),
            nn.Linear(256, num_proteins)
        )
        
    def forward(self, image, gene_data):
        img_features = self.cnn(image)
        gene_features = self.gene_fc(gene_data)
        combined = torch.cat((img_features, gene_features), dim=1)
        return self.fc(combined)
