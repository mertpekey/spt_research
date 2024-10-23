import torch
import torch.nn as nn
import torchvision.models as models

class CombinedModel(nn.Module):
    def __init__(self, gene_input_dim, output_dim):
        super(CombinedModel, self).__init__()
        # Use ResNet for image feature extraction
        resnet = models.resnet50(pretrained=True)
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-1])
        for param in self.resnet_features.parameters():
            param.requires_grad = False  # Freeze ResNet layers

        # Linear layers to combine image and gene expression features
        self.fc1 = nn.Linear(gene_input_dim + 2048, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.relu = nn.ReLU()

    def forward(self, gene_input, image_input):
        # Extract image features using ResNet
        image_features = self.resnet_features(image_input).view(image_input.size(0), -1)

        # Concatenate gene and image features
        combined_features = torch.cat((gene_input, image_features), dim=1)
        x = self.relu(self.fc1(combined_features))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
