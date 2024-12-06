from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn as nn

class HF_Model(nn.Module):
    def __init__(self, num_genes, num_proteins, config):
        super(HF_Model, self).__init__()

        self.processor = AutoImageProcessor.from_pretrained(config['image_model']['hf_repo_id'])
        self.image_model, self.out_features = self._select_backbone(config['image_model']['hf_repo_id'], config['image_model']['pretrained'])
        
        # Freeze parameters if specified
        if config['image_model']['freeze_parameters']:
            for param in self.image_model.parameters():
                param.requires_grad = False

        # Gene processing layers
        self.gene_fc = nn.Sequential(
            nn.Linear(num_genes, 256),
            nn.ReLU()
        )

        # Fully connected layers combining image and gene features
        self.fc = nn.Sequential(
            nn.Linear(256 + self.out_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_proteins)
        )

    def _select_backbone(self, model_name, pretrained):
        # Load Hugging Face model
        if pretrained:
            model = AutoModel.from_pretrained(model_name)
        else:
            model = AutoModel.from_config(AutoModel.from_pretrained(model_name).config)

        # Determine output feature size
        if hasattr(model, "config") and hasattr(model.config, "hidden_size"):
            out_features = model.config.hidden_size
        elif hasattr(model, "config") and hasattr(model.config, "hidden_sizes"):
            out_features = model.config.hidden_sizes[-1]
        elif hasattr(model, "classifier") and hasattr(model.classifier, "out_features"):
            out_features = model.classifier.out_features
        else:
            raise ValueError(f"Cannot determine output features for model: {model_name}")

        # Replace classification layers with Identity if they exist
        # TODO: Burayi kontrol et
        if hasattr(model, "classifier"):
            model.classifier = nn.Identity()
        elif hasattr(model, "fc"):
            model.fc = nn.Identity()

        return model, out_features

    def forward(self, image, gene_data):
        # Process image
        img_features = self.image_model(pixel_values=image).last_hidden_state.mean(dim=1) # Average pooling
        
        # Process gene data
        gene_features = self.gene_fc(gene_data)

        # Combine features and pass through fv layers
        combined = torch.cat((img_features, gene_features), dim=1)
        return self.fc(combined)
