from transformers import AutoImageProcessor, AutoModel
import torch
import torch.nn as nn
from .uni_model import UNIModel

class HF_Model(nn.Module):
    def __init__(self, num_genes, num_proteins, config):
        super(HF_Model, self).__init__()
        
        # Save config for later use
        self.config = config

        # Initialize image model and processor
        self.image_model, self.processor, self.out_features = self._select_backbone(
            config['image_model']['hf_repo_id'], 
            config['image_model']['pretrained']
        )
        
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
        """
        Select and initialize the backbone model
        
        Args:
            model_name: Name/path of the model to use
            pretrained: Whether to use pretrained weights
            
        Returns:
            model: The initialized model
            processor: Image processor/transforms
            out_features: Number of output features
        """
        if model_name.lower() == "uni":
            model = UNIModel(pretrained=pretrained)
            processor = model.processor  # UNI model's transform
            out_features = model.config.hidden_size
        else:
            # Standard HuggingFace model initialization
            processor = AutoImageProcessor.from_pretrained(model_name)
            
            # Configure model based on type and attention settings
            config_kwargs = {}
            if "vit" in model_name.lower() and self.config['image_model'].get('output_attentions', False):
                config_kwargs["output_attentions"] = True
            
            if pretrained:
                model = AutoModel.from_pretrained(model_name, **config_kwargs)
            else:
                model = AutoModel.from_config(AutoModel.from_pretrained(model_name).config, **config_kwargs)

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
            if hasattr(model, "classifier"):
                model.classifier = nn.Identity()
            elif hasattr(model, "fc"):
                model.fc = nn.Identity()

        return model, processor, out_features

    def forward(self, image, gene_data):
        # Process image
        if "vit" in self.config['image_model'].get('hf_repo_id', '').lower() and self.config['image_model'].get('output_attentions', False):
            # For ViT models that need attention output
            img_output = self.image_model(pixel_values=image, output_attentions=True)
            img_features = img_output.pooler_output
            attentions = img_output.attentions
        else:
            # Standard forward pass for other models
            img_output = self.image_model(pixel_values=image)
            img_features = img_output.pooler_output
            attentions = None
            
        img_features = img_features.view(img_features.size(0), -1)
        
        # Process gene data
        gene_features = self.gene_fc(gene_data)

        # Combine features and pass through fc layers
        combined = torch.cat((img_features, gene_features), dim=1)
        output = self.fc(combined)
        
        # Return output and attentions (if applicable)
        if attentions is not None:
            return output, attentions
        else:
            return output