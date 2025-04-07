import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel
from torch_geometric.nn import GCNConv, GATConv  # Import both layers
from .uni_model import UNIModel  # Assuming you have a UNIModel defined

######################################
# Alternative Fusion Modules
######################################
class GatedFusion(nn.Module):
    """
    A gated fusion module that learns to weight image and gene features.
    This version projects gene features to the image feature dimension if needed.
    """
    def __init__(self, img_dim, gene_dim, fusion_dim, hidden_dim):
        super(GatedFusion, self).__init__()
        # Projection layer if gene_dim != fusion_dim
        if gene_dim != fusion_dim:
            self.gene_proj = nn.Linear(gene_dim, fusion_dim)
        else:
            self.gene_proj = nn.Identity()
        
        self.fc = nn.Sequential(
            nn.Linear(img_dim + fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
            nn.Sigmoid()
        )
    
    def forward(self, img_features, gene_features):
        # Project gene features if needed
        gene_features_proj = self.gene_proj(gene_features)
        
        # Concatenate features and compute gate values
        combined = torch.cat((img_features, gene_features_proj), dim=1)
        gate = self.fc(combined)
        gate_img = gate[:, 0].unsqueeze(1)
        gate_gene = gate[:, 1].unsqueeze(1)
        # Weighted sum of features
        fused = gate_img * img_features + gate_gene * gene_features_proj
        return fused


######################################
# Main Model with Configurable Alternatives
######################################
class HF_ModelGraph(nn.Module):
    def __init__(self, num_genes, num_proteins, config):
        super(HF_ModelGraph, self).__init__()
        
        # Save config for later use
        self.config = config
        
        # Initialize image model and processor
        self.image_model, self.processor, self.out_features = self._select_backbone(
            config['image_model']['hf_repo_id'], 
            config['image_model']['pretrained']
        )
        if config['image_model'].get('freeze_parameters', False):
            for param in self.image_model.parameters():
                param.requires_grad = False

        self.gene_dim = config.get('model', {}).get('gene_hidden_dim', 256)

        # Gene processing layers: a deeper, non-linear branch
        self.gene_fc = nn.Sequential(
            nn.Linear(num_genes, self.gene_dim*2),
            nn.BatchNorm1d(self.gene_dim*2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.gene_dim*2, self.gene_dim),
            nn.BatchNorm1d(self.gene_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fusion Options
        self.fusion_method = config['model'].get('fusion_method', 'concat')
        
        fusion_dim = self.out_features  

        if self.fusion_method == 'gated':
            self.fusion = GatedFusion(img_dim=self.out_features, gene_dim=self.gene_dim, fusion_dim=fusion_dim, hidden_dim=128)
            self.fused_dim = fusion_dim
        elif self.fusion_method == 'concat':
            self.fusion = None
            self.fused_dim = self.out_features + self.gene_dim
        else:
            raise ValueError("Unsupported fusion_method in config")
        
        # Graph Module Options
        self.graph_layer_type = config['model'].get('graph_layer', 'GCN')
        if self.graph_layer_type == 'GCN':
            self.graph_layer = GCNConv(self.fused_dim, self.fused_dim)
        elif self.graph_layer_type == 'GAT':
            # Use GATConv with configurable number of heads
            num_heads = config['model'].get('gat_heads', 4)
            # GATConv returns (output_dim * num_heads), so we divide to maintain dimension
            self.graph_layer = GATConv(self.fused_dim, self.fused_dim // num_heads, heads=num_heads)
        else:
            raise ValueError("Unsupported graph_layer in config")
        
        # Final fully connected layers for predicting protein expression.
        self.fc = nn.Sequential(
            nn.Linear(self.fused_dim, 256),
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
            
            if hasattr(model, "classifier"):
                model.classifier = nn.Identity()
            elif hasattr(model, "fc"):
                model.fc = nn.Identity()

        return model, processor, out_features

    def forward(self, image, gene_data, edge_index):
        """
        Args:
            image: Tensor of shape (N, C, H, W) for N spots.
            gene_data: Tensor of shape (N, num_genes)
            edge_index: Graph connectivity (PyTorch Geometric edge_index) of shape (2, num_edges)
        """
        # Process image data through the backbone
        if "vit" in self.config['image_model'].get('hf_repo_id', '').lower() and self.config['image_model'].get('output_attentions', False):
            # For ViT models that need attention output
            img_output = self.image_model(pixel_values=image, output_attentions=True)
            attentions = img_output.attentions
        else:
            # Standard forward pass
            img_output = self.image_model(pixel_values=image)
            attentions = None
            
        if hasattr(img_output, "pooler_output") and img_output.pooler_output is not None:
            img_features = img_output.pooler_output
        else:
            img_features = img_output.last_hidden_state.mean(dim=1)
        img_features = img_features.view(img_features.size(0), -1)  # (N, self.out_features)
        
        # Process gene data
        gene_features = self.gene_fc(gene_data)  # (N, 256)
        
        # Fusion
        if self.fusion_method == 'gated':
            fused_features = self.fusion(img_features, gene_features)
        else:  # 'concat'
            fused_features = torch.cat((img_features, gene_features), dim=1)
        
        # Propagate information through the graph layer.
        # Choose the graph layer type based on the config.
        gcn_features = self.graph_layer(fused_features, edge_index)
        gcn_features = F.relu(gcn_features)
        
        # Final prediction
        out = self.fc(gcn_features)
        
        # Return output and attentions (if applicable)
        if attentions is not None:
            return out, attentions
        else:
            return out
