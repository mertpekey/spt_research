import scanpy as sc
from PIL import Image
from torch.utils.data import DataLoader, Dataset

class SpatialDataset(Dataset):
    def __init__(self, spot_patches, gene_data, protein_data, protein_metadata, processor=None):
        self.spot_patches = spot_patches
        self.gene_data = gene_data
        self.protein_data = protein_data
        
        # Metadata
        self.protein_metadata = protein_metadata

        # Processor for transformations on images
        self.processor = processor
    
    def __len__(self):
        return len(self.spot_patches)
    
    def __getitem__(self, idx):
        spot_patch = self.spot_patches[idx]
        gene_data = self.gene_data[idx]
        protein_data = self.protein_data[idx]
        
        if self.processor:
            spot_patch = self.processor(images=spot_patch, return_tensors="pt")["pixel_values"].squeeze(0)
            # Check processor type by its interface/behavior
            if hasattr(self.processor, '__call__') and hasattr(self.processor, 'transforms'):
                # Torchvision/TIMM style transforms (like UNI model)
                spot_patch = Image.fromarray(spot_patch)
                spot_patch = self.processor(spot_patch)
            else:
                # HuggingFace style processor
                spot_patch = self.processor(images=spot_patch, return_tensors="pt")["pixel_values"].squeeze(0)
        
        return spot_patch, gene_data, protein_data


def get_data_loaders(gene_data, protein_data, spot_patches, image_processor, protein_metadata, config, shuffle=True):
    dataset = SpatialDataset(spot_patches, gene_data, protein_data, protein_metadata, image_processor)
    data_split = 'train' if shuffle else 'test'
    return DataLoader(dataset, batch_size=config['hyperparameters'][f'{data_split}_batch_size'], shuffle=shuffle)


def load_data(config, split):
    adata = sc.read_h5ad(f"data/adatas_{config[f'{split}_data']['sample_id'].split('_')[-1]}.h5ad")
    pdata = sc.read_h5ad(f"data/pdatas_{config[f'{split}_data']['sample_id'].split('_')[-1]}.h5ad")
    return adata, pdata
