import os
import scanpy as sc
from muon import prot as pt
from anndata import AnnData

from torch.utils.data import DataLoader, Dataset

class SpatialDataset(Dataset):
    def __init__(self, spot_patches, gene_data, protein_data):
        self.spot_patches = spot_patches
        self.gene_data = gene_data
        self.protein_data = protein_data
    
    def __len__(self):
        return len(self.spot_patches)
    
    def __getitem__(self, idx):
        return self.spot_patches[idx], self.gene_data[idx], self.protein_data[idx]

def get_data_loaders(gene_data, protein_data, spot_patches, config, shuffle=True):
    dataset = SpatialDataset(spot_patches, gene_data, protein_data)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

def load_data(config):
    adata = sc.read_h5ad(f"data/adatas_{config['sample_id'].split('_')[-1]}.h5ad")
    pdata = sc.read_h5ad(f"data/pdatas_{config['sample_id'].split('_')[-1]}.h5ad")
    return adata, pdata
