import scanpy as sc
from torch.utils.data import Dataset, DataLoader
import torch

class GeneExpressionDataset(Dataset):
    def __init__(self, anndata):
        self.data = anndata

    def __len__(self):
        return self.data.n_obs

    def __getitem__(self, idx):
        gene_expression = self.data[idx].X.toarray()
        protein_expression = self.data[idx].layers['norm']  # Access normalized protein data
        return torch.tensor(gene_expression, dtype=torch.float32), torch.tensor(protein_expression, dtype=torch.float32)

def load_data(h5_file_path):
    # Load AnnData object from file
    adata = sc.read_10x_h5(h5_file_path)
    return adata
