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

def get_data_loaders(spot_patches, gene_data, protein_data, config, shuffle=True):
    dataset = SpatialDataset(spot_patches, gene_data, protein_data)
    return DataLoader(dataset, batch_size=config['batch_size'], shuffle=shuffle)

def load_data(config):
    sample_id = config['sample_id']
    tissue = sample_id.split("_")[2]
    # Initialize loader
    loader = LoadVisiumCITEseq(tissue=tissue, sample_id=sample_id)
    # Load origional raw counts
    loader.load_data(config['h5_file_path'])
    # Filter lower quality cells and genes, option to remove isotype controls
    loader.qc_control(min_n_cells=10, min_n_genes=500, min_pct_genes=1, max_n_genes=None,max_pct_counts_mt=10, remove_isotype=True)
    # Normalize data
    loader.normalize(return_raw=False)
    # Extract preprocessed mRNA (adata) and protein (pdata)
    (adata, pdata) = loader.get_data()
    return adata, pdata


### LoadVisiumCITEseq class is a pipeline for loading 10x Visium data
### It has methods for loading data, quality control, normalization, and returning the data
### The class is used to preprocess the data before training the model
class LoadVisiumCITEseq:
    def __init__(self, tissue="", visium_dir="", sample_id=None, name=""):
        super().__init__()
        self.tissue = tissue
        self.name = name
        self.visium_dir = visium_dir
        self.sample_id = sample_id
        self.adata = None
        self.pdata = None
        self.data = None
        print('Created 10x Visium data loader pipeline.')
        
    def load_data(self, h5_file: str):
        visium_ = sc.read_visium(path=os.path.dirname(h5_file))  
        data = sc.read_10x_h5(h5_file, gex_only=False)
        
        data.uns['spatial'] = visium_.uns['spatial']  
        data.obsm['spatial'] = visium_.obsm['spatial']
        data.obsm['spatial'] = data.obsm['spatial'].astype(float)
        data.var_names_make_unique()
        data.var_names_make_unique()

        self.data = data

        adata = data[:, data.var.feature_types=='Gene Expression']
        adata.var["mt"] = adata.var_names.str.startswith("MT-")
        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True)
                 
        pdata = data[:, data.var.feature_types=='Antibody Capture']
        self.adata = adata
        self.pdata = pdata
        
    def qc_control(self, min_n_cells=5, min_n_genes=200, min_pct_genes=1, max_n_genes=None,max_pct_counts_mt=None, remove_isotype=True):
        sc.pp.filter_genes(self.adata, min_cells=min_n_cells)
        sc.pp.filter_cells(self.adata, min_genes=min_n_genes)
        if min_pct_genes != 1:
            sc.ppc.filter_genes(self.adata, min_cells=min_pct_genes*self.adata.nobs)
        
        if not max_pct_counts_mt is None:
            self.adata = self.adata[self.adata.obs["pct_counts_mt"] < max_pct_counts_mt]
        
        if not max_n_genes is None:
            self.adata = self.adata[self.adata.obs["n_genes_by_counts"] < max_n_genes]
        
        if remove_isotype:
            self.pdata.var["isotype_control"] = (self.pdata.var_names.str.startswith("mouse_") \
            | self.pdata.var_names.str.startswith("rat_") \
            | self.pdata.var_names.str.startswith("HLA_")\
            | self.pdata.var_names.str.startswith("mouse.") \
            | self.pdata.var_names.str.startswith("rat.") \
            | self.pdata.var_names.str.startswith("HLA."))

            self.pdata = self.pdata[:, self.pdata.var.isotype_control==False] 

        self.pdata = self.pdata[self.adata.obs_names,:]
  
    def get_data(self):
        return(self.adata, self.pdata)
 
    def normalize(self, return_raw=False):
        self.adata.layers['raw'] = self.adata.X.copy()
        sc.pp.normalize_total(self.adata)
        sc.pp.log1p(self.adata)
        self.adata.layers['norm'] = self.adata.X.copy()
        
        self.pdata.layers['raw'] = self.pdata.X.copy()
        pt.pp.clr(self.pdata)
        self.pdata.layers['norm'] = self.pdata.X.copy()
    
        if return_raw:
            self.adata.X = self.adata.layers['raw']
            self.padata.X = self.pdata.layers['raw']
