import random
import numpy as np
import scanpy as sc
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_image(adata, config, split):
    img = adata.uns['spatial'][config[f'{split}_data']['hires_image_key']]['images']['hires']
    img = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
    return img

def process_spot_coordinates(adata, config, split):
    scale_factor = adata.uns['spatial'][config[f'{split}_data']['hires_image_key']]['scalefactors']['tissue_hires_scalef']
    coords = adata.obsm['spatial'] * scale_factor
    return coords.astype(int)

def extract_spot_patches(img_tensor, coords, config):
    patch_size = config['hyperparameters']['patch_size']
    half_patch = patch_size // 2
    patches = []
    for x, y in coords:
        x, y = int(x), int(y)
        patch = img_tensor[
            max(0, y-half_patch):min(y+half_patch, img_tensor.shape[0]),
            max(0, x-half_patch):min(x+half_patch, img_tensor.shape[1]),
            :,
        ]
        if patch.shape[0] == patch_size and patch.shape[1] == patch_size:
            patches.append(patch)
    return np.stack(patches)

def get_exclude_indices(config):
    sample1_var = sc.read_h5ad(f"data/adatas_{config['train_data']['sample_id'].split('_')[-1]}.h5ad").var.index
    sample2_var = sc.read_h5ad(f"data/adatas_{config['test_data']['sample_id'].split('_')[-1]}.h5ad").var.index

    common_genes = sample1_var.intersection(sample2_var)

    # Find indices in Sample 1 that are not in the common set
    exclude_indices = [i for i, gene in enumerate(sample1_var) if gene not in common_genes]
    return torch.tensor(exclude_indices)

def filter_genes_by_variance(gene_data, target_num_genes=5000, exclude_indices=None):
    gene_variances = torch.var(gene_data, dim=0)

    # Create a mask to exclude specific indices
    if exclude_indices is not None:
        exclude_mask = torch.zeros_like(gene_variances, dtype=torch.bool)
        exclude_mask[exclude_indices] = True
        # Set variance of excluded indices to a very low value
        gene_variances[exclude_mask] = -float('inf')
    
    top_gene_indices = torch.topk(gene_variances, target_num_genes).indices
    top_gene_indices = torch.sort(top_gene_indices).values
    return top_gene_indices

def map_indices_between_datasets(train_var, test_var, top_gene_indices_train):
    selected_genes_train = train_var.index[top_gene_indices_train.tolist()] # Get the gene indices from Tonsil1
    mapped_indices_test = test_var.index.get_indexer(selected_genes_train) # Find corresponding indices in Tonsil2
    mapped_indices_test = mapped_indices_test[mapped_indices_test != -1]

    if len(mapped_indices_test) != len(top_gene_indices_train):
        raise ValueError('Failed to map indices between datasets')
    
    return torch.tensor(mapped_indices_test)

def load_data_with_split(adata, pdata, spot_patches, coords, config, split, top_gene_indices=None, train_gene_info_df=None, exclude_indices=None):
    gene_data = torch.tensor(adata.layers['norm'].todense()).float()
    protein_data = torch.tensor(pdata.layers['norm'].todense()).float()

    if split == 'train':
        if config['hyperparameters']['use_validation']:
            train_indices, val_indices = train_test_split(
                range(gene_data.shape[0]), 
                test_size=0.2, 
                random_state=config['hyperparameters']['random_seed']
            )
            top_gene_indices = filter_genes_by_variance(
                gene_data[train_indices, :], 
                target_num_genes=config['hyperparameters']['target_num_genes'], 
                exclude_indices=exclude_indices
            )

            train_gene_data = gene_data[train_indices, :]
            val_gene_data = gene_data[val_indices, :]

            return (
                (train_gene_data[:, top_gene_indices], protein_data[train_indices], spot_patches[train_indices], coords[train_indices]),
                (val_gene_data[:, top_gene_indices], protein_data[val_indices], spot_patches[val_indices], coords[val_indices]),
                top_gene_indices
            )
        else:
            top_gene_indices = filter_genes_by_variance(
                gene_data, 
                target_num_genes=config['hyperparameters']['target_num_genes'], 
                exclude_indices=exclude_indices
            )
            return (gene_data[:, top_gene_indices], protein_data, spot_patches, coords), None, top_gene_indices
    
    elif split == 'test':
        if train_gene_info_df is not None and top_gene_indices is not None:
            mapped_indices_test = map_indices_between_datasets(train_gene_info_df, adata.var, top_gene_indices)
            gene_data = gene_data[:, mapped_indices_test]
        return (gene_data, protein_data, spot_patches, coords)
    
    else:
        raise ValueError('Invalid split')

def load_all_splits_randomly(adata, pdata, spot_patches, coords, config):
    gene_data = torch.tensor(adata.layers['norm'].todense()).float()
    protein_data = torch.tensor(pdata.layers['norm'].todense()).float()
    
    train_indices, test_indices = train_test_split(
        range(gene_data.shape[0]), 
        test_size=0.2, 
        random_state=config['hyperparameters']['random_seed']
    )
    train_indices, val_indices = train_test_split(
        train_indices, 
        test_size=0.25, 
        random_state=config['hyperparameters']['random_seed']
    )

    top_gene_indices = filter_genes_by_variance(
        gene_data[train_indices, :], 
        target_num_genes=config['hyperparameters']['target_num_genes']
    )

    return (
        (gene_data[train_indices, top_gene_indices], protein_data[train_indices], spot_patches[train_indices], coords[train_indices]),
        (gene_data[val_indices, top_gene_indices], protein_data[val_indices], spot_patches[val_indices], coords[val_indices]),
        (gene_data[test_indices, top_gene_indices], protein_data[test_indices], spot_patches[test_indices], coords[test_indices])
    )
