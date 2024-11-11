import random
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torchvision.transforms as transforms


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_image(adata, config):
    img = adata.uns['spatial'][config['hires_image_key']]['images']['hires']
    img = (img * 255).astype(np.uint8) if img.dtype == np.float32 else img
    img = Image.fromarray(img)
    transform = transforms.ToTensor()
    img_tensor = transform(img)
    return img_tensor

def process_spot_coordinates(adata, config):
    scale_factor = adata.uns['spatial'][config['hires_image_key']]['scalefactors']['tissue_hires_scalef']
    coords = adata.obsm['spatial'] * scale_factor
    return coords.astype(int)

def extract_spot_patches(img_tensor, coords, config):
    process_fn = get_transformations(config)
    patch_size = config['patch_size']
    half_patch = patch_size // 2
    patches = []
    for x, y in coords:
        x, y = int(x), int(y)
        patch = img_tensor[
            :,
            max(0, y-half_patch):min(y+half_patch, img_tensor.shape[1]),
            max(0, x-half_patch):min(x+half_patch, img_tensor.shape[2])
        ]
        if patch.shape[1] == patch_size and patch.shape[2] == patch_size:
            patch = process_fn(patch) if process_fn is not None else patch
            patches.append(patch)
    return torch.stack(patches)

def get_transformations(config):
    if config['model_name'] == 'vit':
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),  # ViT requires 224x224 patches
            transforms.ToTensor()
        ])
    else:
        return None

def filter_genes_by_variance(gene_data, target_num_genes=5000):
    gene_variances = torch.var(gene_data, dim=0)
    top_gene_indices = torch.topk(gene_variances, target_num_genes).indices
    return gene_data[:, top_gene_indices]

def load_data_with_split(adata, pdata, spot_patches, config):
    gene_data = torch.tensor(adata.layers['norm'].todense()).float()
    gene_data = filter_genes_by_variance(gene_data, target_num_genes=config['target_num_genes'])
    protein_data = torch.tensor(pdata.layers['norm'].todense()).float()
    
    train_indices, test_indices = train_test_split(range(gene_data.shape[0]), test_size=0.2, random_state=config['random_seed'])
    train_indices, val_indices = train_test_split(train_indices, test_size=0.25, random_state=config['random_seed'])  # 0.25 x 0.8 = 0.2

    return (gene_data[train_indices], protein_data[train_indices], spot_patches[train_indices]), \
           (gene_data[val_indices], protein_data[val_indices], spot_patches[val_indices]), \
           (gene_data[test_indices], protein_data[test_indices], spot_patches[test_indices])
