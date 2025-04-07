import yaml
import wandb
import argparse

from src.data_preprocessing import load_image, process_spot_coordinates, extract_spot_patches, load_data_with_split, set_random_seed, load_all_splits_randomly, get_exclude_indices
from src.dataset import get_data_loaders, load_data
from src.models.hf_model import HF_Model
from src.models.two_stage_model import TwoStageModel
from src.models.graph_model import HF_ModelGraph
from src.train import train
from src.test import test_model

def main(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed
    set_random_seed(config['hyperparameters']['random_seed'])

    if config['logging']['use_wandb']:
        wandb.init(project=config['logging']['wandb_project'], 
                  name=f"{config['logging']['wandb_name']}_test" if args.test else config['logging']['wandb_name'], 
                  config=config)

    # Load data
    if config['train_data']['sample_id'] == config['test_data']['sample_id']:
        adata, pdata = load_data(config, split)
        img_tensor = load_image(adata, config, split)
        coords = process_spot_coordinates(adata, config, split)
        spot_patches = extract_spot_patches(img_tensor, coords, config)
        train_data, val_data, test_data = load_all_splits_randomly(adata, pdata, spot_patches, coords, config)
        train_protein_metadata = pdata.var
        val_protein_metadata = pdata.var
        test_protein_metadata = pdata.var
    else:
        # Use only the common genes between the two samples
        exclude_indices = get_exclude_indices(config)
        for split in ['train', 'test']:
            adata, pdata = load_data(config, split)
            img_tensor = load_image(adata, config, split)
            coords = process_spot_coordinates(adata, config, split)
            spot_patches = extract_spot_patches(img_tensor, coords, config)
            if split == 'train':
                train_gene_info_df = adata.var
                train_data, val_data, top_gene_indices = load_data_with_split(adata, pdata, spot_patches, coords, config, split, exclude_indices=exclude_indices)
                train_protein_metadata = pdata.var
                val_protein_metadata = pdata.var
            else:
                test_data = load_data_with_split(adata, pdata, spot_patches, coords, config, split, top_gene_indices=top_gene_indices, train_gene_info_df=train_gene_info_df)
                test_protein_metadata = pdata.var
                if config['image_model'].get('output_attentions', False):
                    import os
                    from PIL import Image
                    os.makedirs("supplementary/temp", exist_ok=True)

                    temp_img_path = os.path.join("supplementary/temp", f"hires_{config['test_data']['sample_id']}.png")
                    Image.fromarray(img_tensor).save(temp_img_path)

    # Model
    model = HF_ModelGraph(num_genes=train_data[0].shape[1], 
                         num_proteins=train_data[1].shape[1], 
                         config=config)

    # Data loaders
    train_loader = get_data_loaders(*train_data, model.processor, train_protein_metadata, config, shuffle=True)
    val_loader = get_data_loaders(*val_data, model.processor, val_protein_metadata, config, shuffle=False) if val_data is not None else None
    test_loader = get_data_loaders(*test_data, model.processor, test_protein_metadata, config, shuffle=False)

    # Either run testing only or full training+testing
    if args.test:
        print("Running in test-only mode...")
        test_model(model, config, test_loader, train_loader)
    else:
        print("Running full training pipeline...")
        train(model, config, train_loader, val_loader, test_loader)

    if config['logging']['use_wandb']:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/debug_vit_config.yaml", help="Path to config file")
    parser.add_argument("--test", action="store_true", help="Run in test-only mode")
    args = parser.parse_args()
    main(args)
    