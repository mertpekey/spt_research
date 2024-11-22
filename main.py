import yaml
import wandb
import argparse

from src.data_preprocessing import load_image, process_spot_coordinates, extract_spot_patches, load_data_with_split, set_random_seed, load_all_splits_randomly
from src.dataset import get_data_loaders, load_data
from src.models.cnn_model import CNN_Model
from src.models.vit_model import VIT_Model
from src.train import train

def main(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed
    set_random_seed(config['hyperparameters']['random_seed'])

    if config['logging']['use_wandb']:
        wandb.init(project=config['logging']['wandb_project'], name=config['logging']['wandb_name'], config=config)

    # Load data
    if config['train_data']['sample_id'] == config['test_data']['sample_id']:
        adata, pdata = load_data(config, split)
        img_tensor = load_image(adata, config, split)
        coords = process_spot_coordinates(adata, config, split)
        spot_patches = extract_spot_patches(img_tensor, coords, config)
        train_data, val_data, test_data = load_all_splits_randomly(adata, pdata, spot_patches, config)
        train_protein_metadata = pdata.var
        val_protein_metadata = pdata.var
        test_protein_metadata = pdata.var
    else:
        for split in ['train', 'test']:
            adata, pdata = load_data(config, split)
            img_tensor = load_image(adata, config, split)
            coords = process_spot_coordinates(adata, config, split)
            spot_patches = extract_spot_patches(img_tensor, coords, config)
            if split == 'train':
                train_data, val_data = load_data_with_split(adata, pdata, spot_patches, config, split)
                train_protein_metadata = pdata.var
                val_protein_metadata = pdata.var
            else:
                test_data = load_data_with_split(adata, pdata, spot_patches, config, split)
                test_protein_metadata = pdata.var

    # Data loaders
    train_loader = get_data_loaders(*train_data, train_protein_metadata, config, shuffle=True)
    val_loader = get_data_loaders(*val_data, val_protein_metadata, config, shuffle=False) if val_data is not None else None
    test_loader = get_data_loaders(*test_data, test_protein_metadata, config, shuffle=False)

    # Model
    if config['image_model']['model_type'] == 'resnet':
        model = CNN_Model(num_genes=train_data[0].shape[1], num_proteins=train_data[1].shape[1], config=config)
    elif config['image_model']['model_type'] == 'vit':
        model = VIT_Model(num_genes=train_data[0].shape[1], num_proteins=train_data[1].shape[1], config=config)
    else:
        print('Model is not valid')

    # Train
    train(model, config, train_loader, val_loader, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="configs/debug_config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)
    