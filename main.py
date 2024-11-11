import yaml
import wandb
import argparse

from src.data_preprocessing import load_image, process_spot_coordinates, extract_spot_patches, load_data_with_split, set_random_seed
from src.dataset import get_data_loaders, load_data
from src.models.cnn_model import CNN_Model
from src.models.vit_model import VIT_Model
from src.train import train

def main(args):
    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Set random seed
    set_random_seed(config['random_seed'])

    if config['use_wandb']:
        wandb.init(project=config['wandb_project'], name=config['wandb_name'], config=config)

    # Load data
    adata, pdata = load_data(config)
    img_tensor = load_image(adata, config)
    coords = process_spot_coordinates(adata, config)
    spot_patches = extract_spot_patches(img_tensor, coords, config)
    train_data, val_data, test_data = load_data_with_split(adata, pdata, spot_patches, config)

    # Data loaders
    train_loader = get_data_loaders(*train_data, config, shuffle=True)
    val_loader = get_data_loaders(*val_data, config, shuffle=False)
    test_loader = get_data_loaders(*test_data, config, shuffle=False)

    # Model
    if config['model_name'] == 'resnet':
        model = CNN_Model(num_genes=train_data[0].shape[1], num_proteins=train_data[1].shape[1], pretrained = config['pretrained'])
    elif config['model_name'] == 'vit':
        model = VIT_Model(num_genes=train_data[0].shape[1], num_proteins=train_data[1].shape[1], pretrained = config['pretrained'])
    else:
        print('Model is not valid')

    # Train
    train(model, config, train_loader, val_loader, test_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, default="debug_config.yaml", help="Path to config file")
    args = parser.parse_args()
    main(args)
    