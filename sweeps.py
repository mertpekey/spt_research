import yaml
import wandb
import argparse

from src.data_preprocessing import load_image, process_spot_coordinates, extract_spot_patches, load_data_with_split, set_random_seed
from src.dataset import get_data_loaders, load_data
from src.models.cnn_model import SpatialTranscriptomicsModel
from src.train import train

def main():
    wandb.init(project=args.sweep_project_name)

    # Load config
    with open(args.config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    for param, value in wandb.config.items():
        config[param] = value

    # Set random seed
    set_random_seed(config['random_seed'])

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
    model = SpatialTranscriptomicsModel(num_genes=train_data[0].shape[1], num_proteins=train_data[1].shape[1])

    # Train
    train(model, config, train_loader, val_loader, test_loader)


if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Sweep Hyperparameter Search')
    parser.add_argument('--config_path', default='configs/example_config.yaml', help='Config yaml file of model', type=str)
    parser.add_argument('--sweep_config_path', default='configs/sweep_config.yaml', help='Sweep Config yaml file path', type=str)
    parser.add_argument('--sweep_project_name', default='spt_sweep', help='Sweep Project Name in Wandb', type=str)
    parser.add_argument('--sweep_count', default=1, help='Number of Different Hyperparameter Runs', type=int)
    
    global args
    args = parser.parse_args()

    with open(args.sweep_config_path, 'r') as file:
        sweep_configuration = yaml.safe_load(file)['sweep']

    sweep_count = None if sweep_configuration['method'] == 'grid' else args.sweep_count

    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.sweep_project_name)
    wandb.agent(sweep_id, function=main, count=sweep_count)
    