import yaml
import wandb
from src.data_preprocessing import load_image, process_spot_coordinates, extract_spot_patches, load_data_with_split, set_random_seed
from src.dataset import get_data_loaders, load_data
from src.models.cnn_model import SpatialTranscriptomicsModel
from src.train import train

# Load config
with open("config.yaml", 'r') as file:
    config = yaml.safe_load(file)

# Set random seed
set_random_seed(config['random_seed'])

if config['use_wandb']:
    wandb.init(project=config['wandb_project'], entity=config['wandb_entity'], config=config)

# Load data
adata, pdata = load_data(config)
img_tensor = load_image(adata, config)
coords = process_spot_coordinates(adata, config)
spot_patches = extract_spot_patches(img_tensor, coords, config)
train_data, val_data, test_data = load_data_with_split(adata, pdata, config)

# Data loaders
train_loader = get_data_loaders(spot_patches, *train_data, config, shuffle=True)
val_loader = get_data_loaders(spot_patches, *val_data, config, shuffle=False)
test_loader = get_data_loaders(spot_patches, *test_data, config, shuffle=False)

# Model
model = SpatialTranscriptomicsModel(num_genes=train_data[0].shape[1], num_proteins=train_data[1].shape[1])

# Train
train(model, train_loader, val_loader, config)
