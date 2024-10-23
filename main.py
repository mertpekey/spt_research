import wandb
from src.data.data_loader import load_data, GeneExpressionDataset
from src.models.combined_model import CombinedModel
from src.trainers.trainer import Trainer
from torch.utils.data import DataLoader

if __name__ == '__main__':
    # Initialize wandb
    wandb.init(project='spatial_transcriptomics')

    # Load data
    adata = load_data('data/Spatial_CITEseq_Brain/filtered_feature_bc_matrix.h5')

    # Prepare DataLoader
    gene_dataset = GeneExpressionDataset(adata)
    train_loader = DataLoader(gene_dataset, batch_size=16, shuffle=True)

    # Initialize model
    model = CombinedModel(gene_input_dim=adata.shape[1], output_dim=adata.shape[1])

    # Train the model
    trainer = Trainer(model=model, dataloader=train_loader)
    trainer.train(epochs=10)
