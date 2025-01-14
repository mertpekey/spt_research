import wandb
import copy

import torch
import torch.nn as nn
import torch.optim as optim

from src.metrics import Metrics

def train(model, config, train_loader, val_loader = None, test_loader = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])

    best_val_loss = float('inf')
    best_model_weights = None
    
    for epoch in range(config['hyperparameters']['epochs']):
        model.train()
        train_metrics = Metrics()

        for img, genes, proteins in train_loader:
            img, genes, proteins = img.to(device), genes.to(device), proteins.to(device)
            optimizer.zero_grad()
            output = model(img, genes)
            loss = criterion(output, proteins)
            loss.backward()
            optimizer.step()

            # Update training metrics
            train_metrics.loss += loss.item()
            train_metrics.update(output.detach().cpu().numpy(), proteins.detach().cpu().numpy())
        
        # Validation
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device)
            val_loss = val_metrics.loss / val_metrics.count
            # Best model saving
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_weights = copy.deepcopy(model.state_dict())  # Copy the model weights
                best_model_weights = {k: v.cpu() for k, v in best_model_weights.items()}

            # Log training loss and metrics
            if config['logging']['use_wandb']:
                wandb.log({"epoch": epoch+1}, commit=False)
                train_metrics.log("train", commit=False)
                val_metrics.log("val", commit=True)
        
            # Print and log validation metrics
            print(f"Epoch [{epoch+1}/{config['hyperparameters']['epochs']}]")
            train_metrics.print_metrics("train")
            val_metrics.print_metrics("val")
            print()
        else:
            best_model_weights = copy.deepcopy(model.state_dict())
            best_model_weights = {k: v.cpu() for k, v in best_model_weights.items()}
            # Log training loss and metrics
            if config['logging']['use_wandb']:
                wandb.log({"epoch": epoch+1}, commit=False)
                train_metrics.log("train", commit=True)

    # Evaluate on Test Set
    if test_loader is not None:
        # Load best model weights
        model.load_state_dict(best_model_weights)
        model.to(device)
        print("Best model loaded.")
    
        test_metrics = evaluate(model, test_loader, criterion, device, train_loader.dataset.protein_metadata)
        
        test_metrics.plot_pearson_heatmap(file_name="test_pearson_heatmap.png", log_wandb=config['logging']['use_wandb'])
        test_metrics.plot_spearman_heatmap(file_name="test_spearman_heatmap.png", log_wandb=config['logging']['use_wandb'])
        test_metrics.plot_rmse_heatmap(file_name="test_rmse_heatmap.png", log_wandb=config['logging']['use_wandb'])

        test_metrics.plot_pearson_boxplot(file_name="test_pearson_boxplot.png", log_wandb=config['logging']['use_wandb'])
        test_metrics.plot_spearman_boxplot(file_name="test_spearman_boxplot.png", log_wandb=config['logging']['use_wandb'])
        test_metrics.plot_rmse_boxplot(file_name="test_rmse_boxplot.png", log_wandb=config['logging']['use_wandb'])

        test_metrics.plot_pearson_boxplot_order(file_name="test_pearson_boxplot_v2.png", log_wandb=config['logging']['use_wandb'])
        
        if config['logging']['use_wandb']:
            test_metrics.log("test", commit=True)
        test_metrics.print_metrics("test")


def evaluate(model, data_loader, criterion, device, train_protein_metadata = None):
    model.eval()
    eval_metrics = Metrics()
    eval_metrics.protein_metadata = data_loader.dataset.protein_metadata
    eval_metrics.set_seen_unseen_indices(train_protein_metadata)

    with torch.no_grad():
        for img, genes, proteins in data_loader:
            img, genes, proteins = img.to(device), genes.to(device), proteins.to(device)
            
            # Only get the seen proteins (for now)
            proteins = proteins[:, eval_metrics.seen_indices]

            output = model(img, genes)
            loss = criterion(output, proteins)

            # Update evaluation metrics
            eval_metrics.loss += loss.item()
            eval_metrics.update(output.cpu().numpy(), proteins.cpu().numpy())
    return eval_metrics