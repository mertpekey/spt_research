import wandb

import torch
import torch.nn as nn
import torch.optim as optim

from src.metrics import Metrics

def train(model, config, train_loader, val_loader, test_loader = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
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
            break
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log training loss and metrics
        if config['use_wandb']:
            wandb.log({"epoch": epoch+1}, commit=False)
            train_metrics.log("train", commit=False)
            val_metrics.log("val", commit=True)
        
        # Print and log validation metrics
        print(f"Epoch [{epoch+1}/{config['epochs']}]")
        train_metrics.print_metrics("train")
        val_metrics.print_metrics("val")
        print()

    # Evaluate on Test Set
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, criterion, device)
        if config['use_wandb']:
            test_metrics.log("test", commit=True)
        test_metrics.print_metrics("test")


def evaluate(model, data_loader, criterion, device):
    model.eval()
    eval_metrics = Metrics()
    with torch.no_grad():
        for img, genes, proteins in data_loader:
            img, genes, proteins = img.to(device), genes.to(device), proteins.to(device)
            output = model(img, genes)
            loss = criterion(output, proteins)

            # Update evaluation metrics
            eval_metrics.loss += loss.item()
            eval_metrics.update(output.cpu().numpy(), proteins.cpu().numpy())
    return eval_metrics