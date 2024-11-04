import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from metrics import Metrics

def train(model, train_loader, val_loader, config):
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
            train_metrics.update(output.cpu().numpy(), proteins.cpu().numpy())
        
        # Validation
        val_metrics = evaluate(model, val_loader, criterion, device)

        # Log training loss and metrics
        if config['use_wandb']:
            wandb.log({"epoch": epoch+1})
            train_metrics.log("train")
            val_metrics.log("val")
        
        # Print and log validation metrics
        print(f"Epoch [{epoch+1}/{config['epochs']}]")
        train_metrics.print_metrics("train")
        val_metrics.print_metrics("val")
        print()

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
