import wandb
import copy
import os

import torch
import torch.nn as nn
import torch.optim as optim

from src.metrics import Metrics
from src.graph_codes import build_edge_index, extract_subgraph

from torch_geometric.utils import add_self_loops

def train(model, config, train_loader, val_loader = None, test_loader = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_edge_index = build_edge_index(train_loader.dataset.coords, k=config['hyperparameters']['graph_k']).to(device)
    num_nodes = len(train_loader.dataset)

    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['hyperparameters']['learning_rate'])

    start_epoch = 0
    best_val_loss = float('inf')
    best_model_weights = None
    best_optimizer_weights = None # Store optimizer state corresponding to best model

    # --- Checkpoint Loading ---
    checkpoint_dir = config['logging'].get('checkpoint_dir', 'checkpoints')
    model_name = config['logging']['wandb_name']
    
    # Define potential checkpoint paths
    best_val_model_path = os.path.join(checkpoint_dir, f"{model_name}_best_val_model.pt")
    best_val_optimizer_path = os.path.join(checkpoint_dir, f"{model_name}_best_val_optimizer.pt")
    final_model_path = os.path.join(checkpoint_dir, f"{model_name}_final_model.pt")
    final_optimizer_path = os.path.join(checkpoint_dir, f"{model_name}_final_optimizer.pt")

    # Determine which checkpoint to load if resuming
    load_model_path = None
    load_optimizer_path = None
    if config['logging'].get('resume_from_checkpoint', False):
        if val_loader is not None and os.path.exists(best_val_model_path):
            load_model_path = best_val_model_path
            load_optimizer_path = best_val_optimizer_path
            print(f"Resuming from best validation checkpoint: {load_model_path}")
        elif val_loader is None and os.path.exists(final_model_path):
            load_model_path = final_model_path
            load_optimizer_path = final_optimizer_path
            print(f"Resuming from final checkpoint: {load_model_path}")
        else:
             print("Resume specified, but no suitable checkpoint found. Starting from scratch.")

        if load_model_path and load_optimizer_path and os.path.exists(load_optimizer_path):
            model.load_state_dict(torch.load(load_model_path))
            optimizer.load_state_dict(torch.load(load_optimizer_path))
            # Note: Resuming epoch number is not implemented here, starting from epoch 0
            # You might want to save/load epoch number as well for perfect resumption
        elif load_model_path:
             # Load only model if optimizer state is missing (less common)
             model.load_state_dict(torch.load(load_model_path))
             print(f"Warning: Loaded model state but not optimizer state from {load_model_path}")


    # Create checkpoint directory if saving is enabled and dir doesn't exist
    if config['logging'].get('save_checkpoints', False):
        os.makedirs(checkpoint_dir, exist_ok=True)

    # --- Training Loop ---
    for epoch in range(start_epoch, config['hyperparameters']['epochs']):
        model.train()
        train_metrics = Metrics()

        for img, genes, proteins, coords, indices in train_loader:
            img, genes, proteins = img.to(device), genes.to(device), proteins.to(device)

            batch_indices = indices.to(device)
            batch_edge_index = extract_subgraph(global_edge_index, batch_indices, num_nodes)
            batch_edge_index, _ = add_self_loops(batch_edge_index, num_nodes=len(genes))

            optimizer.zero_grad()
            output = model(img, genes, batch_edge_index)

            if isinstance(output, tuple) and len(output) == 2:
                output, _ = output

            loss = criterion(output, proteins)
            loss.backward()
            optimizer.step()

            # Update training metrics
            train_metrics.loss += loss.item()
            train_metrics.update(output.detach().cpu().numpy(), proteins.detach().cpu().numpy(), coords.detach().cpu().numpy())
            break

        # --- Validation and Checkpointing ---
        if val_loader is not None:
            val_metrics, _, _ = evaluate(
                model, val_loader, criterion, device, train_loader.dataset.protein_metadata, graph_k=config['hyperparameters']['graph_k']
            )
            val_loss = val_metrics.loss / val_metrics.count

            # Check if this is the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"New best validation loss: {best_val_loss:.4f} at epoch {epoch+1}")
                # Store weights on CPU to save GPU memory
                best_model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
                best_optimizer_weights = {k: v for k, v in optimizer.state_dict().items()} # Keep optimizer state as is

                # Save the *best* model and optimizer state immediately
                if config['logging'].get('save_checkpoints', False):
                    torch.save(best_model_weights, best_val_model_path)
                    torch.save(best_optimizer_weights, best_val_optimizer_path)
                    print(f"Saved best validation checkpoint to {best_val_model_path}")


            # Log metrics
            if config['logging']['use_wandb']:
                wandb.log({"epoch": epoch+1}, commit=False)
                train_metrics.log("train", commit=False)
                val_metrics.log("val", commit=True)

            # Print metrics
            print(f"Epoch [{epoch+1}/{config['hyperparameters']['epochs']}]")
            train_metrics.print_metrics("train")
            val_metrics.print_metrics("val")
            print()
        else:
            # No validation - log training metrics only
            if config['logging']['use_wandb']:
                 wandb.log({"epoch": epoch+1}, commit=False)
                 train_metrics.log("train", commit=True)
                 print(f"Epoch [{epoch+1}/{config['hyperparameters']['epochs']}]")
                #  train_metrics.print_metrics("train")
                 print()
            
            # Store the latest model/optimizer state for potential saving after the loop
            best_model_weights = {k: v.cpu() for k, v in model.state_dict().items()}
            best_optimizer_weights = {k: v for k, v in optimizer.state_dict().items()}

    # --- Post-Training ---

    # Save the final model state if no validation was performed
    if val_loader is None and config['logging'].get('save_checkpoints', False):
         if best_model_weights is not None and best_optimizer_weights is not None:
             torch.save(best_model_weights, final_model_path)
             torch.save(best_optimizer_weights, final_optimizer_path)
             print(f"Saved final model checkpoint to {final_model_path}")
         else:
             print("Warning: No model/optimizer state found to save after training without validation.")


    # Evaluate on Test Set using the best found weights
    print("\nEvaluating on test set...")
    if test_loader is not None:
        if best_model_weights is not None:
            print("Loading best model weights for final evaluation...")
            model.load_state_dict(best_model_weights)
            model.to(device) # Ensure model is on the correct device after loading state_dict
        else:
            print("Warning: No best model weights found to load for testing. Using the current model state.")


        test_metrics, test_attentions, test_coords = evaluate(model, test_loader, criterion, device, train_loader.dataset.protein_metadata, graph_k=config['hyperparameters']['graph_k'])

    # Visualize attention maps if available
    if test_attentions and test_coords and "vit" in config['image_model'].get('hf_repo_id', '').lower() and config['image_model'].get('output_attentions', False):
        from src.visualization import visualize_attention_maps
        
        # Get patch size from config
        patch_size = config['hyperparameters'].get('patch_size', 224)
            
        # Save high-res image temporarily
        temp_img_path = os.path.join("supplementary/temp", f"hires_{config['test_data']['sample_id']}.png")
        
        # Create save path for attention map
        save_path = os.path.join("supplementary", f"attention_map_{model_name}.png")
        
        # Visualize attention maps
        visualize_attention_maps(
            attentions=test_attentions,
            coords=test_coords,
            patch_size=patch_size,
            hires_image_path=temp_img_path,
            save_path=save_path,
            model_name=model_name
        )
        
        # Clean up temp file
        try:
            os.remove(temp_img_path)
        except:
            pass
        
        # Add wandb_name to plot filenames
        model_name = config['logging']['wandb_name']
        test_metrics.plot_pearson_heatmap(file_name=f"test_pearson_heatmap_{model_name}.png", log_wandb=config['logging']['use_wandb'])
        test_metrics.plot_spearman_heatmap(file_name=f"test_spearman_heatmap_{model_name}.png", log_wandb=config['logging']['use_wandb'])
        test_metrics.plot_rmse_heatmap(file_name=f"test_rmse_heatmap_{model_name}.png", log_wandb=config['logging']['use_wandb'])

    test_metrics.plot_pearson_boxplot(file_name=f"test_pearson_boxplot_{model_name}.png", log_wandb=config['logging']['use_wandb'])
    test_metrics.plot_spearman_boxplot(file_name=f"test_spearman_boxplot_{model_name}.png", log_wandb=config['logging']['use_wandb'])
    test_metrics.plot_rmse_boxplot(file_name=f"test_rmse_boxplot_{model_name}.png", log_wandb=config['logging']['use_wandb'])

    test_metrics.plot_pearson_boxplot_order(file_name=f"test_pearson_boxplot_v2_{model_name}.png", log_wandb=config['logging']['use_wandb'])

    if config['logging']['use_wandb']:
        test_metrics.log("test", commit=True)
    test_metrics.print_metrics("test")

    # Save spot-level metrics
    test_metrics.save_spot_metrics_to_csv(file_path=f"supplementary/test_spot_metrics_{model_name}.csv")


def evaluate(model, data_loader, criterion, device, train_protein_metadata = None, graph_k=5):
    model.eval()
    eval_metrics = Metrics()
    eval_metrics.protein_metadata = data_loader.dataset.protein_metadata
    eval_metrics.set_seen_unseen_indices(train_protein_metadata)

    # Build edge_index for the evaluation dataset
    global_edge_index = build_edge_index(data_loader.dataset.coords, k=graph_k).to(device)
    num_nodes = len(data_loader.dataset)

    # Initialize lists for attention maps and coordinates
    all_attentions = []
    all_coords = []

    iii = 0

    with torch.no_grad():
        for img, genes, proteins, coords, indices in data_loader:
            img, genes, proteins = img.to(device), genes.to(device), proteins.to(device)

            batch_indices = indices.to(device)
            batch_edge_index = extract_subgraph(global_edge_index, batch_indices, num_nodes)
            batch_edge_index, _ = add_self_loops(batch_edge_index, num_nodes=len(genes))
            
            # Only get the seen proteins (for now)
            proteins = proteins[:, eval_metrics.seen_indices]

            # Forward pass
            output = model(img, genes, batch_edge_index)
            
            # Check if model returned attention maps
            if isinstance(output, tuple) and len(output) == 2:
                output, attentions = output
                
                # Extract the last layer's attention (usually most informative)
                last_layer_attn = attentions[-1]  # Shape: (batch_size, num_heads, seq_len, seq_len)
                
                # Add one attention tensor per image in the batch
                for i in range(last_layer_attn.shape[0]):
                    all_attentions.append(last_layer_attn[i].cpu().numpy())
                    all_coords.append(coords[i].cpu().numpy())

            loss = criterion(output, proteins)

            # Update evaluation metrics
            eval_metrics.loss += loss.item()
            eval_metrics.update(output.cpu().numpy(), proteins.cpu().numpy(), coords.cpu().numpy())

            iii+=1
            if iii == 4:
                break
    
    return eval_metrics, all_attentions, all_coords