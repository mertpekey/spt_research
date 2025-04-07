import os
import torch
import torch.nn as nn
import wandb

from src.train import evaluate

def test_model(model, config, test_loader, train_loader):
    """Run evaluation on a trained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load the best model weights
    checkpoint_dir = config['logging'].get('checkpoint_dir', 'checkpoints')
    model_name = config['logging']['wandb_name']
    
    final_path = os.path.join(checkpoint_dir, f"{model_name}_final_model.pt")
    
    if os.path.exists(final_path):
        print(f"Loading final checkpoint from {final_path}")
        checkpoint_path = final_path
    else:
        raise ValueError(f"No checkpoint found for model {model_name} in {checkpoint_dir}")
    
    model.load_state_dict(torch.load(checkpoint_path))
    criterion = nn.MSELoss()
    
    # Run evaluation
    print("Starting evaluation...")
    test_metrics, test_attentions, test_coords = evaluate(model, test_loader, criterion, device, train_loader.dataset.protein_metadata, 
                          graph_k=config['hyperparameters'].get('graph_k', 5))
    
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
    
    # Log results
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

    return test_metrics