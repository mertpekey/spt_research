import os
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import cv2
import seaborn as sns
import pandas as pd

def visualize_attention_maps(attentions, coords, patch_size, hires_image_path, save_path, model_name):
    """
    Visualize attention maps from ViT patches on the high-resolution image.
    
    Args:
        attentions: List of attention tensors, each with shape (num_heads, seq_len, seq_len)
                   Each tensor corresponds to a single image's attention map
        coords: List of coordinates for each image/attention tensor
        patch_size: Size of each patch (in pixels) to use for visualization
        hires_image_path: Path to the high-resolution image
        save_path: Path to save the visualization
        model_name: Name of the model for the output filename
    """
    if not attentions or not coords:
        print("No attention maps or coordinates provided")
        return
    
    if not os.path.exists(hires_image_path):
        print(f"High-resolution image not found at {hires_image_path}")
        return
    
    # Debug info
    print(f"Visualizing {len(attentions)} attention maps, first shape: {attentions[0].shape}")
    
    # Load the high-resolution image
    hires_img = Image.open(hires_image_path)
    hires_img = np.array(hires_img)
    
    # Ensure image is in RGB format for consistent processing
    if len(hires_img.shape) == 2:  # If grayscale
        hires_img = cv2.cvtColor(hires_img, cv2.COLOR_GRAY2RGB)
    
    # Initialize attention canvas and count canvas for averaging
    attention_canvas = np.zeros((hires_img.shape[0], hires_img.shape[1]), dtype=np.float32)
    count_canvas = np.zeros((hires_img.shape[0], hires_img.shape[1]), dtype=np.float32)
    
    # Process each patch's attention
    for attn, coord in zip(attentions, coords):
        # attn shape is (num_heads, seq_len, seq_len)
        # Average across attention heads
        # Shape: (seq_len, seq_len)
        attn_map = attn.mean(axis=0)
        
        # Get attention from CLS token to patch tokens
        # Shape: (seq_len,)
        cls_attn = attn_map[0, 1:]  # Skip CLS token
        
        # Reshape to patch grid (sqrt(196) = 14 for 224x224 images with 16x16 patches)
        grid_size = int(np.sqrt(cls_attn.shape[0]))
        attn_map = cls_attn.reshape(grid_size, grid_size)
        
        # Resize to patch size
        attn_map = cv2.resize(attn_map, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
        
        # Calculate position in high-res image
        x, y = coord
        x_start = max(0, int(x - patch_size // 2))
        y_start = max(0, int(y - patch_size // 2))
        x_end = min(hires_img.shape[1], int(x + patch_size // 2))
        y_end = min(hires_img.shape[0], int(y + patch_size // 2))
        
        # Calculate corresponding positions in attention map
        ax_start = max(0, patch_size // 2 - (x - x_start))
        ay_start = max(0, patch_size // 2 - (y - y_start))
        ax_end = min(patch_size, patch_size // 2 + (x_end - x))
        ay_end = min(patch_size, patch_size // 2 + (y_end - y))
        
        # Add attention map to canvas
        attention_canvas[y_start:y_end, x_start:x_end] += attn_map[ay_start:ay_end, ax_start:ax_end]
        count_canvas[y_start:y_end, x_start:x_end] += 1
    
    # Average overlapping regions
    count_canvas[count_canvas == 0] = 1  # Avoid division by zero
    attention_canvas = attention_canvas / count_canvas
    
    # Normalize attention map to 0-255 range
    attention_norm = (attention_canvas - attention_canvas.min()) / (attention_canvas.max() - attention_canvas.min())
    attention_uint8 = (attention_norm * 255).astype(np.uint8)
    
    # Create a heatmap using OpenCV's built-in colormaps
    # COLORMAP_JET gives a blue-green-yellow-red progression
    # COLORMAP_HOT gives a black-red-yellow-white progression
    # COLORMAP_INFERNO gives a black-purple-orange-yellow progression
    heatmap = cv2.applyColorMap(attention_uint8, cv2.COLORMAP_HOT)
    
    # Make sure the original image and heatmap have the same dimensions
    if hires_img.shape[:2] != attention_canvas.shape:
        original_img_resized = cv2.resize(hires_img, (attention_canvas.shape[1], attention_canvas.shape[0]))
    else:
        original_img_resized = hires_img.copy()
    
    # Convert to same data type
    original_img_resized = original_img_resized.astype(np.uint8)
    
    # Create a blended version with the original image visible in the background
    # Adjust alpha values to control the visibility - 0.7 for heatmap means it's more prominent
    alpha_heatmap = 0.6  # Heatmap prominence (0.0-1.0)
    alpha_img = 0.4     # Image prominence (0.0-1.0)
    blended = cv2.addWeighted(original_img_resized, alpha_img, heatmap, alpha_heatmap, 0)
    
    # Save the visualization
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, blended)
    print(f"Saved attention visualization to {save_path}")

def visualize_predicted_proteins(predictions, coords, protein_names, save_path, figsize=(15, 12), n_cols=3):
    """
    Visualize predicted protein expressions as spatial heatmaps.
    
    Args:
        predictions: Array of predicted protein expressions [n_spots, n_proteins]
        coords: Array of spatial coordinates [n_spots, 2]
        protein_names: List of protein names
        save_path: Path to save the visualization
        figsize: Figure size (width, height)
        n_cols: Number of columns in the grid layout
    """
    n_proteins = len(protein_names)
    n_rows = (n_proteins + n_cols - 1) // n_cols  # Ceiling division
    
    # Create figure
    plt.figure(figsize=figsize)
    
    # Create custom colormap (white to red)
    cmap = LinearSegmentedColormap.from_list('custom_cmap', ['#FFFFFF', '#FF0000'])
    
    # Plot each protein
    for i, protein_name in enumerate(protein_names):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Create scatter plot
        scatter = plt.scatter(
            coords[:, 0],
            coords[:, 1],
            c=predictions[:, i],
            cmap=cmap,
            s=10,
            alpha=0.8
        )
        
        plt.colorbar(scatter, fraction=0.046, pad=0.04)
        plt.title(protein_name)
        plt.axis('equal')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved protein visualization to {save_path}")

def plot_prediction_correlation(true_values, predictions, protein_names, save_path, figsize=(12, 10)):
    """
    Plot correlation between true and predicted protein expressions.
    
    Args:
        true_values: Array of true protein expressions [n_spots, n_proteins]
        predictions: Array of predicted protein expressions [n_spots, n_proteins]
        protein_names: List of protein names
        save_path: Path to save the visualization
        figsize: Figure size (width, height)
    """
    n_proteins = len(protein_names)
    n_rows = (n_proteins + 2 - 1) // 2  # 2 columns, ceiling division
    
    plt.figure(figsize=figsize)
    
    for i, protein_name in enumerate(protein_names):
        plt.subplot(n_rows, 2, i + 1)
        
        # Get true and predicted values for this protein
        true = true_values[:, i]
        pred = predictions[:, i]
        
        # Calculate correlation coefficient
        correlation = np.corrcoef(true, pred)[0, 1]
        
        # Create scatter plot
        plt.scatter(true, pred, alpha=0.6, s=5)
        
        # Add diagonal line (perfect prediction)
        min_val = min(np.min(true), np.min(pred))
        max_val = max(np.max(true), np.max(pred))
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.7)
        
        plt.title(f"{protein_name} (r = {correlation:.3f})")
        plt.xlabel("True Expression")
        plt.ylabel("Predicted Expression")
        
        # Add grid
        plt.grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save figure
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved correlation plot to {save_path}") 