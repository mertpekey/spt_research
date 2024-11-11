import wandb

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import root_mean_squared_error, mean_absolute_error


class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.loss = 0
        self.rmse = 0
        self.mae = 0
        self.pearson_corr = 0
        self.spearman_corr = 0
        self.count = 0
        self.errors = []

    def update(self, output, target):
        num_proteins = target.shape[1]
        
        output = output.flatten()
        target = target.flatten()
        self.rmse += root_mean_squared_error(target, output)
        self.mae += mean_absolute_error(target, output)
        self.pearson_corr += pearsonr(target, output)[0]
        self.spearman_corr += spearmanr(target, output)[0]
        self.count += 1

        # Calculate errors and store for visualizatiosn
        rmse_spots = np.sqrt((output.reshape(-1, num_proteins) - target.reshape(-1, num_proteins)) ** 2)
        self.errors.append(rmse_spots)

    def compute(self):
        return {
            "loss": self.loss / self.count if self.count > 0 else 0,
            "rmse": self.rmse / self.count if self.count > 0 else 0,
            "mae": self.mae / self.count if self.count > 0 else 0,
            "pearson_corr": self.pearson_corr / self.count if self.count > 0 else 0,
            "spearman_corr": self.spearman_corr / self.count if self.count > 0 else 0
        }

    def log(self, prefix, commit=True):
        metrics = self.compute()
        metrics_to_log = {}
        for metric_name, metric_value in metrics.items():
            metrics_to_log[f"{prefix}/{metric_name}"] = metric_value
        wandb.log(metrics_to_log, commit=commit)
        return metrics
    
    def print_metrics(self, prefix):
        metrics = self.compute()
        for metric_name, metric_value in metrics.items():
            print(f"{prefix} {metric_name}: {metric_value:.4f}, ", end="")
        return metrics

    def plot_mse_heatmap(self, file_name="rmse_error_heatmap.png", log_wandb=True):
        errors_np = np.vstack(self.errors)  # Shape: (num_spots, num_proteins)
        mean_errors_per_protein = errors_np.mean(axis=0).reshape(1, -1)
    
        num_proteins = mean_errors_per_protein.shape[1]
        fig_width = max(12, num_proteins / 4)
        plt.figure(figsize=(fig_width, 3))
        
        sns.heatmap(
            mean_errors_per_protein,
            cmap='coolwarm',
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7, "rotation": 90, "ha": "center", "va": "center"},
            cbar_kws={'label': 'Average RMSE Error'}
        )
        
        plt.xlabel('Protein Index', fontsize=10)
        plt.ylabel('Average Across Spots', fontsize=10)
        plt.title('Average RMSE Error Per Protein', fontsize=12)
        plt.savefig(f"supplementary/{file_name}", format='png', dpi=300, bbox_inches='tight')
        plt.close()

        if log_wandb:
            wandb.log({"RMSE Error Heatmap": wandb.Image(file_name)}, commit=False)
