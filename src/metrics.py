import wandb

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scanpy as sc
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
        self.pearson_values = []
        self.spearman_values = []

    def update(self, output, target):
        num_proteins = target.shape[1]
        
        output = output.flatten()
        target = target.flatten()
        self.rmse += root_mean_squared_error(target, output)
        self.mae += mean_absolute_error(target, output)
        self.pearson_corr += pearsonr(target, output)[0]
        self.spearman_corr += spearmanr(target, output)[0]
        self.count += 1

        # Calculate per-protein Pearson and Spearman correlations and errors across spots
        output_matrix = output.reshape(-1, num_proteins)
        target_matrix = target.reshape(-1, num_proteins)

        rmse_spots = np.sqrt((output_matrix - target_matrix) ** 2)
        self.errors.append(rmse_spots)
        
        per_protein_pearson = [pearsonr(output_matrix[:, i], target_matrix[:, i])[0] for i in range(num_proteins)]
        per_protein_spearman = [spearmanr(output_matrix[:, i], target_matrix[:, i])[0] for i in range(num_proteins)]

        self.pearson_values.append(per_protein_pearson)
        self.spearman_values.append(per_protein_spearman)

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

    def plot_pearson_heatmap(self, file_name="pearson_heatmap.png", log_wandb=True):
        self._plot_heatmap(self.pearson_values, "Pearson Correlation", file_name, log_wandb, high_good=True)
    
    def plot_spearman_heatmap(self, file_name="spearman_heatmap.png", log_wandb=True):
        self._plot_heatmap(self.spearman_values, "Spearman Correlation", file_name, log_wandb, high_good=True)

    def plot_rmse_heatmap(self, file_name="rmse_heatmap.png", log_wandb=True):
        self._plot_heatmap(self.errors, "RMSE", file_name, log_wandb, high_good=False)

    def _plot_heatmap(self, heatmap_values, title, file_name, log_wandb, high_good=True):
        heatmap_np = np.vstack(heatmap_values)  # Shape: (num_spots, num_proteins)
        mean_val_per_protein = heatmap_np.mean(axis=0)  # Average correlation per protein

        # Load gene IDs
        pdata_gene_ids = self.protein_metadata['gene_ids']
        
        # Sort by correlation and gene IDs
        sorted_indices = np.argsort(-mean_val_per_protein if high_good else mean_val_per_protein)
        mean_val_per_protein = mean_val_per_protein[sorted_indices].reshape(1, -1)
        sorted_gene_ids = pdata_gene_ids.iloc[sorted_indices]
    
        num_proteins = mean_val_per_protein.shape[1]
        fig_width = max(12, num_proteins / 4)
        plt.figure(figsize=(fig_width, 3))

        cmap = "coolwarm_r" if high_good else "coolwarm"
        sns.heatmap(
            mean_val_per_protein,
            cmap=cmap,
            annot=True,
            fmt=".3f",
            annot_kws={"size": 7, "rotation": 90, "ha": "center", "va": "center"},
            cbar_kws={'label': f'Average {title}'}
        )

        plt.xticks(
            ticks=np.arange(num_proteins) + 0.5,
            labels=sorted_gene_ids,
            rotation=90,
            ha="center",
            fontsize=8
        )
        plt.yticks([])
        
        plt.xlabel('Protein Gene ID', fontsize=10)
        plt.ylabel('Average Across Spots', fontsize=10)
        plt.title(f'Average {title} Per Protein', fontsize=12)
        plt.savefig(f"supplementary/{file_name}", format='png', dpi=300, bbox_inches='tight')
        plt.close()

        if log_wandb:
            wandb.log({f"{title} Heatmap": wandb.Image(f"supplementary/{file_name}")}, commit=False)


    def plot_pearson_boxplot(self, file_name="pearson_boxplot.png", log_wandb=True):
        self._plot_boxplot(self.pearson_values, "Pearson Correlation", file_name, log_wandb, high_good=True)
    
    def plot_spearman_boxplot(self, file_name="spearman_boxplot.png", log_wandb=True):
        self._plot_boxplot(self.spearman_values, "Spearman Correlation", file_name, log_wandb, high_good=True)

    def plot_rmse_boxplot(self, file_name="rmse_boxplot.png", log_wandb=True):
        self._plot_boxplot(self.errors, "RMSE", file_name, log_wandb, high_good=False)

    def _plot_boxplot(self, values, title, file_name, log_wandb, high_good=True):
        values_np = np.vstack(values)  # Shape: (num_spots, num_proteins)
        
        pdata_gene_ids = self.protein_metadata['gene_ids']
        
        # Sort by mean values (could be median but wanted to make consistent with heatmaps)
        mean_val_per_protein = np.mean(values_np, axis=0)
        sorted_indices = np.argsort(-mean_val_per_protein if high_good else mean_val_per_protein)
        sorted_values_np = values_np[:, sorted_indices]
        sorted_gene_ids = pdata_gene_ids.iloc[sorted_indices]

        num_proteins = sorted_values_np.shape[1]
        fig_width = max(12, num_proteins / 4)
        plt.figure(figsize=(fig_width, 6))

        # Plot boxplot
        cmap = plt.cm.coolwarm # if high_good else plt.cm.coolwarm
        colors = [cmap(i) for i in np.linspace(0, 1, num_proteins)]
        sns.boxplot(
            data=[sorted_values_np[:, i] for i in range(num_proteins)],  # Data per protein
            palette=colors,
            showfliers=False
        )

        plt.xticks(
            ticks=np.arange(num_proteins),
            labels=sorted_gene_ids,
            rotation=90,
            ha="center",
            fontsize=8
        )

        plt.xlabel('Protein Gene ID', fontsize=10)
        plt.ylabel(f'{title} Across Spots', fontsize=10)
        plt.title(f'{title} Per Protein', fontsize=12)
        
        # Save the plot to file
        plt.savefig(f"supplementary/{file_name}", format='png', dpi=300, bbox_inches='tight')
        plt.close()

        # Optionally log to WandB
        if log_wandb:
            wandb.log({f"{title} Boxplot": wandb.Image(f"supplementary/{file_name}")}, commit=False)
