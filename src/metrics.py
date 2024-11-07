import wandb

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

    def update(self, output, target):
        output = output.flatten()
        target = target.flatten()
        self.rmse += root_mean_squared_error(target, output)
        self.mae += mean_absolute_error(target, output)
        self.pearson_corr += pearsonr(target, output)[0]
        self.spearman_corr += spearmanr(target, output)[0]
        self.count += 1

    def compute(self):
        return {
            "loss": self.loss / self.count if self.count > 0 else 0,
            "rmse": self.rmse / self.count if self.count > 0 else 0,
            "mae": self.mae / self.count if self.count > 0 else 0,
            "pearson_corr": self.pearson_corr / self.count if self.count > 0 else 0,
            "spearman_corr": self.spearman_corr / self.count if self.count > 0 else 0
        }

    def log(self, prefix):
        metrics = self.compute()
        for metric_name, metric_value in metrics.items():
            wandb.log({f"{prefix}/{metric_name}": metric_value})
        return metrics
    
    def print_metrics(self, prefix):
        metrics = self.compute()
        for metric_name, metric_value in metrics.items():
            print(f"{prefix} {metric_name}: {metric_value:.4f}, ", end="")
        return metrics