import wandb

def initialize_logger(project_name):
    wandb.init(project=project_name)
