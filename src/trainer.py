import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from lightning.fabric import Fabric
import wandb

class Trainer:
    def __init__(self, model, dataloader, learning_rate=0.001):
        self.model = model
        self.dataloader = dataloader
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    def train(self, epochs=10):
        fabric = Fabric(accelerator="gpu", precision="16-mixed")  # Use Fabric for mixed precision
        fabric.setup(self.model, self.optimizer)

        for epoch in range(epochs):
            running_loss = 0.0
            for i, (gene_input, protein_target) in enumerate(self.dataloader):
                # Forward pass
                with fabric.autocast():
                    image_input = torch.randn((gene_input.size(0), 3, 224, 224))  # Dummy image input
                    outputs = self.model(gene_input, image_input)
                    loss = self.criterion(outputs, protein_target)

                # Backward pass
                fabric.backward(loss)
                self.optimizer.step()
                self.optimizer.zero_grad()

                running_loss += loss.item()

            avg_loss = running_loss / len(self.dataloader)
            wandb.log({'epoch': epoch + 1, 'loss': avg_loss})
            print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')

        print('Training complete.')
