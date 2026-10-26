import torch
from torch import nn

class SimpleWalker(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )
        self.log_std = nn.Parameter(torch.full((output_dim,), 0.01 ))

    def forward(self, x):
        mean = self.mean(x)
        return torch.distributions.Normal(mean, torch.exp(self.log_std))

