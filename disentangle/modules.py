import torch
import torch.nn as nn
import torch.nn.functional as F


class DENTG_Classifier(nn.Module):
    def __init__(self, 
                in_channels: int,
                out_channels: int):
        super(DENTG_Classifier, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, out_channels)
        )
    
    def forward(self, latent):
        """
        latent vector z to probs,
        Z_r and Z_s share the same structure
        """
        return self.layer(latent)
