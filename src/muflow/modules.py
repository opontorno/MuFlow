"""
Auxiliary modules for MuFlow training.

This module contains:
    - ProjectionLayer: convolutional projection bottleneck for contrastive learning
    - contrastive_loss: Siamese contrastive loss function
    - ConvAutoencoder: per-scale convolutional autoencoder for feature reconstruction
"""

import torch
import torch.nn as nn


# ===========================================================================
# Projection Layer (contrastive learning)
# ===========================================================================

class ProjectionLayer(nn.Module):
    """
    Convolutional projection layer with bottleneck architecture.

    Architecture:
        Conv1x1 (C → C_hidden) - Compression
        ReLU
        Conv1x1 (C_hidden → C) - Reconstruction

    Args:
        in_channels (int): Number of input channels
        hidden_ratio (float): Ratio for hidden dimension (default: 0.5)
    """

    def __init__(self, in_channels, hidden_ratio=0.5):
        super(ProjectionLayer, self).__init__()

        self.in_channels = in_channels
        hidden_channels = int(in_channels * hidden_ratio)

        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        return self.projection(x)


def contrastive_loss(features1, features2, labels, margin=1.0):
    """
    Standard Contrastive Loss for Siamese networks.

    Formula: L = Y * D^2 + (1-Y) * max(0, margin - D)^2

    Args:
        features1: First set of features (B, C, H, W)
        features2: Second set of features (B, C, H, W)
        labels: Binary labels (B,) — 1=positive pair, 0=negative pair
        margin: Margin for negative pairs (default: 1.0)

    Returns:
        Scalar loss value
    """
    features1_flat = features1.flatten(1)
    features2_flat = features2.flatten(1)

    distances = torch.nn.functional.pairwise_distance(features1_flat, features2_flat, p=2)

    loss_positive = labels * distances.pow(2)
    loss_negative = (1 - labels) * torch.clamp(margin - distances, min=0.0).pow(2)

    return (loss_positive + loss_negative).mean()


# ===========================================================================
# Convolutional Autoencoder (feature reconstruction)
# ===========================================================================

class ConvAutoencoder(nn.Module):
    """
    Per-scale convolutional autoencoder for backbone feature reconstruction.

    Used to regularise the normalizing flows: the AE is trained to reconstruct
    real-image feature maps. At inference, the reconstruction error acts as an
    additional anomaly signal combined (via ae_lambda) with the Mahalanobis
    distance coming from the NF flows.

    Architecture (encoder):
        Conv3x3  C  → C_hidden    (stride 1, padding=same)
        BatchNorm + ReLU
        Conv3x3  C_hidden → C_hidden  (stride 1, padding=same)
        BatchNorm + ReLU

    Architecture (decoder, symmetric):
        Conv3x3  C_hidden → C_hidden  (stride 1, padding=same)
        BatchNorm + ReLU
        Conv3x3  C_hidden → C         (stride 1, padding=same)
        Tanh

    The spatial resolution H×W is preserved throughout (no pooling/upsampling),
    which keeps the output shape identical to the input.

    Args:
        in_channels (int): Number of feature channels C
        hidden_ratio (float): Ratio for bottleneck hidden channels (default: 0.5)
    """

    def __init__(self, in_channels: int, hidden_ratio: float = 0.5):
        super(ConvAutoencoder, self).__init__()

        self.in_channels = in_channels
        hidden_channels = max(1, int(in_channels * hidden_ratio))

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, in_channels, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor of shape (B, C, H, W)
        Returns:
            Reconstructed tensor of shape (B, C, H, W)
        """
        return self.decoder(self.encoder(x))

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes mean MSE reconstruction loss between input and reconstruction.

        Args:
            x: Feature tensor (B, C, H, W)
        Returns:
            Scalar loss
        """
        x_hat = self.forward(x)
        return torch.nn.functional.mse_loss(x_hat, x)
