"""
Projection Layer for FastFlow with Contrastive Learning.

This module provides projection layers that transform feature maps
before feeding them to the normalizing flow. The layers are trained
with contrastive loss and reconstruction loss.
"""

import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Convolutional projection layer with bottleneck architecture.
    
    Architecture:
        Conv3x3 (C → C_hidden) - Compression
        ReLU
        Conv3x3 (C_hidden → C) - Reconstruction
    
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
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        return self.projection(x)


def contrastive_loss(features1, features2, labels, margin=1.0):
    """
    Standard Contrastive Loss for Siamese networks.
    
    This loss function encourages similar pairs to have small distances
    and dissimilar pairs to have large distances (at least margin).
    
    Formula: L = Y * D^2 + (1-Y) * max(0, margin - D)^2
    
    Args:
        features1: First set of features (B, C, H, W)
        features2: Second set of features (B, C, H, W)
        labels: Binary labels (B,) where:
                - 1 = positive pair (similar, should be close)
                - 0 = negative pair (dissimilar, should be far apart)
        margin: Margin for negative pairs (default: 1.0)
    
    Returns:
        Scalar loss value
    """
    # Flatten features for distance computation
    features1_flat = features1.flatten(1)  # (B, C*H*W)
    features2_flat = features2.flatten(1)  # (B, C*H*W)
    
    # Compute Euclidean distance between pairs
    distances = torch.nn.functional.pairwise_distance(features1_flat, features2_flat, p=2)
    
    # Contrastive loss
    # For positive pairs (label=1): minimize distance -> Y * D^2
    # For negative pairs (label=0): maximize distance -> (1-Y) * max(0, margin - D)^2
    loss_positive = labels * distances.pow(2)
    loss_negative = (1 - labels) * torch.clamp(margin - distances, min=0.0).pow(2)
    
    # Average over batch
    loss = (loss_positive + loss_negative).mean()
    
    return loss

