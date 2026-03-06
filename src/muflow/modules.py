"""
Auxiliary modules for MuFlow training.

This module contains:
    - ProjectionLayer: convolutional projection bottleneck for contrastive learning
    - contrastive_loss: Siamese contrastive loss function
    - infonce_loss: InfoNCE contrastive loss function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


def infonce_loss(anchors, positives, negatives, temperature=0.07):
    """
    InfoNCE (Noise-Contrastive Estimation) loss.

    For each anchor, maximises similarity with the positive while minimising
    similarity with the negatives using a softmax-based cross-entropy
    formulation.

    Formula:
        L = -log( exp(sim(a, p) / τ) / (exp(sim(a, p) / τ) + Σ_j exp(sim(a, n_j) / τ)) )

    where sim is cosine similarity and τ is the temperature.

    Args:
        anchors:    Feature tensor (B, C, H, W) — anchor samples
        positives:  Feature tensor (B, C, H, W) — positive matches (one per anchor)
        negatives:  Feature tensor (B, C, H, W) — negative samples (one per anchor,
                    all negatives in the batch are shared across anchors)
        temperature (float): Temperature scaling factor (default: 0.07)

    Returns:
        Scalar loss value
    """
    # Flatten spatial dimensions: (B, C, H, W) -> (B, D)
    a = anchors.flatten(1)
    p = positives.flatten(1)
    n = negatives.flatten(1)

    # L2-normalise for cosine similarity
    a = F.normalize(a, dim=1)
    p = F.normalize(p, dim=1)
    n = F.normalize(n, dim=1)

    # Positive similarity: (B,) — each anchor with its own positive
    pos_sim = (a * p).sum(dim=1, keepdim=True) / temperature  # (B, 1)

    # Negative similarities: each anchor against ALL negatives in the batch
    neg_sim = torch.mm(a, n.t()) / temperature  # (B, B)

    # Logits: positive in column 0, negatives in columns 1..B
    logits = torch.cat([pos_sim, neg_sim], dim=1)  # (B, 1+B)

    # Target: the positive is always at index 0
    targets = torch.zeros(a.size(0), dtype=torch.long, device=a.device)

    return F.cross_entropy(logits, targets)
