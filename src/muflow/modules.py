"""
Auxiliary modules for MuFlow training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def contrastive_loss(features1, features2, labels, margin=1.0):
    """
    Standard Contrastive Loss for Siamese networks.

    Formula: L = Y * D^2 + (1-Y) * max(0, margin - D)^2
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
    """
    a = F.normalize(anchors.flatten(1),   dim=1)
    p = F.normalize(positives.flatten(1), dim=1)
    n = F.normalize(negatives.flatten(1), dim=1)

    pos_sim = (a * p).sum(dim=1, keepdim=True) / temperature
    neg_sim = torch.mm(a, n.t()) / temperature
    logits  = torch.cat([pos_sim, neg_sim], dim=1)
    targets = torch.zeros(a.size(0), dtype=torch.long, device=a.device)
    return F.cross_entropy(logits, targets)
