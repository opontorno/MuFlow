"""
Auxiliary modules for MuFlow training.

This module contains:
    - ProjectionLayer: convolutional projection bottleneck for contrastive learning
    - contrastive_loss: Siamese contrastive loss function
    - infonce_loss: InfoNCE contrastive loss function
    - SRMScorer: SRM-filter-based anomaly scorer for noise-level artifact detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import kurtosis as scipy_kurtosis
from sklearn.ensemble import IsolationForest


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


# ===========================================================================
# SRM-based Anomaly Scorer
# ===========================================================================

def build_srm_kernels():
    """
    Build 30 SRM (Spatial Rich Model) high-pass filter kernels from the
    steganalysis literature (Fridrich & Kodovský, 2012).

    Returns:
        np.ndarray of shape (30, 5, 5)
    """
    filters = []

    # ── 1st-order edge detectors ─────────────────────────────────────────
    for (r1, c1, r2, c2) in [(2, 1, 2, 2), (1, 2, 2, 2),
                               (1, 1, 2, 2), (1, 3, 2, 2)]:
        f = np.zeros((5, 5), dtype=np.float32)
        f[r1, c1], f[r2, c2] = -1, 1
        filters.append(f)

    # ── 2nd-order Laplacian variants ─────────────────────────────────────
    # 3×3 Laplacian
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = 1, 1, 1, 1
    f[2, 2] = -4
    filters.append(f)

    for (r1, c1, r2, c2) in [(2, 1, 2, 3), (1, 2, 3, 2),
                               (1, 1, 3, 3), (1, 3, 3, 1)]:
        f = np.zeros((5, 5), dtype=np.float32)
        f[r1, c1], f[r2, c2] = 1, 1
        f[2, 2] = -2
        filters.append(f)

    # ── 3rd-order ────────────────────────────────────────────────────────
    f = np.zeros((5, 5), dtype=np.float32)
    f[2, 0], f[2, 4] = -1, -1; f[2, 1], f[2, 3] = 3, 3; f[2, 2] = -4
    filters.append(f / 3.0)

    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 2], f[4, 2] = -1, -1; f[1, 2], f[3, 2] = 3, 3; f[2, 2] = -4
    filters.append(f / 3.0)

    # ── SQUARE 3×3 ──────────────────────────────────────────────────────
    f = np.zeros((5, 5), dtype=np.float32)
    f[1:4, 1:4] = 1.0 / 8.0; f[2, 2] = -1.0
    filters.append(f)

    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = 2, 2, 2, 2
    f[1, 1], f[1, 3], f[3, 1], f[3, 3] = 1, 1, 1, 1
    f[2, 2] = -12
    filters.append(f / 12.0)

    # ── SQUARE 5×5 ──────────────────────────────────────────────────────
    f = np.ones((5, 5), dtype=np.float32) / 24.0; f[2, 2] = -1.0
    filters.append(f)

    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 2], f[2, 0], f[2, 4], f[4, 2] = 1, 1, 1, 1
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = 2, 2, 2, 2
    f[1, 1], f[1, 3], f[3, 1], f[3, 3] = 1, 1, 1, 1
    f[2, 2] = -16
    filters.append(f / 16.0)

    # ── Sobel / Prewitt ──────────────────────────────────────────────────
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 2], f[1, 3] = -1, -2, -1
    f[3, 1], f[3, 2], f[3, 3] = 1, 2, 1
    filters.append(f / 4.0)

    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[2, 1], f[3, 1] = -1, -2, -1
    f[1, 3], f[2, 3], f[3, 3] = 1, 2, 1
    filters.append(f / 4.0)

    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[1, 3], f[2, 3] = -1, -1, -1
    f[2, 1], f[3, 1], f[3, 2] = 1, 1, 1
    filters.append(f / 3.0)

    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 2], f[2, 1] = -1, -1, -1
    f[2, 3], f[3, 2], f[3, 3] = 1, 1, 1
    filters.append(f / 3.0)

    # ── Roberts cross ────────────────────────────────────────────────────
    for (r1, c1) in [(1, 1), (1, 3)]:
        f = np.zeros((5, 5), dtype=np.float32)
        f[r1, c1], f[2, 2] = 1, -1
        filters.append(f)

    # ── LoG 5×5 ──────────────────────────────────────────────────────────
    f = np.array([[0,  0, -1,  0,  0],
                  [0, -1, -2, -1,  0],
                  [-1, -2, 16, -2, -1],
                  [0, -1, -2, -1,  0],
                  [0,  0, -1,  0,  0]], dtype=np.float32)
    filters.append(f / 16.0)

    # ── Checkerboard ─────────────────────────────────────────────────────
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 3], f[3, 1], f[3, 3] = 1, 1, 1, 1
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = -1, -1, -1, -1
    filters.append(f / 4.0)

    # ── High-frequency period-2 ──────────────────────────────────────────
    f = np.zeros((5, 5), dtype=np.float32)
    f[2, 0], f[2, 2], f[2, 4] = 1, -1, 1
    f[2, 1], f[2, 3] = -1, -1
    filters.append(f / 3.0)

    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 2], f[2, 2], f[4, 2] = 1, -1, 1
    f[1, 2], f[3, 2] = -1, -1
    filters.append(f / 3.0)

    # ── Full 5×5 Laplacian ───────────────────────────────────────────────
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 2], f[1, 3] = 1, 1, 1
    f[2, 1], f[2, 3] = 1, 1
    f[3, 1], f[3, 2], f[3, 3] = 1, 1, 1
    f[0, 2], f[2, 0], f[2, 4], f[4, 2] = 1, 1, 1, 1
    f[2, 2] = -12
    filters.append(f / 12.0)

    # ── Diagonal / anti-diagonal HF ──────────────────────────────────────
    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 0], f[2, 2], f[4, 4] = 1, -2, 1
    filters.append(f / 2.0)

    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 4], f[2, 2], f[4, 0] = 1, -2, 1
    filters.append(f / 2.0)

    # Exactly 30 kernels
    filters = filters[:30]
    while len(filters) < 30:
        f_rot = np.rot90(filters[len(filters) % len(filters)]).copy()
        filters.append(f_rot)

    return np.stack(filters, axis=0)  # (30, 5, 5)


class SRMConv(nn.Module):
    """
    Fixed (non-trainable) convolution applying 30 SRM kernels to each
    input channel independently, producing 30 residual maps (averaged
    across RGB channels).
    """

    def __init__(self):
        super().__init__()
        kernels = build_srm_kernels()                        # (30, 5, 5)
        # Expand for 3 input channels: (90, 1, 5, 5) — groups=3
        weight = torch.from_numpy(kernels).unsqueeze(1)      # (30, 1, 5, 5)
        weight = weight.repeat(3, 1, 1, 1)                   # (90, 1, 5, 5)
        self.register_buffer('weight', weight)
        self.n_filters = 30

    @torch.no_grad()
    def forward(self, x):
        """
        Args:
            x: (B, 3, H, W) RGB images in [0, 1] or [0, 255]
        Returns:
            residuals: (B, 30, H, W) — averaged across RGB channels
        """
        B, C, H, W = x.shape
        out = F.conv2d(x, self.weight, padding=2, groups=C)   # (B, 90, H, W)
        out = out.view(B, C, self.n_filters, H, W).mean(dim=1)
        return out


class SRMScorer:
    """
    SRM-filter-based anomaly scorer.

    Workflow:
        1. ``extract_stats(images)`` — apply 30 SRM filters, compute per-filter
           energy, variance, kurtosis → 90-dim feature vector per image.
        2. ``fit(stats)`` — fit an Isolation Forest on *real* image stats.
        3. ``score(images)`` — return anomaly scores (higher = more anomalous).

    The scorer is designed to be used **alongside** the NF loss::

        final_score = nf_loss + λ_srm * srm_score
    """

    def __init__(self, device='cpu', random_state=42):
        self.srm_conv = SRMConv()
        self.device = device
        self.srm_conv.to(device)
        self.model = None          # IsolationForest, fitted after calling fit()
        self._is_fitted = False
        self.random_state = random_state
        # Normalisation stats (computed during fit, applied during score)
        self._fit_mean = None
        self._fit_std = None

    @property
    def is_fitted(self):
        return self._is_fitted

    @torch.no_grad()
    def extract_stats(self, images):
        """
        Extract 90-dim SRM statistics from a batch of images.

        Args:
            images: torch.Tensor (B, 3, H, W) — expected in [0, 1] range.
        Returns:
            np.ndarray (B, 90) — [energies(30) | variances(30) | kurtoses(30)]
        """
        images = images.to(self.device)
        residuals = self.srm_conv(images)                    # (B, 30, H, W)
        B, K = residuals.shape[0], residuals.shape[1]
        res_np = residuals.cpu().numpy().reshape(B, K, -1)   # (B, 30, H*W)

        energies  = np.mean(res_np ** 2, axis=2)             # (B, 30)
        variances = np.var(res_np, axis=2)                    # (B, 30)
        kurts     = scipy_kurtosis(res_np, axis=2, fisher=True)  # (B, 30)

        stats = np.concatenate([energies, variances, kurts], axis=1)  # (B, 90)
        stats = np.nan_to_num(stats, nan=0.0, posinf=0.0, neginf=0.0)
        return stats

    def fit(self, stats):
        """
        Fit the Isolation Forest on *real-image* SRM statistics.

        Args:
            stats: np.ndarray (N, 90)
        """
        self._fit_mean = stats.mean(axis=0)
        self._fit_std  = stats.std(axis=0) + 1e-8
        stats_norm = (stats - self._fit_mean) / self._fit_std

        self.model = IsolationForest(
            n_estimators=200,
            contamination='auto',
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.model.fit(stats_norm)
        self._is_fitted = True
        print(f"[SRM] Isolation Forest fitted on {stats.shape[0]} samples "
              f"(feature dim = {stats.shape[1]})")

    def score(self, images):
        """
        Compute SRM anomaly scores for a batch of images.

        Returns scores where **higher = more anomalous** (we negate the
        Isolation Forest ``decision_function`` which returns high values for
        inliers).

        Args:
            images: torch.Tensor (B, 3, H, W)
        Returns:
            np.ndarray (B,)
        """
        if not self._is_fitted:
            raise RuntimeError("SRMScorer not fitted yet. Call fit() first.")
        stats = self.extract_stats(images)
        stats_norm = (stats - self._fit_mean) / self._fit_std
        # decision_function: high → inlier, low → outlier
        # Negate so that high → outlier (consistent with NF loss convention)
        return -self.model.decision_function(stats_norm).astype(np.float64)
