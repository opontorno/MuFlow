#!/usr/bin/env python3
"""
SRM Filter Analysis for MuFlow
===============================

This script applies Spatial Rich Model (SRM) filters — originally designed
for steganalysis — to real (FFHQ) and fake (StyleGAN/2/3, Diffusion Models)
images and analyses the statistical differences in the noise-residual domain.

The core idea: SRM filters suppress semantic content and amplify low-level
noise patterns and manipulation artefacts.  While StyleGAN images are
semantically indistinguishable from FFHQ in backbone feature space, they
leave different noise-residual fingerprints that SRM filters can reveal.

Analyses produced:
    1. Visual comparison of residual maps (real vs StyleGAN vs DM)
    2. Per-filter energy (L2 norm) distributions
    3. Spectral analysis of residuals (FFT radial power)
    4. t-SNE / PCA of residual statistics
    5. Simple linear separability test (Logistic Regression)

Usage:
    cd /home/opontorno/projects/MuFlow
    python scripts/analysis_srm_filters.py [--n_samples 200] [--save_dir scripts/.pictures/srm]
"""

import os
import sys
import argparse
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from PIL import Image, ImageFile
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

# Allow truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Add project root to path so we can import muflow
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, PROJECT_ROOT)

from muflow import constants as const

# ═══════════════════════════════════════════════════════════════════════════════
# SRM FILTER DEFINITIONS
# ═══════════════════════════════════════════════════════════════════════════════

def build_srm_kernels():
    """
    Build the 30 SRM (Spatial Rich Model) high-pass filter kernels.

    These filters come from the steganalysis literature (Fridrich & Kodovský, 2012)
    and are designed to capture noise residuals of increasing complexity:

        - 1st-order edge filters (differences)
        - 2nd-order filters (Laplacian-like)
        - 3rd-order filters
        - SQUARE 3×3 / 5×5 filters
        - EDGE 3×3 / 5×5 filters

    Returns:
        np.ndarray of shape (30, 5, 5) — 30 kernels, each 5×5
    """
    # We define the most commonly used subset of SRM filters.
    # All filters are normalised so their non-zero entries sum to 0
    # (high-pass property: DC component is suppressed).

    filters = []

    # ── 1st-order: basic edge detectors ──────────────────────────────────
    # Horizontal 1st-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[2, 1], f[2, 2] = -1, 1
    filters.append(f)

    # Vertical 1st-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[2, 2] = -1, 1
    filters.append(f)

    # Diagonal 1st-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[2, 2] = -1, 1
    filters.append(f)

    # Anti-diagonal 1st-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 3], f[2, 2] = -1, 1
    filters.append(f)

    # ── 2nd-order: Laplacian variants ────────────────────────────────────
    # 3×3 Laplacian
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = 1, 1, 1, 1
    f[2, 2] = -4
    filters.append(f)

    # Horizontal 2nd-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[2, 1], f[2, 3] = 1, 1
    f[2, 2] = -2
    filters.append(f)

    # Vertical 2nd-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[3, 2] = 1, 1
    f[2, 2] = -2
    filters.append(f)

    # Diagonal 2nd-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[3, 3] = 1, 1
    f[2, 2] = -2
    filters.append(f)

    # Anti-diagonal 2nd-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 3], f[3, 1] = 1, 1
    f[2, 2] = -2
    filters.append(f)

    # ── 3rd-order filters ────────────────────────────────────────────────
    # Horizontal 3rd-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[2, 0], f[2, 4] = -1, -1
    f[2, 1], f[2, 3] = 3, 3
    f[2, 2] = -4  # Note: sums to -1+3-4+3-1 = 0 ✓ — but we use standard SRM normalisation
    # Normalise: divide by the sum of positive entries
    f = f / 3.0
    filters.append(f)

    # Vertical 3rd-order
    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 2], f[4, 2] = -1, -1
    f[1, 2], f[3, 2] = 3, 3
    f[2, 2] = -4
    f = f / 3.0
    filters.append(f)

    # ── SQUARE 3×3 filters ───────────────────────────────────────────────
    # Average 3×3 minus centre (high pass)
    f = np.zeros((5, 5), dtype=np.float32)
    f[1:4, 1:4] = 1.0 / 8.0
    f[2, 2] = -1.0
    filters.append(f)

    # Weighted 3×3: cross pattern
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = 2, 2, 2, 2
    f[1, 1], f[1, 3], f[3, 1], f[3, 3] = 1, 1, 1, 1
    f[2, 2] = -12
    f = f / 12.0
    filters.append(f)

    # ── SQUARE 5×5 filters ───────────────────────────────────────────────
    # Average 5×5 minus centre (high pass)
    f = np.ones((5, 5), dtype=np.float32) / 24.0
    f[2, 2] = -1.0
    filters.append(f)

    # Cross-weighted 5×5
    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 2], f[2, 0], f[2, 4], f[4, 2] = 1, 1, 1, 1
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = 2, 2, 2, 2
    f[1, 1], f[1, 3], f[3, 1], f[3, 3] = 1, 1, 1, 1
    f[2, 2] = -16
    f = f / 16.0
    filters.append(f)

    # ── EDGE 3×3 filters (Sobel-like, high-pass variants) ────────────────
    # Horizontal Sobel-like high-pass
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 2], f[1, 3] = -1, -2, -1
    f[3, 1], f[3, 2], f[3, 3] = 1, 2, 1
    f = f / 4.0
    filters.append(f)

    # Vertical Sobel-like high-pass
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[2, 1], f[3, 1] = -1, -2, -1
    f[1, 3], f[2, 3], f[3, 3] = 1, 2, 1
    f = f / 4.0
    filters.append(f)

    # Diagonal Prewitt-like
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 2], f[1, 3], f[2, 3] = -1, -1, -1
    f[2, 1], f[3, 1], f[3, 2] = 1, 1, 1
    f = f / 3.0
    filters.append(f)

    # Anti-diagonal Prewitt-like
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 2], f[2, 1] = -1, -1, -1
    f[2, 3], f[3, 2], f[3, 3] = 1, 1, 1
    f = f / 3.0
    filters.append(f)

    # ── Additional high-pass kernels (commonly used in forensics) ────────
    # Roberts cross variant
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[2, 2] = 1, -1
    filters.append(f)

    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 3], f[2, 2] = 1, -1
    filters.append(f)

    # 5×5 LoG (Laplacian of Gaussian) approximation
    f = np.array([
        [0,  0, -1,  0,  0],
        [0, -1, -2, -1,  0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1,  0],
        [0,  0, -1,  0,  0],
    ], dtype=np.float32)
    f = f / 16.0
    filters.append(f)

    # Checkerboard detector
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 3], f[3, 1], f[3, 3] = 1, 1, 1, 1
    f[1, 2], f[2, 1], f[2, 3], f[3, 2] = -1, -1, -1, -1
    f = f / 4.0
    filters.append(f)

    # Horizontal high-frequency detector (period-2)
    f = np.zeros((5, 5), dtype=np.float32)
    f[2, 0], f[2, 2], f[2, 4] = 1, 1, 1
    f[2, 1], f[2, 3] = -1, -1
    f[2, 2] = -1  # centre: -1 total
    f = f / 3.0
    filters.append(f)

    # Vertical high-frequency detector (period-2)
    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 2], f[2, 2], f[4, 2] = 1, 1, 1
    f[1, 2], f[3, 2] = -1, -1
    f[2, 2] = -1
    f = f / 3.0
    filters.append(f)

    # Full 5×5 Laplacian (8-connected extended)
    f = np.zeros((5, 5), dtype=np.float32)
    f[1, 1], f[1, 2], f[1, 3] = 1, 1, 1
    f[2, 1], f[2, 3] = 1, 1
    f[3, 1], f[3, 2], f[3, 3] = 1, 1, 1
    f[0, 2], f[2, 0], f[2, 4], f[4, 2] = 1, 1, 1, 1
    f[2, 2] = -12
    f = f / 12.0
    filters.append(f)

    # Diagonal high-frequency
    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 0], f[2, 2], f[4, 4] = 1, -2, 1
    f = f / 2.0
    filters.append(f)

    # Anti-diagonal high-frequency
    f = np.zeros((5, 5), dtype=np.float32)
    f[0, 4], f[2, 2], f[4, 0] = 1, -2, 1
    f = f / 2.0
    filters.append(f)

    # ── Truncate / pad to exactly 30 ─────────────────────────────────────
    filters = filters[:30]
    while len(filters) < 30:
        # Fill remaining with rotated versions of existing filters
        idx = len(filters) % len(filters)
        f_rot = np.rot90(filters[idx]).copy()
        filters.append(f_rot)

    return np.stack(filters, axis=0)  # (30, 5, 5)


# Names for the first filters (for visualisation)
SRM_FILTER_NAMES = [
    "1st-H", "1st-V", "1st-D", "1st-AD",
    "Lap3×3", "2nd-H", "2nd-V", "2nd-D", "2nd-AD",
    "3rd-H", "3rd-V",
    "Sq3HP", "Sq3W",
    "Sq5HP", "Sq5W",
    "Sob-H", "Sob-V", "Prew-D", "Prew-AD",
    "Rob1", "Rob2",
    "LoG5×5", "Checker",
    "HiFreqH", "HiFreqV",
    "Lap5Full", "DiagHF", "ADiagHF",
    "F29", "F30",
]


class SRMConv(nn.Module):
    """
    Fixed (non-trainable) convolution layer using SRM kernels.

    Applies 30 SRM filters to each input channel independently,
    producing a (B, 30*C_in, H, W) residual map.
    For greyscale analysis set in_channels=1 (convert to luma first).
    """

    def __init__(self, in_channels=1):
        super().__init__()
        kernels = build_srm_kernels()  # (30, 5, 5)
        n_filters = kernels.shape[0]

        # Repeat for each input channel: groups=in_channels
        weight = np.zeros((n_filters * in_channels, 1, 5, 5), dtype=np.float32)
        for c in range(in_channels):
            weight[c * n_filters:(c + 1) * n_filters, 0] = kernels

        self.register_buffer(
            'weight',
            torch.from_numpy(weight)
        )
        self.in_channels = in_channels
        self.n_filters = n_filters

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) tensor, float, [0, 1] range
        Returns:
            (B, 30*C, H, W) residual maps
        """
        return F.conv2d(x, self.weight, padding=2, groups=self.in_channels)


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def load_image_as_tensor(path, size=256):
    """Load an image, resize, convert to float tensor (B=1, C=1, H, W) in [0,1]."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    # Convert to luma (greyscale) for SRM analysis
    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return torch.from_numpy(luma).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)


def load_image_rgb_tensor(path, size=256):
    """Load image as (1, 3, H, W) float tensor in [0, 1]."""
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0)


def collect_images(pattern, n_samples, seed=42):
    """Glob images and sample n_samples."""
    paths = sorted(glob(pattern, recursive=True))
    if not paths:
        print(f"  [WARN] No images found for pattern: {pattern}")
        return []
    rng = random.Random(seed)
    rng.shuffle(paths)
    return paths[:n_samples]


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_residual_stats(image_paths, srm_conv, size=256):
    """
    For each image, compute SRM residuals and extract statistics.

    Returns:
        energies: (N, 30) — per-filter L2 energy
        variances: (N, 30) — per-filter variance
        spectral: (N, 30) — per-filter mean spectral energy (FFT)
        kurtoses: (N, 30) — per-filter kurtosis (peakedness of residual distribution)
    """
    energies = []
    variances = []
    spectral = []
    kurtoses = []

    for path in tqdm(image_paths, desc="Computing SRM residuals", leave=False):
        img_t = load_image_as_tensor(path, size)  # (1, 1, H, W)
        with torch.no_grad():
            residuals = srm_conv(img_t)  # (1, 30, H, W)
        res = residuals.squeeze(0).numpy()  # (30, H, W)

        # Per-filter statistics
        e = np.sqrt((res ** 2).mean(axis=(1, 2)))  # L2 energy, (30,)
        v = res.var(axis=(1, 2))  # Variance, (30,)

        # Spectral energy: mean of |FFT|² per filter
        fft_res = np.fft.fft2(res, axes=(1, 2))
        s = np.abs(fft_res).mean(axis=(1, 2))  # (30,)

        # Kurtosis: measures tail heaviness
        mu = res.mean(axis=(1, 2), keepdims=True)
        sigma = res.std(axis=(1, 2), keepdims=True) + 1e-8
        kurt = (((res - mu) / sigma) ** 4).mean(axis=(1, 2)) - 3.0  # Excess kurtosis

        energies.append(e)
        variances.append(v)
        spectral.append(s)
        kurtoses.append(kurt)

    return (
        np.stack(energies),
        np.stack(variances),
        np.stack(spectral),
        np.stack(kurtoses),
    )


def compute_radial_profile(image_paths, srm_conv, size=256, n_bins=64):
    """
    Compute the azimuthally-averaged radial power spectrum of SRM residuals.
    Useful to see if StyleGAN leaves periodic artefacts at specific frequencies.

    Returns:
        profiles: dict {class_name: (n_bins,)} averaged radial power
    """
    H, W = size, size
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    R = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2).astype(int)
    max_r = min(cy, cx)

    bin_edges = np.linspace(0, max_r, n_bins + 1).astype(int)
    profiles = []

    for path in tqdm(image_paths, desc="Radial spectrum", leave=False):
        img_t = load_image_as_tensor(path, size)
        with torch.no_grad():
            residuals = srm_conv(img_t)
        # Average over all 30 filters
        res_mean = residuals.squeeze(0).mean(0).numpy()  # (H, W)
        fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(res_mean))) ** 2

        profile = np.zeros(n_bins)
        for b in range(n_bins):
            mask = (R >= bin_edges[b]) & (R < bin_edges[b + 1])
            if mask.sum() > 0:
                profile[b] = fft_mag[mask].mean()
        profiles.append(profile)

    return np.stack(profiles).mean(axis=0)  # (n_bins,)


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_filter_kernels(save_dir):
    """Visualise the 30 SRM kernels."""
    kernels = build_srm_kernels()
    fig, axes = plt.subplots(5, 6, figsize=(18, 15))
    for i, ax in enumerate(axes.flat):
        if i < len(kernels):
            im = ax.imshow(kernels[i], cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
            ax.set_title(SRM_FILTER_NAMES[i] if i < len(SRM_FILTER_NAMES) else f"F{i+1}", fontsize=9)
        ax.axis('off')
    fig.suptitle('30 SRM High-Pass Filter Kernels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '01_srm_kernels.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 01_srm_kernels.png")


def plot_residual_examples(data_dict, srm_conv, save_dir, size=256):
    """Show example residual maps for one image per class."""
    selected_filters = [0, 4, 11, 15, 21, 22]  # Interesting subset
    n_classes = len(data_dict)
    n_filters = len(selected_filters)

    fig, axes = plt.subplots(n_classes, n_filters + 1, figsize=(3 * (n_filters + 1), 3 * n_classes))
    if n_classes == 1:
        axes = axes[np.newaxis, :]

    for row, (cls_name, paths) in enumerate(data_dict.items()):
        if not paths:
            continue
        # Load first image
        img = Image.open(paths[0]).convert('RGB').resize((size, size))
        img_arr = np.array(img)

        # Original
        axes[row, 0].imshow(img_arr)
        axes[row, 0].set_title(f'{cls_name}\n(original)', fontsize=9)
        axes[row, 0].axis('off')

        # SRM residuals
        img_t = load_image_as_tensor(paths[0], size)
        with torch.no_grad():
            residuals = srm_conv(img_t).squeeze(0).numpy()

        for col, fi in enumerate(selected_filters):
            vmax = np.percentile(np.abs(residuals[fi]), 99)
            axes[row, col + 1].imshow(residuals[fi], cmap='RdBu_r',
                                       vmin=-vmax, vmax=vmax, interpolation='nearest')
            axes[row, col + 1].set_title(SRM_FILTER_NAMES[fi], fontsize=9)
            axes[row, col + 1].axis('off')

    fig.suptitle('SRM Residual Maps: Real vs Fake', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '02_residual_examples.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 02_residual_examples.png")


def plot_energy_distributions(stats_dict, save_dir):
    """Box/violin plots of per-filter energy for each class."""
    # Pick most discriminative filters: Laplacian, LoG, Checker, SobelH
    selected = [4, 11, 15, 21, 22, 25]
    n_sel = len(selected)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for idx, (ax, fi) in enumerate(zip(axes.flat, selected)):
        plot_data = []
        for cls_name, (energies, _, _, _) in stats_dict.items():
            for e in energies[:, fi]:
                plot_data.append({'Class': cls_name, 'Energy': e})

        import pandas as pd
        df = pd.DataFrame(plot_data)
        sns.violinplot(data=df, x='Class', y='Energy', ax=ax, cut=0, inner='quartile')
        ax.set_title(f'Filter: {SRM_FILTER_NAMES[fi]}', fontsize=11)
        ax.tick_params(axis='x', rotation=30)

    fig.suptitle('SRM Filter Energy Distribution by Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '03_energy_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 03_energy_distributions.png")


def plot_mean_energy_heatmap(stats_dict, save_dir):
    """Heatmap: classes × filters, showing mean energy per filter."""
    import pandas as pd

    class_names = list(stats_dict.keys())
    n_filters = 30
    matrix = np.zeros((len(class_names), n_filters))

    for i, (cls_name, (energies, _, _, _)) in enumerate(stats_dict.items()):
        matrix[i] = energies.mean(axis=0)

    df = pd.DataFrame(matrix, index=class_names,
                      columns=[SRM_FILTER_NAMES[j] if j < len(SRM_FILTER_NAMES) else f"F{j+1}" for j in range(n_filters)])

    fig, ax = plt.subplots(figsize=(20, 6))
    sns.heatmap(df, annot=False, fmt='.3f', cmap='YlOrRd', ax=ax, linewidths=0.3)
    ax.set_title('Mean SRM Filter Energy per Class', fontsize=14, fontweight='bold')
    ax.set_ylabel('Class')
    ax.set_xlabel('SRM Filter')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '04_energy_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 04_energy_heatmap.png")


def plot_radial_spectra(radial_dict, save_dir):
    """Compare radial power spectra of SRM residuals across classes."""
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab10
    for i, (cls_name, profile) in enumerate(radial_dict.items()):
        ax.plot(profile, label=cls_name, color=cmap(i), linewidth=2 if 'Real' in cls_name or 'FFHQ' in cls_name else 1.2)

    ax.set_xlabel('Radial Frequency Bin')
    ax.set_ylabel('Mean Spectral Power')
    ax.set_title('Radial Power Spectrum of SRM Residuals', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_yscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '05_radial_spectra.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 05_radial_spectra.png")


def plot_tsne_pca(stats_dict, save_dir):
    """t-SNE and PCA on concatenated SRM statistics."""
    features_all = []
    labels_all = []
    class_names = list(stats_dict.keys())

    for cls_name, (energies, variances, spectral, kurtoses) in stats_dict.items():
        # Concatenate all stats: (N, 30*4 = 120) feature vector
        feat = np.concatenate([energies, variances, spectral, kurtoses], axis=1)
        features_all.append(feat)
        labels_all.extend([cls_name] * len(feat))

    X = np.vstack(features_all)
    labels = np.array(labels_all)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── PCA ──
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # ── t-SNE ──
    tsne = TSNE(n_components=2, perplexity=min(30, len(X) // 4), random_state=42, max_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    cmap = plt.cm.tab10
    for i, cls_name in enumerate(class_names):
        mask = labels == cls_name
        marker = 'o' if 'Real' in cls_name or 'ffhq' in cls_name.lower() else 's'
        size = 60 if 'Real' in cls_name or 'ffhq' in cls_name.lower() else 30
        ax1.scatter(X_pca[mask, 0], X_pca[mask, 1], label=cls_name, c=[cmap(i)],
                    alpha=0.7, s=size, marker=marker, edgecolors='k', linewidths=0.3)
        ax2.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=cls_name, c=[cmap(i)],
                    alpha=0.7, s=size, marker=marker, edgecolors='k', linewidths=0.3)

    ax1.set_title(f'PCA (var explained: {pca.explained_variance_ratio_.sum():.1%})', fontsize=12)
    ax1.set_xlabel('PC1'); ax1.set_ylabel('PC2')
    ax1.legend(fontsize=8, loc='best')

    ax2.set_title('t-SNE', fontsize=12)
    ax2.set_xlabel('Dim 1'); ax2.set_ylabel('Dim 2')
    ax2.legend(fontsize=8, loc='best')

    fig.suptitle('SRM Residual Statistics — Dimensionality Reduction', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '06_tsne_pca.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 06_tsne_pca.png")


def linear_separability_test(stats_dict, save_dir):
    """
    Train a Logistic Regression (Real vs each class) on SRM stats
    to quantify how separable they are in this space.
    """
    real_key = [k for k in stats_dict if 'ffhq' in k.lower() or 'Real' in k]
    if not real_key:
        print("  [WARN] No 'Real/FFHQ' class found, skipping separability test.")
        return
    real_key = real_key[0]
    real_energies, real_var, real_spec, real_kurt = stats_dict[real_key]
    X_real = np.concatenate([real_energies, real_var, real_spec, real_kurt], axis=1)

    results = {}
    for cls_name, (energies, variances, spectral, kurtoses) in stats_dict.items():
        if cls_name == real_key:
            continue
        X_fake = np.concatenate([energies, variances, spectral, kurtoses], axis=1)

        X = np.vstack([X_real, X_fake])
        y = np.concatenate([np.zeros(len(X_real)), np.ones(len(X_fake))])

        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        n_splits = min(5, min(len(X_real), len(X_fake)))
        if n_splits < 2:
            results[cls_name] = float('nan')
            continue

        clf = LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs')
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = cross_val_score(clf, X_s, y, cv=cv, scoring='accuracy')
        results[cls_name] = scores.mean()

    # Print and save
    print("\n" + "=" * 55)
    print("  Linear Separability (Logistic Regression, 5-fold CV)")
    print("  Real baseline: " + real_key)
    print("=" * 55)
    for cls_name, acc in sorted(results.items(), key=lambda x: x[1]):
        bar = "█" * int(acc * 40)
        print(f"  {cls_name:<35s}  {acc:.4f}  {bar}")
    print("=" * 55)

    # Save as bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(results.keys())
    accs = [results[n] for n in names]
    colors = ['#e74c3c' if a < 0.6 else '#f39c12' if a < 0.75 else '#2ecc71' for a in accs]
    bars = ax.barh(names, accs, color=colors, edgecolor='k', linewidth=0.5)
    ax.set_xlim(0.4, 1.05)
    ax.axvline(0.5, color='grey', linestyle='--', linewidth=0.8, label='chance')
    ax.set_xlabel('Accuracy (5-fold CV)')
    ax.set_title('SRM-based Linear Separability: Real vs each Fake class', fontsize=13, fontweight='bold')
    ax.legend()
    for bar, acc in zip(bars, accs):
        ax.text(acc + 0.01, bar.get_y() + bar.get_height() / 2, f'{acc:.3f}',
                va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '07_linear_separability.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: 07_linear_separability.png")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='SRM Filter Analysis for MuFlow')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of images to sample per class')
    parser.add_argument('--size', type=int, default=256,
                        help='Image resize dimension')
    parser.add_argument('--save_dir', type=str, default='scripts/.pictures/srm',
                        help='Directory to save analysis plots')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # ── Data sources ─────────────────────────────────────────────────────
    DATA_DIR = const.DATA_DIR

    data_sources = {
        'Real (FFHQ)':     f'{DATA_DIR}/ffhq/**/*.png',
        'StyleGAN':         f'{DATA_DIR}/WILD/Open Set/StyleGAN/*.png',
        'StyleGAN2':        f'{DATA_DIR}/WILD/Open Set/StyleGAN2/*.png',
        'StyleGAN3':        f'{DATA_DIR}/WILD/Open Set/StyleGAN3/*.png',
        'STARGAN':          f'{DATA_DIR}/WILD/Open Set/STARGAN/*.png',
        'AttGAN':           f'{DATA_DIR}/WILD/Open Set/AttGAN/*.png',
        'Stable Diff XL':   f'{DATA_DIR}/WILD/Closed Set/Stable Diffusion XL/*.png',
        'Flux.1':           f'{DATA_DIR}/WILD/Closed Set/Flux.1/*.png',
        'Midjourney':       f'{DATA_DIR}/WILD/Closed Set/Midjourney/*.png',
        'Dall-E 3':         f'{DATA_DIR}/WILD/Closed Set/Dall-E 3/*.png',
    }

    # Collect images
    print("Collecting images...")
    data_dict = {}
    for cls_name, pattern in data_sources.items():
        paths = collect_images(pattern, args.n_samples, args.seed)
        data_dict[cls_name] = paths
        print(f"  {cls_name}: {len(paths)} images")

    # Remove empty classes
    data_dict = {k: v for k, v in data_dict.items() if v}
    if not data_dict:
        print("ERROR: No images found. Check data paths in constants.py.")
        return

    # ── Build SRM conv ───────────────────────────────────────────────────
    srm_conv = SRMConv(in_channels=1)
    srm_conv.eval()

    # ── Analysis 1: Visualise kernels ────────────────────────────────────
    print("\n[1/6] Plotting SRM filter kernels...")
    plot_filter_kernels(args.save_dir)

    # ── Analysis 2: Residual map examples ────────────────────────────────
    print("[2/6] Plotting residual map examples...")
    plot_residual_examples(data_dict, srm_conv, args.save_dir, args.size)

    # ── Analysis 3: Compute statistics ───────────────────────────────────
    print("[3/6] Computing SRM residual statistics per class...")
    stats_dict = {}
    for cls_name, paths in data_dict.items():
        print(f"  Processing: {cls_name}")
        energies, variances, spectral, kurtoses = compute_residual_stats(paths, srm_conv, args.size)
        stats_dict[cls_name] = (energies, variances, spectral, kurtoses)

    # ── Analysis 4: Energy distributions ─────────────────────────────────
    print("[4/6] Plotting energy distributions...")
    plot_energy_distributions(stats_dict, args.save_dir)
    plot_mean_energy_heatmap(stats_dict, args.save_dir)

    # ── Analysis 5: Radial power spectra ─────────────────────────────────
    print("[5/6] Computing radial power spectra...")
    radial_dict = {}
    for cls_name, paths in data_dict.items():
        radial_dict[cls_name] = compute_radial_profile(paths, srm_conv, args.size)
    plot_radial_spectra(radial_dict, args.save_dir)

    # ── Analysis 6: Dimensionality reduction + separability ──────────────
    print("[6/6] Running PCA, t-SNE, and linear separability test...")
    plot_tsne_pca(stats_dict, args.save_dir)
    linear_separability_test(stats_dict, args.save_dir)

    print(f"\n✅ All analysis saved to: {args.save_dir}/")


if __name__ == '__main__':
    main()
