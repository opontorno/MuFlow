"""
patches.py — native-patch cropping for the patch-based µFlow.

An image is represented through square patches of side P = config["input_size"],
cropped at NATIVE resolution (no resize → forensic traces preserved). Two modes:

  • random_patches  — k random crops (training; torch RNG, reseeded per worker/epoch)
  • repr_patches    — k deterministic crops (fixed seed → reproducible representation
                      used at inference and for the mean-image GMM centroids)

If an image is smaller than P (rare: only sub-P sources), it is center-cropped to
its short side and, as a last resort, resized up to P (logged once).
"""
import numpy as np
import torch
from PIL import Image

def _center_square(img, P):
    """Center-crop to the short side; resize up to P only if unavoidable."""
    W, H = img.size
    s = min(W, H)
    left, top = (W - s) // 2, (H - s) // 2
    crop = img.crop((left, top, left + s, top + s))
    if s != P:
        crop = crop.resize((P, P), Image.BILINEAR)
    return crop


def random_patches(img, P, k):
    """k random native P×P crops (positions from torch RNG → per-worker/epoch seeded)."""
    W, H = img.size
    if W < P or H < P:
        return [_center_square(img, P)] * k
    out = []
    for _ in range(k):
        left = int(torch.randint(0, W - P + 1, (1,)).item())
        top = int(torch.randint(0, H - P + 1, (1,)).item())
        out.append(img.crop((left, top, left + P, top + P)))
    return out


def repr_patches(img, P, k, seed):
    """k deterministic native P×P crops (fixed seed → reproducible representation)."""
    W, H = img.size
    if W < P or H < P:
        return [_center_square(img, P)] * k
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        left = int(rng.integers(0, W - P + 1))
        top = int(rng.integers(0, H - P + 1))
        out.append(img.crop((left, top, left + P, top + P)))
    return out
