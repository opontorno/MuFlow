"""
SRM (Spatial Rich Model) Utilities for MuFlow Project
=======================================================

This module provides a unified implementation of SRM-based preprocessing
used across the entire project (analysis, training, parameter generation).

The transformation converts RGB images to SRM residual maps:
    RGB → SRM convolution (30 kernels) → abs per filter → sample 3 random filters
        → log1p → min-max [0,1] per channel → (H, W, 3)

This mirrors the Fourier preprocessing pipeline in fourier_utils.py and fulfils
the same role: provide a noise-level representation of the image that is
discriminative between real and generated content.

Reference:
    Fridrich, J. & Kodovský, J. (2012). Rich Models for Steganalysis of
    Digital Images. IEEE TIFS, 7(3), 868–882.
"""

import numpy as np
from PIL import Image

try:
    from scipy.ndimage import convolve as nd_convolve
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


# ===========================================================================
# Kernel construction
# ===========================================================================

def build_srm_kernels():
    """
    Build 30 SRM (Spatial Rich Model) high-pass filter kernels from the
    steganalysis literature (Fridrich & Kodovský, 2012).

    Returns
    -------
    np.ndarray  shape (30, 5, 5)  dtype float32
    """
    filters = []

    # ── 1st-order edge detectors ─────────────────────────────────────────
    for (r1, c1, r2, c2) in [(2, 1, 2, 2), (1, 2, 2, 2),
                               (1, 1, 2, 2), (1, 3, 2, 2)]:
        f = np.zeros((5, 5), dtype=np.float32)
        f[r1, c1], f[r2, c2] = -1, 1
        filters.append(f)

    # ── 2nd-order Laplacian variants ─────────────────────────────────────
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

    # Pad / trim to exactly 30
    filters = filters[:30]
    while len(filters) < 30:
        filters.append(np.rot90(filters[len(filters) % len(filters)]).copy())

    return np.stack(filters, axis=0)  # (30, 5, 5)


# Precompute kernels once at module load  (shared across all calls)
_SRM_KERNELS: np.ndarray = build_srm_kernels()   # (30, 5, 5)


# ===========================================================================
# Core preprocessing function
# ===========================================================================

def calculate_srm_residuals(image, seed=None):
    """
    Convert an RGB image to an SRM residual map suitable for CNN input.

    Processing pipeline:
        1. Convert PIL Image → (H, W, 3) uint8 numpy if needed.
        2. Normalise pixel values to [0, 1].
        3. Apply all 30 SRM high-pass kernels (mean over RGB channels)
           → (30, H, W) residual maps.
        4. Take absolute value per filter map.
        5. **Randomly sample 3 filter indices** (no averaging) → 3 channels.
           In training this acts as stochastic augmentation; pass a fixed
           *seed* (int) for deterministic/eval behaviour.
        6. Apply log1p compression (×20) independently per channel.
        7. Min-max normalise each channel to [0, 1] independently.
        → output (H, W, 3) float32 — each channel is one complete filter
          response, carrying full spatial information.

    Args:
        image : np.ndarray (H, W, 3) uint8, or PIL Image.
        seed  : int or None.  None → new random selection each call
                (augmentation).  Int → reproducible selection (eval).

    Returns:
        np.ndarray  (H, W, 3)  float32  [0, 1]
    """
    if isinstance(image, Image.Image):
        image = np.array(image.convert("RGB"))

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H, W, 3), got shape {image.shape}")

    img_f = image.astype(np.float32) / 255.0   # (H, W, 3)

    # ── Step 1: compute all 30 filter responses (mean over RGB channels) ──
    H, W = img_f.shape[:2]
    filter_maps = np.zeros((30, H, W), dtype=np.float32)  # (30, H, W)

    if _HAS_SCIPY:
        for k in range(30):
            kernel = _SRM_KERNELS[k]                            # (5, 5)
            acc = np.zeros((H, W), dtype=np.float32)
            for c in range(3):
                acc += nd_convolve(img_f[:, :, c], kernel, mode="reflect")
            filter_maps[k] = acc / 3.0
    else:
        pad = 2
        for k in range(30):
            k_flip = _SRM_KERNELS[k, ::-1, ::-1]
            acc = np.zeros((H, W), dtype=np.float32)
            for c in range(3):
                ch   = np.pad(img_f[:, :, c], pad, mode="reflect")
                view = np.lib.stride_tricks.sliding_window_view(ch, (5, 5))
                acc += (view * k_flip).sum(axis=(-2, -1))
            filter_maps[k] = acc / 3.0

    # ── Step 2: abs per filter ────────────────────────────────────────────
    filter_maps = np.abs(filter_maps)

    # ── Step 3: sample 3 random filters → 3 independent channels ─────────
    rng     = np.random.default_rng(seed)          # None → unpredictable each call
    indices = rng.choice(30, size=3, replace=False) # 3 distinct filter indices

    channels = []
    for idx in indices:
        ch = filter_maps[idx]                       # (H, W) — no averaging
        ch = np.log1p(ch * 20.0)
        c_min, c_max = ch.min(), ch.max()
        ch = (ch - c_min) / (c_max - c_min + 1e-8)
        channels.append(ch)

    residual_rgb = np.stack(channels, axis=-1)              # (H, W, 3)
    return residual_rgb.astype(np.float32)


# ===========================================================================
# torchvision-compatible transform
# ===========================================================================

class SRMResidualTransform:
    """
    torchvision-compatible transform that converts an image to its SRM
    residual map.

    Drop-in replacement for ``FourierMagnitudeTransform``.  Use together
    with ``ToTensorNoScale`` (from fourier_utils) in a
    ``transforms.Compose`` pipeline::

        from muflow.srm_utils import SRMResidualTransform
        from muflow.fourier_utils import ToTensorNoScale

        # Training  (stochastic — different 3 filters per image per epoch)
        transform = transforms.Compose([
            transforms.Resize(256),
            SRMResidualTransform(),
            ToTensorNoScale(),
        ])

        # Eval  (deterministic — same 3 filters every time)
        transform = transforms.Compose([
            transforms.Resize(256),
            SRMResidualTransform(seed=0),
            ToTensorNoScale(),
        ])
    """

    def __init__(self, seed=None):
        """
        Args:
            seed : int or None.
                   None  → random filter selection per call (training augmentation).
                   Int   → fixed selection every call (eval reproducibility).
        """
        self.seed = seed

    def __call__(self, img):
        """
        Args:
            img : PIL Image or np.ndarray (H, W, 3) uint8.
        Returns:
            np.ndarray (H, W, 3) float32 in [0, 1].
        """
        return calculate_srm_residuals(img, seed=self.seed)

    def __repr__(self):
        return f"{self.__class__.__name__}(seed={self.seed})"
