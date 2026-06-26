"""
patch_utils.py — Shared patch-extraction and feature-computation utilities.

Central module for all patch-based operations in µFlow:

  Patch extraction
  ─────────────────
  _center_square      fallback for images smaller than the target P
  random_patches      k random native crops (training)
  repr_patches        k deterministic crops (fixed seed; inference & GMM)

  Transform
  ─────────
  make_patch_transform  ToTensor + backbone-specific Normalize (no resize)

  Feature pooling
  ───────────────
  pool_features       (B, C, H, W) tensor → 1-D numpy vector

  Feature extraction
  ──────────────────
  extract_features    PIL patch → list of (1, C, H, W) tensors, one per layer
                      (backbone-agnostic; handles CNN / DINOv2 / CLIP / CaiT-DeiT)

  Centroid representation
  ───────────────────────
  image_centroid      PIL image → list of 1-D numpy vectors (one per layer)
                      = mean of pooled features over repr_patches
                      used by generate_parameters and analyze_means
"""
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from muflow import constants as const


# ════════════════════════════════════════════════════════════════════════════
# Patch extraction
# ════════════════════════════════════════════════════════════════════════════

def _center_square(img: Image.Image, P: int) -> Image.Image:
    """Center-crop to the short side; resize up to P only if unavoidable."""
    W, H = img.size
    s = min(W, H)
    left, top = (W - s) // 2, (H - s) // 2
    crop = img.crop((left, top, left + s, top + s))
    if s != P:
        crop = crop.resize((P, P), Image.BILINEAR)
    return crop


def random_patches(img: Image.Image, P: int, k: int) -> list:
    """k random native P×P crops (positions from torch RNG → per-worker/epoch seeded)."""
    W, H = img.size
    if W < P or H < P:
        return [_center_square(img, P)] * k
    out = []
    for _ in range(k):
        left = int(torch.randint(0, W - P + 1, (1,)).item())
        top  = int(torch.randint(0, H - P + 1, (1,)).item())
        out.append(img.crop((left, top, left + P, top + P)))
    return out


def repr_patches(img: Image.Image, P: int, k: int, seed: int) -> list:
    """k deterministic native P×P crops (fixed seed → reproducible representation)."""
    W, H = img.size
    if W < P or H < P:
        return [_center_square(img, P)] * k
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(k):
        left = int(rng.integers(0, W - P + 1))
        top  = int(rng.integers(0, H - P + 1))
        out.append(img.crop((left, top, left + P, top + P)))
    return out


# ════════════════════════════════════════════════════════════════════════════
# Transform
# ════════════════════════════════════════════════════════════════════════════

def make_patch_transform(norm_mean=None, norm_std=None):
    """ToTensor + backbone-specific Normalize. No resize — patches are already P×P."""
    mean = norm_mean if norm_mean is not None else const.IMAGENET_MEAN
    std  = norm_std  if norm_std  is not None else const.IMAGENET_STD
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ════════════════════════════════════════════════════════════════════════════
# Feature pooling
# ════════════════════════════════════════════════════════════════════════════

def pool_features(feats: torch.Tensor, pooling_type: str) -> np.ndarray:
    """Spatial pooling of a (B, C, H, W) feature map → 1-D numpy vector."""
    if pooling_type == 'mean':
        return feats.mean(dim=(2, 3)).flatten().cpu().numpy()
    elif pooling_type == 'max':
        return feats.flatten(2).max(-1)[0].cpu().numpy().squeeze(0)
    elif pooling_type == 'mean_std':
        m = feats.mean(dim=(2, 3)).flatten().cpu().numpy()
        s = feats.std(dim=(2, 3)).flatten().cpu().numpy()
        return np.concatenate([m, s])
    elif pooling_type == 'flatten':
        return feats.mean(-1).flatten(1).cpu().numpy().squeeze(0)
    else:  # full_flatten
        return feats.flatten(1).cpu().numpy().squeeze(0)


# ════════════════════════════════════════════════════════════════════════════
# Feature extraction  (backbone-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def extract_features(
    patch,
    backbone,
    backbone_type: str,
    out_indices,
    input_size: int,
    norm_mean,
    norm_std,
    device,
) -> list:
    """
    Extract backbone features from a single native patch.

    Args:
        patch         : PIL Image (P×P) or np.ndarray (P×P×3 uint8)
        backbone      : frozen backbone model (eval mode)
        backbone_type : 'cnn' | 'dino' | 'clip' | 'cait_deit'
        out_indices   : layer indices (used for DINO forward)
        input_size    : backbone input side P (used for DeiT/CaiT token reshape)
        norm_mean, norm_std : backbone-specific normalisation stats
        device        : torch.device

    Returns:
        list of (1, C, H, W) float tensors — one per requested layer
    """
    import timm.models.vision_transformer as _vit

    if isinstance(patch, np.ndarray):
        patch = Image.fromarray(patch.astype(np.uint8))

    mean = torch.tensor(norm_mean).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(norm_std).view(1, 3, 1, 1).to(device)
    img_t = (torch.from_numpy(np.array(patch)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0 - mean) / std

    with torch.no_grad():
        if backbone_type == 'cait_deit':
            if isinstance(backbone, _vit.VisionTransformer):  # DeiT
                x = backbone.patch_embed(img_t)
                cls = backbone.cls_token.expand(x.shape[0], -1, -1)
                if backbone.dist_token is None:
                    x = torch.cat((cls, x), dim=1)
                else:
                    x = torch.cat((cls, backbone.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = backbone.pos_drop(x + backbone.pos_embed)
                for i in range(8):
                    x = backbone.blocks[i](x)
                x = backbone.norm(x)[:, 2:, :]
                N, _, C = x.shape
                return [x.permute(0, 2, 1).reshape(N, C, input_size // 16, input_size // 16)]
            else:  # CaiT
                x = backbone.patch_embed(img_t) + backbone.pos_embed
                x = backbone.pos_drop(x)
                for i in range(41):
                    x = backbone.blocks[i](x)
                x = backbone.norm(x)
                N, _, C = x.shape
                return [x.permute(0, 2, 1).reshape(N, C, input_size // 16, input_size // 16)]

        elif backbone_type == 'dino':
            return list(backbone.get_intermediate_layers(img_t, n=list(out_indices), reshape=True))

        else:  # cnn / clip
            return list(backbone(img_t))


# ════════════════════════════════════════════════════════════════════════════
# Centroid representation
# ════════════════════════════════════════════════════════════════════════════

def image_centroid(
    img: Image.Image,
    P: int,
    k: int,
    seed: int,
    backbone,
    backbone_type: str,
    out_indices,
    input_size: int,
    pooling_type: str,
    norm_mean,
    norm_std,
    device,
) -> list:
    """
    Patch-centroid representation of one image, per backbone layer.

    Crops k deterministic P×P patches (seed-fixed), extracts and pools
    features per patch, then averages across patches → one centroid vector
    per layer. Used by generate_parameters (GMM fitting) and analyze_means.

    Returns:
        list of 1-D np.ndarray, one per layer (len = len(out_indices))
    """
    plist = repr_patches(img, P, k, seed)
    per_layer = None
    for patch in plist:
        feats  = extract_features(patch, backbone, backbone_type, out_indices,
                                  input_size, norm_mean, norm_std, device)
        pooled = [pool_features(f, pooling_type) for f in feats]
        if per_layer is None:
            per_layer = [[] for _ in pooled]
        for li, p in enumerate(pooled):
            per_layer[li].append(p)
    return [np.mean(np.stack(layer), axis=0) for layer in per_layer]
