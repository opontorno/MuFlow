import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from muflow import constants as const


def _center_square(img, P):
    """Center-crop an image to its short side, resizing to P only if needed."""
    W, H = img.size
    s = min(W, H)
    left, top = (W - s) // 2, (H - s) // 2
    crop = img.crop((left, top, left + s, top + s))
    if s != P:
        crop = crop.resize((P, P), Image.BILINEAR)
    return crop


def random_patches(img, P, k):
    """Crop k random native P×P patches (torch RNG positions)."""
    W, H = img.size
    if W < P or H < P:
        return [_center_square(img, P)] * k
    out = []
    for _ in range(k):
        left = int(torch.randint(0, W - P + 1, (1,)).item())
        top  = int(torch.randint(0, H - P + 1, (1,)).item())
        out.append(img.crop((left, top, left + P, top + P)))
    return out


def repr_patches(img, P, k, seed):
    """Crop k deterministic native P×P patches (fixed-seed positions)."""
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


def make_patch_transform(norm_mean=None, norm_std=None):
    """Build the patch tensor transform (ToTensor + Normalize, no resize)."""
    mean = norm_mean if norm_mean is not None else const.IMAGENET_MEAN
    std  = norm_std  if norm_std  is not None else const.IMAGENET_STD
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def pool_features(feats, pooling_type):
    """Spatially pool a feature map into a 1-D vector."""
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
    else:
        return feats.flatten(1).cpu().numpy().squeeze(0)


def extract_features(patch, backbone, backbone_type, out_indices, input_size,
                     norm_mean, norm_std, device):
    """Extract per-layer backbone features from a single patch."""
    if isinstance(patch, np.ndarray):
        patch = Image.fromarray(patch.astype(np.uint8))

    mean = torch.tensor(norm_mean).view(1, 3, 1, 1).to(device)
    std  = torch.tensor(norm_std).view(1, 3, 1, 1).to(device)
    img_t = (torch.from_numpy(np.array(patch)).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0 - mean) / std

    with torch.no_grad():
        return list(backbone(img_t))


def image_centroid(img, P, k, seed, backbone, backbone_type, out_indices,
                   input_size, pooling_type, norm_mean, norm_std, device):
    """Patch-centroid representation of one image, per backbone layer."""
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
