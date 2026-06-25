import torch
import os
import numpy as np
from glob import glob
from PIL import Image
import timm
import yaml
import argparse
from sklearn.mixture import GaussianMixture
from muflow import constants as const
from muflow.patches import repr_patches
from tqdm import tqdm

PREFIX = ""
MEANS_DIR = os.path.join(const.DATA_DIR, "datasets_means", "500")

# === Argument Parser ===
parser = argparse.ArgumentParser(description='Generate GMM parameters for FastFlow')
parser.add_argument('-model', '--model_name', type=str, default='resnet50', help='Backbone model name')
parser.add_argument('-reals', '--reals', type=str, default='ffhq', choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'], help='Real images dataset')
args = parser.parse_args()

# === Hyperparameters ===
model_name  = args.model_name
reals       = args.reals

config_path = f"{const.WORKING_DIR}/configs/{model_name}.yaml"
config = yaml.safe_load(open(config_path, "r"))
print("Model config: ", config)
print(f"Reals dataset: {reals}")

# === Model Setup ===
out_indices  = config.get("out_indices", [1, 2, 3])
pooling_type = config.get("pooling_type", "mean")
n_components = config.get("gmm_n_components", 1)
print(f"Using out_indices: {out_indices}")
print(f"Using pooling_type: {pooling_type}")
print(f"Number of GMM components: {n_components}")

# ── Build feature extractor ───────────────────────────────────────────────────
if model_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
    model = timm.create_model(config['backbone_name'], pretrained=True, in_chans=3)
    channels   = [768]
    scales     = [16]
    backbone_type = 'cait_deit'
elif model_name in const.DINO_BACKBONES:
    timm_name  = const.DINO_TIMM_NAMES[model_name]
    model      = timm.create_model(timm_name, pretrained=True,
                                   img_size=config['input_size'])
    ch         = const.DINO_CHANNELS[model_name]
    ps         = const.DINO_PATCH_SIZE[model_name]
    channels   = [ch] * len(out_indices)
    scales     = [ps] * len(out_indices)
    backbone_type = 'dino'
elif model_name in const.CLIP_BACKBONES:
    from muflow.model import CLIPVisualExtractor
    model      = CLIPVisualExtractor(model_name, out_block_indices=out_indices)
    ch         = const.CLIP_CHANNELS[model_name]
    ps         = const.CLIP_PATCH_SIZE[model_name]
    channels   = [ch] * len(out_indices)
    scales     = [ps] * len(out_indices)
    backbone_type = 'clip'
else:
    model = timm.create_model(config['backbone_name'], pretrained=True,
                               features_only=True, in_chans=3, out_indices=out_indices)
    channels   = model.feature_info.channels()
    scales     = model.feature_info.reduction()
    backbone_type = 'cnn'

model.eval()

print("Channels: ", channels)
print("Scales:   ", scales)
num_layers = len(out_indices)
print(f"Number of layers: {num_layers}")

# Normalisation stats matching the backbone (CLIP stats for CLIP, else ImageNet).
# Must mirror muflow.constants.get_norm_stats so the GMM is fit on the same
# feature distribution the model sees at train/test time.
NORM_MEAN, NORM_STD = const.get_norm_stats(model_name)


def _normalize(img_t):
    """Apply backbone-specific normalisation to a (1,3,H,W) float tensor in [0,255]."""
    img_t = img_t / 255.0
    mean  = torch.tensor(NORM_MEAN).view(1, 3, 1, 1)
    std   = torch.tensor(NORM_STD).view(1, 3, 1, 1)
    return (img_t - mean) / std

# === Get features ===
def get_features(img, apply_normalization=False):
    """
    Extract features from an image array.

    Args:
        img: numpy array (RGB uint8)
        apply_normalization: if True normalise with backbone stats (RGB only)
    Returns:
        list of feature tensors (one per layer / block)
    """
    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()

    if apply_normalization:
        img_t = _normalize(img_t)

    with torch.no_grad():
        if backbone_type == 'cait_deit':
            if isinstance(model, timm.models.vision_transformer.VisionTransformer):
                x = model.patch_embed(img_t)
                cls_token = model.cls_token.expand(x.shape[0], -1, -1)
                if model.dist_token is None:
                    x = torch.cat((cls_token, x), dim=1)
                else:
                    x = torch.cat((cls_token, model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
                x = model.pos_drop(x + model.pos_embed)
                for i in range(8):
                    x = model.blocks[i](x)
                x = model.norm(x)
                x = x[:, 2:, :]
                N, _, C = x.shape
                x = x.permute(0, 2, 1)
                x = x.reshape(N, C, config['input_size'] // 16, config['input_size'] // 16)
                return [x]
            else:  # CaiT
                x = model.patch_embed(img_t)
                x = x + model.pos_embed
                x = model.pos_drop(x)
                for i in range(41):
                    x = model.blocks[i](x)
                N, _, C = x.shape
                x = model.norm(x)
                x = x.permute(0, 2, 1)
                x = x.reshape(N, C, config['input_size'] // 16, config['input_size'] // 16)
                return [x]
        elif backbone_type == 'dino':
            # get_intermediate_layers returns list of (B,C,H,W) with reshape=True
            return model.get_intermediate_layers(
                img_t, n=list(out_indices), reshape=True
            )
        elif backbone_type == 'clip':
            # CLIPVisualExtractor expects already-normalised input
            # (CLIP stats applied above via _normalize)
            return model(img_t)
        else:  # cnn
            return model(img_t)


def _pool(feats, pooling_type):
    """Spatial pooling of a (B, C, H, W) feature map → 1-D vector (matches model.py)."""
    if pooling_type == 'mean':
        return feats.mean([2, 3]).flatten().cpu().numpy()
    elif pooling_type == 'max':
        return feats.flatten(2).max(-1)[0].cpu().numpy().squeeze(0)
    elif pooling_type == 'mean_std':
        m = feats.mean([2, 3]).flatten().cpu().numpy()
        s = feats.std([2, 3]).flatten().cpu().numpy()
        return np.concatenate([m, s])
    elif pooling_type == 'flatten':
        return feats.mean(-1).flatten(1).cpu().numpy().squeeze(0)
    else:  # full_flatten
        return feats.flatten(1).cpu().numpy().squeeze(0)


def mean_image_centroid(path):
    """Patch-centroid representation of one MEAN image (per layer).

    Crops PATCH_NUM_REPR native patches (fixed seed, no resize), extracts and
    pools features per patch, then averages over patches → the image centroid.
    The GMM target is the distribution of these centroids.
    """
    img = Image.open(path).convert("RGB")
    plist = repr_patches(img, config['input_size'], const.PATCH_NUM_REPR, const.PATCH_SEED)
    per_layer = None
    for patch in plist:
        feats = get_features(np.array(patch), apply_normalization=True)
        pooled = [_pool(f, pooling_type) for f in feats]
        if per_layer is None:
            per_layer = [[] for _ in pooled]
        for li, p in enumerate(pooled):
            per_layer[li].append(p)
    return [np.mean(np.stack(layer), axis=0) for layer in per_layer]


MEAN_SIZE = os.path.basename(MEANS_DIR.rstrip("/"))  # e.g. "500"

_HOWTO_MEANS = (
    "\nThe average images are produced by scripts/generate_means.py. Run, e.g.:\n"
    f"    python scripts/generate_means.py --mean_size {MEAN_SIZE}\n"
    "(use --num_images N to set how many means per source, --data_root to point "
    "at a different dataset root). This populates the directory above with one "
    "subfolder per source (ffhq/, celeba_hq/, and each generator)."
)

# ── Check that average images have been computed ──────────────────────────────
if not os.path.isdir(MEANS_DIR):
    raise FileNotFoundError(
        f"Average-images directory not found: {MEANS_DIR}\n"
        f"Compute the average images before generating GMM parameters." + _HOWTO_MEANS)

generators = os.listdir(MEANS_DIR)

patterns_means = {}
for gen in generators:
    patterns_means[gen] = os.path.join(MEANS_DIR, gen, '*.png')

# Select real samples based on reals parameter
required = ['ffhq', 'celeba_hq'] if reals == 'ffhq+celeba_hq' else [reals]
missing = [r for r in required if r not in patterns_means]
if missing:
    raise FileNotFoundError(
        f"No average-images subfolder(s) for {missing} under {MEANS_DIR}. "
        f"Available: {sorted(patterns_means.keys())}" + _HOWTO_MEANS)

real_sample = []
for r in required:
    real_sample += glob(patterns_means[r])

if len(real_sample) == 0:
    raise FileNotFoundError(
        f"No average images (*.png) found for reals='{reals}' under {MEANS_DIR}." + _HOWTO_MEANS)

print(f"Number of real samples: {len(real_sample)}")
print("Extracting features...")

print(f"Patch-centroid representation: {const.PATCH_NUM_REPR} patches of "
      f"{config['input_size']}px per mean image (seed {const.PATCH_SEED}).")
real_features = []
for img_path in tqdm(real_sample, total=len(real_sample)):
    real_features.append(mean_image_centroid(img_path))

print("Feature extraction complete!")
    
gmm = {"real": []}

print("\nFitting Gaussian Mixture Models...")
REG_COVAR_SCHEDULE = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]
for i in range(num_layers):
    real_features_ = np.stack([feats[i] for feats in real_features]).astype(np.float64)
    n_samples, n_features = real_features_.shape
    print(f"  Layer {i} (out_indices[{i}]={out_indices[i]}): shape {real_features_.shape}")
    
    fitted = False
    for reg in REG_COVAR_SCHEDULE:
        try:
            clf = GaussianMixture(n_components=n_components, reg_covar=reg, random_state=42)
            clf.fit(real_features_)
            if reg > REG_COVAR_SCHEDULE[0]:
                print(f"    ⚠️  Fitted with reg_covar={reg} (n={n_samples} < d={n_features}, covariance is rank-deficient)")
            fitted = True
            break
        except ValueError:
            continue
    
    if not fitted:
        raise RuntimeError(f"GMM fit failed for layer {i} even with reg_covar={REG_COVAR_SCHEDULE[-1]}. "
                           f"Consider using pooling_type='mean' for this backbone.")
    
    gmm["real"].append([clf.means_, clf.covariances_])

output_filename = f"{const.WORKING_DIR}/parameters/{PREFIX}{n_components}-gmm_parameters_{model_name}_indices_{str(out_indices)}_{reals}_{config['input_size']}_{pooling_type}.npy"

print(f"\nSaving GMM parameters to: {output_filename}")
np.save(output_filename, gmm)
print("Done!")