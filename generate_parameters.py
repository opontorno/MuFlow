import torch
import os
import torch.nn as nn
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
import random
import timm
import yaml
import argparse
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from muflow import constants as const
from muflow.fourier_utils import calculate_fourier_magnitude_rgb
from tqdm import tqdm
import pdb


# === Argument Parser ===
parser = argparse.ArgumentParser(description='Generate GMM parameters for FastFlow')
parser.add_argument('--model_name', type=str, default='resnet50', help='Backbone model name')
parser.add_argument('--reals', type=str, default='ffhq', choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'], help='Real images dataset')
parser.add_argument('--use_fourier', action='store_true', help='Use Fourier magnitude spectrum instead of RGB')
args = parser.parse_args()

# === Hyperparameters ===
model_name  = args.model_name
reals       = args.reals
use_fourier = args.use_fourier

if use_fourier and model_name in (const.DINO_BACKBONES + const.CLIP_BACKBONES):
    raise ValueError("use_fourier is not supported for DINOv2 / CLIP backbones (they require 3-channel RGB input).")

config_path = f"{const.WORKING_DIR}/configs/{model_name}.yaml"
config = yaml.safe_load(open(config_path, "r"))
print("Model config: ", config)
print(f"Use Fourier: {use_fourier}")
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

def _imagenet_normalize(img_t):
    """Apply ImageNet normalisation to a (1,3,H,W) float tensor in [0,255]."""
    img_t = img_t / 255.0
    mean  = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std   = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (img_t - mean) / std

# === Get features ===
def get_features(img, apply_normalization=False):
    """
    Extract features from an image array.

    Args:
        img: numpy array (RGB uint8 or Fourier float32)
        apply_normalization: if True normalise with ImageNet stats (RGB only)
    Returns:
        list of feature tensors (one per layer / block)
    """
    img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()

    if apply_normalization:
        img_t = _imagenet_normalize(img_t)

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
            # CLIPVisualExtractor expects ImageNet-normalised input
            # and handles its own renormlisation internally
            return model(img_t)
        else:  # cnn
            return model(img_t)
            x = x + model.pos_embed
def get_features_from_path(path, apply_fourier=False): 
    """
    Load image from path and extract features.
    
    Args:
        path: Path to image file
        apply_fourier: If True, apply Fourier transform before feature extraction
        
    Returns:
        features: Extracted features from the model
    """
    # Load and resize image
    img = np.array(Image.open(path).convert("RGB").resize([config['input_size'], config['input_size']]))
    
    # Apply Fourier transform if requested
    if apply_fourier:
        img = calculate_fourier_magnitude_rgb(img)
    
    # Extract features with ImageNet normalization (only for RGB, not Fourier)
    features = get_features(img, apply_normalization=not apply_fourier)
    return features


common_path = "/media/orazio_mattia_group/ad4dd/WILD_means/500"
generators = os.listdir(common_path)

patterns_means = {}
for gen in generators:
    patterns_means[gen] = os.path.join(common_path, gen, '*.png')

# Select real samples based on reals parameter
if reals == 'ffhq':
    real_sample = glob(patterns_means['ffhq'])
elif reals == 'celeba_hq':
    real_sample = glob(patterns_means['celeba_hq'])
else:
    real_sample = glob(patterns_means['ffhq']) + glob(patterns_means['celeba_hq'])

print(f"Number of real samples: {len(real_sample)}")
print(f"Extracting features (Fourier: {use_fourier})...")

real_features = []
for i, img_path in tqdm(enumerate(real_sample), total=len(real_sample)):
    
    # Extract features (with or without Fourier transform)
    features = get_features_from_path(img_path, apply_fourier=use_fourier)
    feats_ = []
    for feats in features:
        if pooling_type == 'mean':
            # Spatial pooling: (B, C, H, W) -> (B, C)
            feats_.append(feats.mean([2, 3]).flatten().cpu().numpy())
        elif pooling_type == 'max':
            # Max pooling: (B, C, H, W) -> (B, C)
            feats_.append(feats.flatten(2).max(-1)[0].cpu().numpy().squeeze(0))
        elif pooling_type == 'mean_std':
            # Mean + Std concatenation: (B, C, H, W) -> (B, 2*C)
            mean_feats = feats.mean([2, 3]).flatten().cpu().numpy()
            std_feats = feats.std([2, 3]).flatten().cpu().numpy()
            feats_.append(np.concatenate([mean_feats, std_feats]))
        elif pooling_type == 'flatten':
            # Mean on channels, then flatten spatial: (B, C, H, W) -> (B, C, H) -> (B, C*H)
            feats_.append(feats.mean(-1).flatten(1).cpu().numpy().squeeze(0))
        else:  # full_flatten
            # Full flatten: (B, C, H, W) -> (B, C*H*W)
            feats_.append(feats.flatten(1).cpu().numpy().squeeze(0))
    real_features.append(feats_)

print("Feature extraction complete!")
    
gmm = {"real": []}

print("\nFitting Gaussian Mixture Models...")
clf = GaussianMixture(n_components=n_components, reg_covar=1e-6, random_state=42)
for i in range(num_layers):
    real_features_ = np.stack([feats[i] for feats in real_features])
    print(f"  Layer {i} (out_indices[{i}]={out_indices[i]}): shape {real_features_.shape}")
    
    clf.fit(real_features_)
    gmm["real"].append([clf.means_, clf.covariances_])

if use_fourier:
    output_filename = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{model_name}_indices_{str(out_indices)}_fourier_{reals}_{config['input_size']}_{pooling_type}.npy"
else:
    output_filename = f"{const.WORKING_DIR}/parameters/{n_components}-gmm_parameters_{model_name}_indices_{str(out_indices)}_{reals}_{config['input_size']}_{pooling_type}.npy"

print(f"\nSaving GMM parameters to: {output_filename}")
np.save(output_filename, gmm)
print("Done!")