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
import FastFlow.constants as const
from fourier_utils import calculate_fourier_magnitude_rgb
from tqdm import tqdm
import pdb


# === Argument Parser ===
parser = argparse.ArgumentParser(description='Generate GMM parameters for FastFlow')
parser.add_argument('--model_name', type=str, default='resnet18', 
                    help='Backbone model name')
parser.add_argument('--reals', type=str, default='ffhq', 
                    choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'],
                    help='Real images dataset')
parser.add_argument('--use_fourier', action='store_true',
                    help='Use Fourier magnitude spectrum instead of RGB')
parser.add_argument('--pooling_type', type=str, default='mean',
                    choices=['mean', 'flatten'],
                    help='Spatial pooling type: mean or flatten')
args = parser.parse_args()

# === Hyperparameters ===
model_name = args.model_name
reals = args.reals
use_fourier = args.use_fourier
pooling_type = args.pooling_type

config_path = f"{const.WORKING_DIR}/FastFlow/configs/{model_name}.yaml" 
config = yaml.safe_load(open(config_path, "r"))
print("Model config: ", config)
print(f"Use Fourier: {use_fourier}")
print(f"Reals dataset: {reals}")
print(f"Pooling type: {pooling_type}")


# === Model Setup ===
out_indices = config.get("out_indices", [1, 2, 3])  # Get from config or default
print(f"Using out_indices: {out_indices}")

if config['backbone_name'] in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
    model = timm.create_model(config['backbone_name'], pretrained=True, in_chans=3)
    channels = [768]
    scales = [16]
else:
    model = timm.create_model(config['backbone_name'], pretrained=True, features_only=True, in_chans=3, out_indices=out_indices)
    channels = model.feature_info.channels()
    scales = model.feature_info.reduction()

model.eval()


print("Channels: ", channels)
print("Scales: ", scales)
num_layers = len(out_indices)
print(f"Number of layers: {num_layers}")

# === Get features ===
def get_features(img, apply_normalization=False):
    """
    Extract features from an image array.
    
    Args:
        img: Image as numpy array (RGB uint8 [0-255] or Fourier float32 [0-1])
        apply_normalization: If True, apply ImageNet normalization (for RGB images)
    
    Returns:
        features: Extracted features from the model
    """
    # Convert to tensor: [H, W, C] -> [1, C, H, W]
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    
    # If applying normalization, this is RGB (uint8) - normalize to [0,1] and apply ImageNet stats
    if apply_normalization:
        # Normalize to [0, 1] range
        img = img / 255.0
        
        # Apply ImageNet normalization (identical to torchvision.transforms.Normalize)
        # Uses same dtype and device as the tensor, as per torchvision implementation
        mean = torch.as_tensor([0.485, 0.456, 0.406], dtype=img.dtype, device=img.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], dtype=img.dtype, device=img.device)
        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)
        img = img.sub(mean).div(std)
    # else: Fourier magnitude is already in [0, 1] range, no normalization needed
    
    with torch.no_grad():
        if isinstance(model, timm.models.vision_transformer.VisionTransformer):
            x = model.patch_embed(img)
            cls_token = model.cls_token.expand(x.shape[0], -1, -1)
            if model.dist_token is None:
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, model.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = model.pos_drop(x + model.pos_embed)
            for i in range(8):  # paper Table 6. Block Index = 7
                x = model.blocks[i](x)
            x = model.norm(x)
            x = x[:, 2:, :]
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, config['input_size'] // 16, config['input_size'] // 16)
            features = x
        elif isinstance(model, timm.models.cait.Cait):
            x = model.patch_embed(img)
            x = x + model.pos_embed
            x = model.pos_drop(x)
            for i in range(41):  # paper Table 6. Block Index = 40
                x = model.blocks[i](x)
            N, _, C = x.shape
            x = x.permute(0, 2, 1)
            x = x.reshape(N, C, config['input_size'] // 16, config['input_size'] // 16)
            features = x
        else:
            features = model(img)
    return features


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

flux_1_means = glob(patterns_means['Flux.1'])
ffhq_means = glob(patterns_means['ffhq'])
stable_diffusion_35_means = glob(patterns_means['Stable DIffusion 3.5'])
starry_ai_means = glob(patterns_means['Starry AI'])
stylegan3_means = glob(patterns_means['StyleGAN3'])
stable_diffusion_xl_means = glob(patterns_means['Stable Diffusion XL'])
attend_and_excite_means = glob(patterns_means['Stable Diffusion Attend and Excite'])
midjourney_means = glob(patterns_means['Midjourney'])
stable_cascade_means = glob(patterns_means['Stable Cascade'])
flux_1_1_pro_means = glob(patterns_means['Flux.1.1 Pro'])
deep_ai_means = glob(patterns_means['Deep AI'])
stylegan2_means = glob(patterns_means['StyleGAN2'])
celeba_hq_means = glob(patterns_means['celeba_hq'])
hotpot_ai_means = glob(patterns_means['Hotpot AI'])
tencent_hunyuan_means = glob(patterns_means['Tencent Hunyuan'])
dalle_3_means = glob(patterns_means['Dall-E 3'])
stylegan_means = glob(patterns_means['StyleGAN'])
nvidia_sana_pag_means = glob(patterns_means['Nvidia Sana PAG'])

# Select real samples based on reals parameter
if reals == 'ffhq':
    real_sample = ffhq_means
elif reals == 'celeba_hq':
    real_sample = celeba_hq_means
else:
    real_sample = ffhq_means + celeba_hq_means

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
        else:  # flatten
            # Flatten spatial dims: (B, C, H, W) -> (B, C*H*W)
            feats_.append(feats.flatten(1).cpu().numpy().squeeze(0))
    real_features.append(feats_)

print("Feature extraction complete!")
    
gmm = {"real": []}

print("\nFitting Gaussian Mixture Models...")
clf = GaussianMixture(n_components=1, reg_covar=1e-6, random_state=42)
for i in range(num_layers):
    real_features_ = np.stack([feats[i] for feats in real_features])
    print(f"  Layer {i} (out_indices[{i}]={out_indices[i]}): shape {real_features_.shape}")
    
    clf.fit(real_features_)
    gmm["real"].append([clf.means_, clf.covariances_])

# Generate output filename based on parameters
if use_fourier:
    output_filename = f"{const.WORKING_DIR}/parameters/gmm_parameters_{model_name}_indices_{str(out_indices)}_fourier_{reals}_{config['input_size']}_{pooling_type}.npy"
else:
    output_filename = f"{const.WORKING_DIR}/parameters/gmm_parameters_{model_name}_indices_{out_indices}_{reals}_{config['input_size']}_{pooling_type}.npy"

print(f"\nSaving GMM parameters to: {output_filename}")
np.save(output_filename, gmm)
print("Done!")