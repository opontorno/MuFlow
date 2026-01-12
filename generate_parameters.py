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

# === Argument Parser ===
parser = argparse.ArgumentParser(description='Generate GMM parameters for FastFlow')
parser.add_argument('--model_name', type=str, default='resnet18', 
                    help='Backbone model name')
parser.add_argument('--reals', type=str, default='ffhq', 
                    choices=['ffhq', 'celeba_hq', 'ffhq+celeba_hq'],
                    help='Real images dataset')
parser.add_argument('--use_fourier', action='store_true',
                    help='Use Fourier magnitude spectrum instead of RGB')
args = parser.parse_args()

# === Hyperparameters ===
model_name = args.model_name
reals = args.reals
use_fourier = args.use_fourier

config_path = f"{const.WORKING_DIR}/FastFlow/configs/{model_name}.yaml" 
config = yaml.safe_load(open(config_path, "r"))
print("Model config: ", config)
print(f"Use Fourier: {use_fourier}")
print(f"Reals dataset: {reals}")


# === Model Setup ===
model = timm.create_model(model_name, pretrained=True, features_only=True, in_chans=3, out_indices=[1, 2, 3])
model.eval()

channels = model.feature_info.channels()
print("Channels: ", channels)
scales = model.feature_info.reduction()
print("Scales: ", scales)

# === Fourier Transform Function ===
def calculate_fourier_magnitude_rgb(image):
    """
    Calculate Fourier magnitude spectrum from RGB image via grayscale conversion.
    Matches implementation in analysis_WILD_fourier.py and FastFlow/dataset.py
    
    Args:
        image: numpy array of shape (H, W, 3) with RGB channels
        
    Returns:
        magnitude_rgb: numpy array of shape (H, W, 3) with replicated magnitude spectrum
    """
    # Convert RGB to grayscale using standard luminance weights
    # Y = 0.299*R + 0.587*G + 0.114*B
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # Compute 2D FFT on grayscale image
    f = np.fft.fft2(gray)
    # Shift zero frequency to center
    fshift = np.fft.fftshift(f)
    # Compute magnitude spectrum with log scale
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    # Normalize to [0, 255] range
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8) * 255
    
    # Replicate to 3 channels for CNN input (H, W) → (H, W, 3)
    magnitude_rgb = np.stack([magnitude, magnitude, magnitude], axis=-1)
    
    return magnitude_rgb.astype(np.uint8)


# === Get features ===
def get_features(img):
    """Extract features from an image array."""
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
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
    
    # Extract features
    features = get_features(img)
    return features


common_path = "/media/orazio_mattia_group/ad4dd/FF4ALL_means/500"
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
for i, img_path in enumerate(real_sample):
    if (i + 1) % 10 == 0:
        print(f"  Processing {i+1}/{len(real_sample)}...")
    
    # Extract features (with or without Fourier transform)
    features = get_features_from_path(img_path, apply_fourier=use_fourier)
    feats_ = []
    for feats in features:
        feats_.append(feats.mean([2, 3]).flatten().cpu().numpy())
    real_features.append(feats_)

print("Feature extraction complete!")
    
gmm = {"real": []}

print("\nFitting Gaussian Mixture Models...")
clf = GaussianMixture()
for i in range(len(real_features[0])):
    real_features_ = np.stack([feats[i] for feats in real_features])
    print(f"  Layer {i}: shape {real_features_.shape}")
    
    clf.fit(real_features_)
    gmm["real"].append([clf.means_, clf.covariances_])

# Generate output filename based on parameters
if use_fourier:
    output_filename = f"{const.WORKING_DIR}/parameters/gmm_parameters_{model_name}_fourier_{reals}_{config['input_size']}.npy"
else:
    output_filename = f"{const.WORKING_DIR}/parameters/gmm_parameters_{model_name}_{reals}_{config['input_size']}.npy"

print(f"\nSaving GMM parameters to: {output_filename}")
np.save(output_filename, gmm)
print("Done!")