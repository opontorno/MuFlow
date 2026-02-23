import os
import random
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import timm
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# Enable loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True

# === Hyperparameters ===
model_name = "resnet50"
config_path = f"configs/{model_name}.yaml"
config = yaml.safe_load(open(config_path, "r"))
print("Model config: ", config)

# === Model Setup ===
out_indices = config.get("out_indices", [1, 2, 3])  # Get from config or default
print(f"Using out_indices: {out_indices}")

pooling_type = config.get("pooling_type", "mean")  # Get from config or default
print(f"Using pooling_type: {pooling_type}")

model = timm.create_model(model_name, pretrained=True, features_only=True, in_chans=3, out_indices=out_indices)
model.eval()

channels = model.feature_info.channels()
print("Channels: ", channels)
scales = model.feature_info.reduction()
print("Scales: ", scales)
num_layers = len(out_indices)
print(f"Number of layers: {num_layers}")

# Create output directory for plots
os.makedirs('.pictures', exist_ok=True)


# === Helper Functions ===
def get_features(img):
    """Extract features from an image array."""
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        features = model(img)
    return features


def get_features_from_path(path): 
    """Load image from path and extract features."""
    img = np.array(Image.open(path).convert("RGB").resize([config['input_size'], config['input_size']]))
    return get_features(img)


def calculate_fourier_spectrum(image):
    """Calculate Fourier spectrum of an image."""
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    return magnitude_spectrum


def build_generators_dict(patterns_dict):
    """Build a dictionary mapping generator names to lists of image paths."""
    return {
        'Flux.1': glob(patterns_dict.get('Flux.1', '')),
        'ffhq': glob(patterns_dict.get('ffhq', '')),
        'Stable DIffusion 3.5': glob(patterns_dict.get('Stable DIffusion 3.5', '')),
        'Starry AI': glob(patterns_dict.get('Starry AI', '')),
        'StyleGAN3': glob(patterns_dict.get('StyleGAN3', '')),
        'Stable Diffusion XL': glob(patterns_dict.get('Stable Diffusion XL', '')),
        'SD Attend and Excite': glob(patterns_dict.get('Stable Diffusion Attend and Excite', '')),
        'Midjourney': glob(patterns_dict.get('Midjourney', '')),
        'Stable Cascade': glob(patterns_dict.get('Stable Cascade', '')),
        'Flux.1.1 Pro': glob(patterns_dict.get('Flux.1.1 Pro', '')),
        'Deep AI': glob(patterns_dict.get('Deep AI', '')),
        'StyleGAN2': glob(patterns_dict.get('StyleGAN2', '')),
        'celeba_hq': glob(patterns_dict.get('celeba_hq', '')),
        'Hotpot AI': glob(patterns_dict.get('Hotpot AI', '')),
        'Tencent Hunyuan': glob(patterns_dict.get('Tencent Hunyuan', '')),
        'Dall-E 3': glob(patterns_dict.get('Dall-E 3', '')),
        'StyleGAN': glob(patterns_dict.get('StyleGAN', '')),
        'Nvidia Sana PAG': glob(patterns_dict.get('Nvidia Sana PAG', '')),
        'StarGAN': glob(patterns_dict.get('STARGAN', '')),
        'AttGAN': glob(patterns_dict.get('ATTGAN', '')),
        'GDWCT': glob(patterns_dict.get('GDWCT', ''))
    }


def plot_generators_grid(generators_dict, save_path, cols=6):
    """Plot a grid of sample images from each generator."""
    num_generators = len(generators_dict)
    rows = (num_generators + cols - 1) // cols
    
    plt.figure(figsize=(3 * cols, 3 * rows))
    
    for idx, (name, img_list) in enumerate(generators_dict.items()):
        if not img_list:
            continue
        
        img_path = random.choice(img_list)
        try:
            img = Image.open(img_path).convert("RGB")
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255
            
            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img_tensor.permute(1, 2, 0))
            ax.set_title(name, fontsize=20)
            ax.axis('off')
        except Exception as e:
            print(f"Error with image {img_path}: {e}")
            continue
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.15, wspace=0.001)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved grid to {save_path}")


def extract_features_by_class(patterns_dict, sample_size=100, num_layers=3, pooling_type='mean'):
    """Extract features from images grouped by class/generator.
    
    Args:
        patterns_dict: Dictionary mapping labels to file patterns
        sample_size: Number of samples per class
        num_layers: Number of feature layers to extract
        pooling_type: Type of spatial pooling ('mean', 'max', 'mean_std', 'flatten')
    """
    features_by_class = {}
    labels = []
    
    for label, pattern in patterns_dict.items():
        files = glob(pattern)
        if not files:
            print(f"[!] No files found for {label}. Skipping.")
            continue
        
        sample = random.sample(files, min(sample_size, len(files)))
        layer_features = {i: [] for i in range(num_layers)}
        
        for img_path in sample:
            try:
                features = get_features_from_path(img_path)
                for i in range(len(features)):
                    feats = features[i]
                    
                    # Apply pooling based on pooling_type
                    if pooling_type == 'mean':
                        # Spatial pooling: (B, C, H, W) -> (B, C)
                        vec = feats.mean(dim=(2, 3)).flatten().cpu().numpy()
                    elif pooling_type == 'max':
                        # Max pooling: (B, C, H, W) -> (B, C)
                        vec = feats.flatten(2).max(-1)[0].cpu().numpy().squeeze(0)
                    elif pooling_type == 'mean_std':
                        # Mean + Std concatenation: (B, C, H, W) -> (B, 2*C)
                        mean_feats = feats.mean(dim=(2, 3)).flatten().cpu().numpy()
                        std_feats = feats.std(dim=(2, 3)).flatten().cpu().numpy()
                        vec = np.concatenate([mean_feats, std_feats])
                    else:  # flatten
                        # Mean on channels, then flatten spatial: (B, C, H, W) -> (B, H, W) -> (B, H*W)
                        vec = feats.mean(1).flatten(1).cpu().numpy().squeeze(0)
                    
                    layer_features[i].append(vec)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        
        features_by_class[label] = {
            i: np.array(layer_features[i]) for i in range(len(layer_features))
        }
        if len(layer_features[0]) > 0:
            labels.extend([label] * len(layer_features[0]))
    
    return features_by_class, labels


def plot_tsne_analysis(features_by_class, colors_dict, save_prefix, num_layers=None):
    """Run t-SNE analysis and plot for each feature layer."""
    if num_layers is None:
        # Infer from features_by_class
        num_layers = len(next(iter(features_by_class.values())))
    
    for layer_idx in range(num_layers):
        print(f"Applying t-SNE for layer {layer_idx}...")
        
        # Stack features across classes for this layer
        all_features = np.vstack([features_by_class[label][layer_idx] 
                                  for label in features_by_class 
                                  if len(features_by_class[label][layer_idx]) > 0])
        
        tsne = TSNE(n_components=2, random_state=42)
        features_2d = tsne.fit_transform(all_features)
        
        # Plotting
        print(f"Plotting layer {layer_idx}...")
        plt.figure(figsize=(12, 10))
        start = 0
        
        for label in features_by_class:
            feats = features_by_class[label][layer_idx]
            if len(feats) == 0:
                continue
            count = len(feats)
            subset_2d = features_2d[start:start + count]
            plt.scatter(
                subset_2d[:, 0], subset_2d[:, 1],
                label=label,
                c=colors_dict.get(label, 'gray'),
                alpha=0.6
            )
            start += count
        
        plt.title(f't-SNE visualization of image features (Layer {layer_idx})')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        
        save_path = f'.pictures/{save_prefix}_layer_{layer_idx}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved to {save_path}")


# === Color mapping for plotting ===
colors = {
    'Leonardo AI': 'lightcoral',
    'Flux.1': 'lightskyblue',

    'ffhq': 'black',
    'celeba_hq': 'crimson',

    'Stable DIffusion 3.5': 'lightseagreen',
    'Freepik': 'paleturquoise',
    'Starry AI': 'hotpink',
    'StyleGAN3': 'mediumpurple',
    'Stable Diffusion XL': 'lightgreen',
    'Stable Diffusion Attend and Excite': 'palegreen',
    'SD Attend and Excite': 'mediumspringgreen',
    'Dall-E': 'lightsteelblue',
    'Midjourney': 'violet',
    'Stable Cascade': 'cornflowerblue',
    'Adobe Firefly': 'lightsalmon',
    'Flux.1.1 Pro': 'khaki',
    'Deep AI': 'gold',
    'StyleGAN2': 'mediumaquamarine',
    'Hotpot AI': 'salmon',
    'Tencent Hunyuan': 'silver',
    'Dall-E 3': 'gainsboro',
    'StyleGAN': 'rosybrown',
    'Nvidia Sana PAG': 'lemonchiffon',
    'StarGAN': 'steelblue',
    'AttGAN': 'orchid',
    'GDWCT': 'yellowgreen'
}



# ============================================================================
# PART 1: Analysis on Single Images
# ============================================================================
print("\n" + "="*80)
print("PART 1: Analyzing Single Images")
print("="*80 + "\n")

common_path = "/media/orazio_mattia_group/ad4dd/WILD"
patterns = {}

# Get generators from Closed Set
closed_set_path = os.path.join(common_path, 'Closed Set')
if os.path.exists(closed_set_path):
    generators = [folder for folder in os.listdir(closed_set_path) 
                  if os.path.isdir(os.path.join(closed_set_path, folder))]
    for gen in generators:
        patterns[gen] = os.path.join(closed_set_path, gen, '*.png')

# Get generators from Open Set
open_set_path = os.path.join(common_path, 'Open Set')
if os.path.exists(open_set_path):
    generators = [folder for folder in os.listdir(open_set_path) 
                  if os.path.isdir(os.path.join(open_set_path, folder))]
    for gen in generators:
        patterns[gen] = os.path.join(open_set_path, gen, '*.png')

dfx_common_path = '/media/orazio_mattia_group/ad4dd/datasets_DFX'
if os.path.exists(dfx_common_path):
    dfx_generators = [folder for folder in os.listdir(dfx_common_path) 
                      if os.path.isdir(os.path.join(dfx_common_path, folder))]
    for gen in dfx_generators:
        patterns[gen] = os.path.join(dfx_common_path, gen, '*.png')

# Add real image datasets
patterns['ffhq'] = '/media/orazio_mattia_group/ad4dd/ffhq/*/*.png'
patterns['celeba_hq'] = '/media/orazio_mattia_group/ad4dd/celeba_hq/*/*/*.jpg'

# Build generators dictionary
generators_dict_single = build_generators_dict(patterns)

# Plot grid of sample images
print("Plotting sample images grid...")
plot_generators_grid(generators_dict_single, f'.pictures/analysis-{model_name}_single_images_grid.png')

# Extract features
print("Extracting features from single images...")
features_by_class_single, labels_single = extract_features_by_class(patterns, sample_size=100, num_layers=num_layers, pooling_type=pooling_type)

# Run t-SNE analysis and plot
print("Running t-SNE analysis on single images...")
plot_tsne_analysis(features_by_class_single, colors, f'analysis-{model_name}_tsne_single', num_layers=num_layers)


# ============================================================================
# PART 2: Analysis on Mean Images
# ============================================================================
print("\n" + "="*80)
print("PART 2: Analyzing Mean Images")
print("="*80 + "\n")

common_path_means = "/media/orazio_mattia_group/ad4dd/WILD_means/500"
patterns_means = {}

if os.path.exists(common_path_means):
    generators_means = os.listdir(common_path_means)
    for gen in generators_means:
        patterns_means[gen] = os.path.join(common_path_means, gen, '*.png')
else:
    print(f"Warning: Mean images path does not exist: {common_path_means}")

# Build generators dictionary for means
generators_dict_means = build_generators_dict(patterns_means)

# Plot grid of mean images
print("Plotting mean images grid...")
plot_generators_grid(generators_dict_means, f'.pictures/analysis-{model_name}_mean_images_grid.png')

# Extract features from mean images
print("Extracting features from mean images...")
features_by_class_means, labels_means = extract_features_by_class(patterns_means, sample_size=100, num_layers=num_layers, pooling_type=pooling_type)

# Run t-SNE analysis and plot
print("Running t-SNE analysis on mean images...")
plot_tsne_analysis(features_by_class_means, colors, f'analysis-{model_name}_tsne_means', num_layers=num_layers)

print("\n" + "="*80)
print("Analysis complete! All plots saved to .pictures/ folder")
print("="*80)
