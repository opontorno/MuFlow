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
model_name = "resnet18"
config_path = f"FastFlow/configs/{model_name}.yaml"
config = yaml.safe_load(open(config_path, "r"))
print("Model config: ", config)

# === Model Setup ===
model = timm.create_model(model_name, pretrained=True, features_only=True, in_chans=3, out_indices=[1, 2, 3])
model.eval()

channels = model.feature_info.channels()
print("Channels: ", channels)
scales = model.feature_info.reduction()
print("Scales: ", scales)

# Create output directory for plots
os.makedirs('.pictures', exist_ok=True)


# === Helper Functions ===
def calculate_fourier_magnitude_rgb(image):
    """Calculate Fourier magnitude spectrum from RGB image via grayscale conversion.
    
    Converts RGB to grayscale using standard luminance weights, then computes FFT
    on the single grayscale channel and replicates to 3 channels for CNN input.
    
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
    
    # Replicate to 3 channels for CNN input (H, W) → (H, W, 3)
    magnitude_rgb = np.stack([magnitude, magnitude, magnitude], axis=-1)
    return magnitude_rgb


def get_features(img):
    """Extract features from an image array."""
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        features = model(img)
    return features


def get_features_from_path(path): 
    """Load image from path, transform to Fourier magnitude domain, and extract features."""
    # Load and resize image
    img = np.array(Image.open(path).convert("RGB").resize([config['input_size'], config['input_size']]))
    
    # Transform to Fourier magnitude domain
    img_fourier = calculate_fourier_magnitude_rgb(img)
    
    # Normalize to [0, 255] range for CNN input
    img_fourier = (img_fourier - img_fourier.min()) / (img_fourier.max() - img_fourier.min() + 1e-8) * 255
    
    # Extract features from Fourier representation
    return get_features(img_fourier)


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
        'Nvidia Sana PAG': glob(patterns_dict.get('Nvidia Sana PAG', ''))
    }


def plot_generators_grid_with_fourier(generators_dict, save_path, cols=6, show_fourier=True):
    """Plot a grid of sample images from each generator with optional Fourier visualization.
    
    Args:
        generators_dict: Dictionary mapping generator names to image paths
        save_path: Path to save the figure
        cols: Number of columns in the grid
        show_fourier: If True, show Fourier magnitude spectrum instead of original image
    """
    num_generators = len(generators_dict)
    rows = (num_generators + cols - 1) // cols
    
    plt.figure(figsize=(3 * cols, 3 * rows))
    
    for idx, (name, img_list) in enumerate(generators_dict.items()):
        if not img_list:
            continue
        
        img_path = random.choice(img_list)
        try:
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img.resize([config['input_size'], config['input_size']]))
            
            if show_fourier:
                # Show Fourier magnitude spectrum
                img_fourier = calculate_fourier_magnitude_rgb(img_array)
                # Normalize for visualization
                img_display = (img_fourier - img_fourier.min()) / (img_fourier.max() - img_fourier.min() + 1e-8)
            else:
                # Show original image
                img_display = img_array / 255.0
            
            ax = plt.subplot(rows, cols, idx + 1)
            ax.imshow(img_display)
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


def plot_fourier_comparison(generators_dict, save_path, num_samples=6):
    """Plot a comparison grid showing original images vs their Fourier magnitude spectra.
    
    Args:
        generators_dict: Dictionary mapping generator names to image paths
        save_path: Path to save the figure
        num_samples: Number of samples to show
    """
    # Select random samples from different generators
    samples = []
    labels = []
    
    for name, img_list in generators_dict.items():
        if img_list:
            samples.append(random.choice(img_list))
            labels.append(name)
            if len(samples) >= num_samples:
                break
    
    fig, axes = plt.subplots(2, len(samples), figsize=(3 * len(samples), 6))
    
    for idx, (img_path, label) in enumerate(zip(samples, labels)):
        try:
            # Load and resize image
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img.resize([config['input_size'], config['input_size']]))
            
            # Original image
            axes[0, idx].imshow(img_array)
            axes[0, idx].set_title(f"{label}\n(Original)", fontsize=10)
            axes[0, idx].axis('off')
            
            # Fourier magnitude spectrum
            img_fourier = calculate_fourier_magnitude_rgb(img_array)
            img_fourier_norm = (img_fourier - img_fourier.min()) / (img_fourier.max() - img_fourier.min() + 1e-8)
            axes[1, idx].imshow(img_fourier_norm)
            axes[1, idx].set_title("Fourier Magnitude", fontsize=10)
            axes[1, idx].axis('off')
            
        except Exception as e:
            print(f"Error with image {img_path}: {e}")
            continue
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Saved comparison to {save_path}")


def extract_features_by_class(patterns_dict, sample_size=100):
    """Extract features from images grouped by class/generator."""
    features_by_class = {}
    labels = []
    
    for label, pattern in patterns_dict.items():
        files = glob(pattern)
        if not files:
            print(f"[!] No files found for {label}. Skipping.")
            continue
        
        sample = random.sample(files, min(sample_size, len(files)))
        layer_features = {0: [], 1: [], 2: []}
        
        for img_path in sample:
            try:
                features = get_features_from_path(img_path)
                for i in range(len(features)):
                    vec = features[i].mean(dim=(2, 3)).flatten().cpu().numpy()
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


def plot_tsne_analysis(features_by_class, colors_dict, save_prefix, num_layers=3):
    """Run t-SNE analysis and plot for each feature layer."""
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
        
        plt.title(f't-SNE visualization of Fourier magnitude features (Layer {layer_idx})')
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
    'ffhq': 'darkblue',
    'Stable DIffusion 3.5': 'lightseagreen',
    'Freepik': 'lightcyan',
    'Starry AI': 'lightpink',
    'StyleGAN3': 'plum',
    'Stable Diffusion XL': 'lightgreen',
    'Stable Diffusion Attend and Excite': 'palegreen',
    'SD Attend and Excite': 'palegreen',
    'Dall-E': 'lightsteelblue',
    'Midjourney': 'violet',
    'Stable Cascade': 'lightblue',
    'Adobe Firefly': 'lightsalmon',
    'Flux.1.1 Pro': 'khaki',
    'Deep AI': 'lightgoldenrodyellow',
    'StyleGAN2': 'mediumaquamarine',
    'celeba_hq': 'darkgoldenrod',
    'Hotpot AI': 'lightcoral',
    'Tencent Hunyuan': 'lightgray',
    'Dall-E 3': 'gainsboro',
    'StyleGAN': 'lightcoral',
    'Nvidia Sana PAG': 'lightyellow'
}


# ============================================================================
# PART 1: Analysis on Single Images (Fourier Domain)
# ============================================================================
print("\n" + "="*80)
print("PART 1: Analyzing Single Images in Fourier Magnitude Domain")
print("="*80 + "\n")

common_path = "/media/orazio_mattia_group/ad4dd/FF4ALL"
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

# Add real image datasets
patterns['ffhq'] = '/media/orazio_mattia_group/ad4dd/ffhq/*/*.png'
patterns['celeba_hq'] = '/media/orazio_mattia_group/ad4dd/celeba_hq/*/*/*.jpg'

# Build generators dictionary
generators_dict_single = build_generators_dict(patterns)

# Plot comparison: original vs Fourier
print("Plotting comparison: Original vs Fourier magnitude...")
plot_fourier_comparison(generators_dict_single, f'.pictures/analysis-{model_name}_fourier_comparison.png')

# Plot grid of Fourier magnitude spectra
print("Plotting Fourier magnitude spectra grid...")
plot_generators_grid_with_fourier(generators_dict_single, 
                                   f'.pictures/analysis-{model_name}_fourier_single_images_grid.png', 
                                   show_fourier=True)

# Extract features (from Fourier domain)
print("Extracting features from Fourier magnitude spectra...")
features_by_class_single, labels_single = extract_features_by_class(patterns, sample_size=100)

# Run t-SNE analysis and plot
print("Running t-SNE analysis on Fourier features...")
plot_tsne_analysis(features_by_class_single, colors, f'analysis-{model_name}_fourier_tsne_single')


# ============================================================================
# PART 2: Analysis on Mean Images (Fourier Domain)
# ============================================================================
print("\n" + "="*80)
print("PART 2: Analyzing Mean Images in Fourier Magnitude Domain")
print("="*80 + "\n")

common_path_means = "/media/orazio_mattia_group/ad4dd/FF4ALL_means/500"
patterns_means = {}

if os.path.exists(common_path_means):
    generators_means = os.listdir(common_path_means)
    for gen in generators_means:
        patterns_means[gen] = os.path.join(common_path_means, gen, '*.png')
else:
    print(f"Warning: Mean images path does not exist: {common_path_means}")

# Build generators dictionary for means
generators_dict_means = build_generators_dict(patterns_means)

# Plot grid of Fourier magnitude spectra for mean images
print("Plotting Fourier magnitude spectra grid for mean images...")
plot_generators_grid_with_fourier(generators_dict_means, 
                                   f'.pictures/analysis-{model_name}_fourier_mean_images_grid.png',
                                   show_fourier=True)

# Extract features from mean images (in Fourier domain)
print("Extracting features from Fourier magnitude spectra of mean images...")
features_by_class_means, labels_means = extract_features_by_class(patterns_means, sample_size=100)

# Run t-SNE analysis and plot
print("Running t-SNE analysis on Fourier features of mean images...")
plot_tsne_analysis(features_by_class_means, colors, f'analysis-{model_name}_fourier_tsne_means')

print("\n" + "="*80)
print("Fourier Analysis complete! All plots saved to .pictures/ folder")
print("Files saved with prefix: analysis-{model_name}_fourier_*")
print("="*80)
