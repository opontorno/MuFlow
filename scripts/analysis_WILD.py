#!/usr/bin/env python3
"""
WILD Dataset Feature Analysis for MuFlow
=========================================

Analyses backbone feature representations (RGB, Fourier magnitude, or SRM
residue domain) on the WILD deepfake dataset using t-SNE visualisations.

Three operating modes (set via --preprocessing):
    none  (default) -- plain RGB, ImageNet-normalised features
    fourier         -- features from Fourier magnitude spectrum (CNN only)
    srm             -- features from SRM high-pass residual maps (CNN only)

Usage:
    python scripts/analysis_WILD.py
    python scripts/analysis_WILD.py --preprocessing fourier --model resnet50
    python scripts/analysis_WILD.py --preprocessing srm     --model resnet50
    python scripts/analysis_WILD.py --model clip_vitb16 --sample_size 200
    python scripts/analysis_WILD.py --save_dir /tmp/plots --seed 0
"""

import os
import sys
import argparse
import random
from glob import glob

import numpy as np
import torch
import timm
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE

# -- Project root on sys.path ------------------------------------------------
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from muflow import constants as const
from muflow.fourier_utils import calculate_fourier_magnitude_rgb
from muflow.srm_utils import calculate_srm_residuals

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ImageNet normalisation stats
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ===========================================================================
# CLI
# ===========================================================================

def get_args():
    parser = argparse.ArgumentParser(
        description="WILD feature-space analysis -- RGB or Fourier magnitude domain."
    )
    parser.add_argument(
        "--model", type=str, default="resnet50",
        choices=const.SUPPORTED_BACKBONES,
        help="Backbone model name (must have a matching config in configs/)."
    )
    parser.add_argument(
        "--preprocessing", type=str, default="none",
        choices=["none", "fourier", "srm"],
        help=(
            "Image preprocessing before feature extraction. "
            "'fourier' and 'srm' are only supported for CNN backbones "
            "(resnet*, wide_resnet*, densenet*)."
        ),
    )
    parser.add_argument(
        "--sample_size", type=int, default=100,
        help="Number of images to sample per class/generator (default: 100).",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None,
        help="Directory for output plots. Defaults to scripts/.pictures/.",
    )
    parser.add_argument("--seed",    type=int,   default=42)
    parser.add_argument("--cols",    type=int,   default=7,   help="Grid columns (default: 7).")
    parser.add_argument("--tsne_alpha", type=float, default=0.6, help="t-SNE scatter opacity.")
    parser.add_argument("--tsne_s",     type=float, default=90,  help="t-SNE marker size.")
    return parser.parse_args()


# ===========================================================================
# Backbone setup
# ===========================================================================

def build_backbone(model_name, config, preprocessing, device):
    """
    Load and configure the feature-extractor backbone.

    Returns
    -------
    model        : nn.Module -- eval mode, on *device*
    backbone_type: str       -- 'cnn' | 'dino' | 'clip' | 'cait_deit'
    channels     : list[int] -- output channels per layer
    scales       : list[int] -- spatial-reduction factor per layer
    out_indices  : list[int]
    """
    out_indices = config.get("out_indices", [1, 2, 3])

    if preprocessing in ("fourier", "srm") and model_name in (const.DINO_BACKBONES + const.CLIP_BACKBONES):
        raise ValueError(
            f"--preprocessing {preprocessing} is not supported for backbone '{model_name}'. "
            "Fourier/SRM analysis only works with CNN backbones "
            "(resnet18/50/101, wide_resnet50_2, densenet121, ...)."
        )

    if model_name in const.DINO_BACKBONES:
        timm_name     = const.DINO_TIMM_NAMES[model_name]
        model         = timm.create_model(timm_name, pretrained=True,
                                          img_size=config["input_size"])
        channels      = [const.DINO_CHANNELS[model_name]] * len(out_indices)
        scales        = [const.DINO_PATCH_SIZE[model_name]] * len(out_indices)
        backbone_type = "dino"

    elif model_name in const.CLIP_BACKBONES:
        from muflow.model import CLIPVisualExtractor
        model         = CLIPVisualExtractor(model_name, out_block_indices=out_indices)
        channels      = [const.CLIP_CHANNELS[model_name]] * len(out_indices)
        scales        = [const.CLIP_PATCH_SIZE[model_name]] * len(out_indices)
        backbone_type = "clip"

    elif model_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
        model         = timm.create_model(model_name, pretrained=True, in_chans=3)
        channels      = [768]
        scales        = [16]
        backbone_type = "cait_deit"

    else:  # CNN (ResNet, WideResNet, DenseNet, ...)
        model = timm.create_model(
            model_name, pretrained=True,
            features_only=True,
            out_indices=out_indices,
            in_chans=3,
        )
        channels      = model.feature_info.channels()
        scales        = model.feature_info.reduction()
        backbone_type = "cnn"

    model.eval().to(device)
    return model, backbone_type, channels, scales, out_indices


# ===========================================================================
# Feature extraction
# ===========================================================================

def get_features(img_array, model, backbone_type, out_indices, device, preprocessing):
    """
    Extract multi-scale feature maps from a (H, W, 3) uint8 numpy array.

    none     : applies ImageNet normalisation.
    fourier  : converts to Fourier magnitude spectrum [0,1] first.
    srm      : converts to SRM residual map [0,1] first.
    CLIP     : CLIPVisualExtractor handles its own renormalisation internally.

    Returns
    -------
    list of (1, C, H', W') tensors -- one per output layer.
    """
    if preprocessing == "fourier":
        arr = calculate_fourier_magnitude_rgb(img_array)   # (H, W, 3) float32 [0,1]
        t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    elif preprocessing == "srm":
        arr = calculate_srm_residuals(img_array, seed=0)   # (H, W, 3) float32 [0,1]
        t   = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
    else:
        t = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t = (t - _IMAGENET_MEAN) / _IMAGENET_STD

    t = t.to(device)
    with torch.no_grad():
        if backbone_type == "dino":
            features = model.get_intermediate_layers(t, n=list(out_indices), reshape=True)
        elif backbone_type in ("clip", "cnn"):
            features = model(t)
        else:  # cait_deit
            features = [model(t)]
    return features


def get_features_from_path(path, config, model, backbone_type, out_indices, device, preprocessing):
    """Load an image from *path*, resize to config input_size, and extract features."""
    img = np.array(
        Image.open(path).convert("RGB").resize(
            [config["input_size"], config["input_size"]], Image.BILINEAR
        )
    )
    return get_features(img, model, backbone_type, out_indices, device, preprocessing)


# ===========================================================================
# Dataset helpers
# ===========================================================================

# Canonical colour palette for t-SNE scatter plots
COLORS = {
    "ffhq":                               "black",
    "celeba_hq":                          "crimson",
    "WDF Real":                           "dimgray",
    "Flux.1":                             "lightskyblue",
    "Flux.1.1 Pro":                       "khaki",
    "Stable DIffusion 3.5":               "lightseagreen",
    "Stable Diffusion XL":                "lightgreen",
    "Stable Cascade":                     "cornflowerblue",
    "Stable Diffusion Attend and Excite": "palegreen",
    "SD Attend and Excite":               "mediumspringgreen",
    "Midjourney":                         "violet",
    "Dall-E 3":                           "gainsboro",
    "Starry AI":                          "hotpink",
    "Deep AI":                            "gold",
    "Hotpot AI":                          "salmon",
    "Tencent Hunyuan":                    "silver",
    "Nvidia Sana PAG":                    "lemonchiffon",
    "StyleGAN":                           "rosybrown",
    "StyleGAN2":                          "mediumaquamarine",
    "StyleGAN3":                          "mediumpurple",
    "STARGAN":                            "steelblue",
    "StarGAN":                            "steelblue",
    "AttGAN":                             "orchid",
    "GDWCT":                              "yellowgreen",
}


def build_data_patterns(data_dir):
    """
    Scan WILD, datasets_DFX, ffhq and celeba_hq under *data_dir* and return
    a {generator_name: glob_pattern} dict.
    """
    patterns = {}

    wild_root = os.path.join(data_dir, "WILD")
    for subset in ("Closed Set", "Open Set"):
        subset_path = os.path.join(wild_root, subset)
        if os.path.isdir(subset_path):
            for gen in os.listdir(subset_path):
                if os.path.isdir(os.path.join(subset_path, gen)):
                    patterns[gen] = os.path.join(subset_path, gen, "*.png")

    dfx_root = os.path.join(data_dir, "datasets_DFX")
    if os.path.isdir(dfx_root):
        for gen in os.listdir(dfx_root):
            if os.path.isdir(os.path.join(dfx_root, gen)):
                patterns[gen] = os.path.join(dfx_root, gen, "*.png")

    patterns["ffhq"]      = os.path.join(data_dir, "ffhq", "*", "*.png")
    patterns["celeba_hq"] = os.path.join(data_dir, "celeba_hq", "*", "*", "*.jpg")
    return patterns


def build_means_patterns(data_dir):
    """Build patterns dict for the WILD_means/500 directory (RGB mode only)."""
    patterns  = {}
    means_root = os.path.join(data_dir, "WILD_means", "500")
    if not os.path.isdir(means_root):
        print(f"  [WARN] Mean-images directory not found: {means_root}")
        return patterns
    for gen in os.listdir(means_root):
        if os.path.isdir(os.path.join(means_root, gen)) and gen != "WDF Real":
            patterns[gen] = os.path.join(means_root, gen, "*.png")
    return patterns


def build_generators_dict(patterns_dict):
    """Resolve all glob patterns and return {name: [image_paths]}."""
    return {name: glob(pat) for name, pat in patterns_dict.items() if pat}


# ===========================================================================
# Plotting helpers
# ===========================================================================

def plot_generators_grid(generators_dict, save_path, cols=7):
    """Grid of one random RGB sample per generator."""
    items = [(n, p) for n, p in generators_dict.items() if p]
    rows  = max(1, (len(items) + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten()

    for idx, (name, img_list) in enumerate(items):
        try:
            img = Image.open(random.choice(img_list)).convert("RGB")
            axes_flat[idx].imshow(np.array(img))
            axes_flat[idx].set_title(name, fontsize=14)
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
        axes_flat[idx].axis("off")

    for ax in axes_flat[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.15, wspace=0.001)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved grid -> {save_path}")


def plot_fourier_grid(generators_dict, save_path, input_size, cols=6):
    """
    Grid of Fourier magnitude spectra -- one image per generator.
    Only called in preprocessing=fourier mode.
    """
    items = [(n, p) for n, p in generators_dict.items() if p]
    rows  = max(1, (len(items) + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten()

    for idx, (name, img_list) in enumerate(items):
        try:
            img_array = np.array(
                Image.open(random.choice(img_list)).convert("RGB")
                .resize([input_size, input_size])
            )
            spectrum = calculate_fourier_magnitude_rgb(img_array)
            axes_flat[idx].imshow(spectrum)
            axes_flat[idx].set_title(name, fontsize=14)
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
        axes_flat[idx].axis("off")

    for ax in axes_flat[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.15, wspace=0.001)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Fourier grid -> {save_path}")


def plot_fourier_comparison(generators_dict, save_path, input_size, num_samples=6):
    """
    Side-by-side comparison: original RGB vs Fourier magnitude spectrum.
    Only called in preprocessing=fourier mode.
    """
    samples, labels = [], []
    for name, img_list in generators_dict.items():
        if img_list:
            samples.append(random.choice(img_list))
            labels.append(name)
            if len(samples) >= num_samples:
                break

    if not samples:
        print("  [WARN] No images available for Fourier comparison.")
        return

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes[:, np.newaxis]

    for idx, (img_path, label) in enumerate(zip(samples, labels)):
        try:
            img_array = np.array(
                Image.open(img_path).convert("RGB").resize([input_size, input_size])
            )
            axes[0, idx].imshow(img_array)
            axes[0, idx].set_title(f"{label}\n(RGB)", fontsize=10)
            axes[0, idx].axis("off")

            spectrum = calculate_fourier_magnitude_rgb(img_array)
            axes[1, idx].imshow(spectrum)
            axes[1, idx].set_title("Fourier magnitude", fontsize=10)
            axes[1, idx].axis("off")
        except Exception as e:
            print(f"  [WARN] {img_path}: {e}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved Fourier comparison -> {save_path}")


def plot_srm_grid(generators_dict, save_path, input_size, cols=6):
    """
    Grid of SRM residual maps -- one image per generator.
    Only called in preprocessing=srm mode.
    """
    items = [(n, p) for n, p in generators_dict.items() if p]
    rows  = max(1, (len(items) + cols - 1) // cols)

    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes_flat = np.array(axes).flatten()

    for idx, (name, img_list) in enumerate(items):
        try:
            img_array = np.array(
                Image.open(random.choice(img_list)).convert("RGB")
                .resize([input_size, input_size])
            )
            residual = calculate_srm_residuals(img_array, seed=0)
            axes_flat[idx].imshow(residual)
            axes_flat[idx].set_title(name, fontsize=14)
        except Exception as e:
            print(f"  [WARN] {name}: {e}")
        axes_flat[idx].axis("off")

    for ax in axes_flat[len(items):]:
        ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92, hspace=0.15, wspace=0.001)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved SRM residual grid -> {save_path}")


def plot_srm_comparison(generators_dict, save_path, input_size, num_samples=6):
    """
    Side-by-side comparison: original RGB vs SRM residual map.
    Only called in preprocessing=srm mode.
    """
    samples, labels = [], []
    for name, img_list in generators_dict.items():
        if img_list:
            samples.append(random.choice(img_list))
            labels.append(name)
            if len(samples) >= num_samples:
                break

    if not samples:
        print("  [WARN] No images available for SRM comparison.")
        return

    n = len(samples)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 6))
    if n == 1:
        axes = axes[:, np.newaxis]

    for idx, (img_path, label) in enumerate(zip(samples, labels)):
        try:
            img_array = np.array(
                Image.open(img_path).convert("RGB").resize([input_size, input_size])
            )
            axes[0, idx].imshow(img_array)
            axes[0, idx].set_title(f"{label}\n(RGB)", fontsize=10)
            axes[0, idx].axis("off")

            residual = calculate_srm_residuals(img_array, seed=0)
            axes[1, idx].imshow(residual)
            axes[1, idx].set_title("SRM residual", fontsize=10)
            axes[1, idx].axis("off")
        except Exception as e:
            print(f"  [WARN] {img_path}: {e}")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Saved SRM comparison -> {save_path}")



# ===========================================================================
# Feature collection
# ===========================================================================

def extract_features_by_class(
    patterns_dict, config, model, backbone_type, out_indices, device,
    preprocessing="none", sample_size=100, pooling_type="mean",
):
    """
    Extract pooled feature vectors grouped by class/generator.

    Samples up to *sample_size* images per class and extracts features from
    every backbone output layer.  Spatial pooling strategy:
        - 'mean'     : average pool  → (C,)
        - 'max'      : max pool      → (C,)
        - 'mean_std' : mean + std    → (2*C,)
        - 'flatten'  : channel-mean then flatten spatial → (H*W,)

    Returns
    -------
    dict {label: {layer_idx: np.ndarray (N, D)}}
    """
    features_by_class = {}
    num_layers        = len(out_indices)

    for label, pattern in patterns_dict.items():
        files = glob(pattern)
        if not files:
            print(f"  [!] No files for '{label}'. Skipping.")
            continue

        sample        = random.sample(files, min(sample_size, len(files)))
        layer_features = {i: [] for i in range(num_layers)}

        for img_path in sample:
            try:
                feats_list = get_features_from_path(
                    img_path, config, model, backbone_type, out_indices, device, preprocessing
                )
                for i, feats in enumerate(feats_list):
                    if pooling_type == "mean":
                        vec = feats.mean(dim=(2, 3)).flatten().cpu().numpy()
                    elif pooling_type == "max":
                        vec = feats.flatten(2).max(-1)[0].cpu().numpy().squeeze(0)
                    elif pooling_type == "mean_std":
                        m   = feats.mean(dim=(2, 3)).flatten().cpu().numpy()
                        s   = feats.std(dim=(2, 3)).flatten().cpu().numpy()
                        vec = np.concatenate([m, s])
                    else:  # flatten
                        vec = feats.mean(1).flatten(1).cpu().numpy().squeeze(0)
                    layer_features[i].append(vec)
            except Exception as e:
                print(f"  [WARN] {img_path}: {e}")

        features_by_class[label] = {
            i: np.array(layer_features[i]) for i in range(num_layers)
        }

    return features_by_class


# ===========================================================================
# t-SNE helpers
# ===========================================================================

def _tsne_scatter(ax, features_by_class, layer_idx, colors_dict, alpha=0.6, s=90):
    """Fit t-SNE on *layer_idx* features and draw scatter on *ax*."""
    valid = [
        lbl for lbl in features_by_class
        if layer_idx in features_by_class[lbl]
        and len(features_by_class[lbl][layer_idx]) > 0
    ]
    if not valid:
        return []

    all_feats = np.vstack([features_by_class[lbl][layer_idx] for lbl in valid])
    coords    = TSNE(n_components=2, random_state=42).fit_transform(all_feats)

    handles, start = [], 0
    for lbl in valid:
        n = len(features_by_class[lbl][layer_idx])
        h = ax.scatter(
            coords[start:start + n, 0], coords[start:start + n, 1],
            label=lbl,
            c=colors_dict.get(lbl, "gray"),
            alpha=alpha, s=s,
        )
        handles.append((h, lbl))
        start += n

    ax.grid(True, linestyle="--", alpha=0.6)
    return handles


def plot_tsne_paired(
    features_single, features_means, colors_dict,
    save_prefix, save_dir, num_layers, alpha=0.6, s=90,
):
    """
    Side-by-side t-SNE: (single images | mean images) with a shared legend.
    Used in RGB mode when the means directory is available.
    """
    for layer_idx in range(num_layers):
        print(f"  t-SNE layer {layer_idx} (paired) ...")
        fig, (ax_l, ax_r) = plt.subplots(
            1, 2, figsize=(36, 20), gridspec_kw={"wspace": 0.06}
        )

        h_l = _tsne_scatter(ax_l, features_single, layer_idx, colors_dict, alpha, s)
        ax_l.set_title('(a) Features from single images',  fontsize=30, fontweight="bold", pad=20)
        ax_l.tick_params(axis='both', which='major', labelsize=20)

        h_r = _tsne_scatter(ax_r, features_means,  layer_idx, colors_dict, alpha, s)
        ax_r.set_title('(b) Features from averaged images', fontsize=30, fontweight="bold", pad=20)
        ax_r.tick_params(axis='both', which='major', labelsize=20)

        seen, handles_leg, labels_leg = set(), [], []
        for h, lbl in h_l + h_r:
            if lbl not in seen:
                seen.add(lbl)
                handles_leg.append(h)
                labels_leg.append(lbl)

        if handles_leg:
            fig.legend(
                handles_leg, labels_leg,
                loc='lower center',
                ncol=min(6, len(labels_leg)),
                bbox_to_anchor=(0.5, -0.1),
                frameon=True,
                fontsize=30,
                markerscale=3
            )

        fig.subplots_adjust(left=0.02, right=0.995, top=0.92, bottom=0.14, wspace=0.06)
        path = os.path.join(save_dir, f"{save_prefix}_layer_{layer_idx}.png")
        fig.savefig(path, dpi=400)
        plt.close()
        print(f"  Saved -> {path}")


def plot_tsne_single(
    features_by_class, colors_dict,
    save_prefix, save_dir, num_layers, alpha=0.6, s=90,
):
    """
    Standard single-panel t-SNE plot per layer.
    Used in Fourier mode (no means available) or as fallback in RGB mode.
    """
    for layer_idx in range(num_layers):
        print(f"  t-SNE layer {layer_idx} ...")
        fig, ax = plt.subplots(figsize=(16, 12))

        handles = _tsne_scatter(ax, features_by_class, layer_idx, colors_dict, alpha, s)
        ax.set_title(f"t-SNE -- layer {layer_idx}", fontsize=20, fontweight="bold")

        if handles:
            ax.legend(
                [h for h, _ in handles], [lbl for _, lbl in handles],
                bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12,
            )

        plt.tight_layout()
        path = os.path.join(save_dir, f"{save_prefix}_layer_{layer_idx}.png")
        plt.savefig(path, dpi=400, bbox_inches="tight")
        plt.close()
        print(f"  Saved -> {path}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    args = get_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load model config
    config_path = os.path.join(PROJECT_ROOT, "configs", f"{args.model}.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = yaml.safe_load(open(config_path))

    use_fourier  = args.use_fourier if hasattr(args, 'use_fourier') else False
    preprocessing = getattr(args, 'preprocessing', 'fourier' if use_fourier else 'none')
    pooling_type = config.get("pooling_type", "mean")
    save_dir     = args.save_dir or os.path.join(SCRIPT_DIR, ".pictures")
    os.makedirs(save_dir, exist_ok=True)

    mode_tag = preprocessing
    prefix   = f"analysis-{args.model}_{mode_tag}"

    print("=" * 70)
    print(f"  Model      : {args.model}")
    print(f"  Preprocessing: {preprocessing}")
    print(f"  Pooling    : {pooling_type}")
    print(f"  Samples    : {args.sample_size} per class")
    print(f"  Save dir   : {save_dir}")
    print("=" * 70)

    # Build backbone
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, backbone_type, channels, scales, out_indices = build_backbone(
        args.model, config, preprocessing, device
    )
    num_layers = len(out_indices)
    print(f"  Backbone   : {backbone_type}")
    print(f"  Layers     : {num_layers}  channels={list(channels)}  scales={list(scales)}")

    # Build data patterns
    data_dir         = const.DATA_DIR
    patterns         = build_data_patterns(data_dir)
    generators_dict  = build_generators_dict(patterns)

    if not any(generators_dict.values()):
        print("\n[ERROR] No images found. Check DATA_DIR in muflow/constants.py.")
        return

    # Common feature-extraction kwargs
    feat_kwargs = dict(
        config=config,
        model=model,
        backbone_type=backbone_type,
        out_indices=out_indices,
        device=device,
        preprocessing=preprocessing,
        sample_size=args.sample_size,
        pooling_type=pooling_type,
    )

    # -----------------------------------------------------------------------
    # Step 1 -- mode-specific visualisation
    # -----------------------------------------------------------------------
    if preprocessing == "none":
        print("\n[1/5] Plotting RGB sample grid ...")
        plot_generators_grid(
            generators_dict,
            os.path.join(save_dir, f"sample_grid.png"),
            cols=args.cols,
        )

    elif preprocessing == "fourier":
        print("\n[1/5] Plotting Fourier magnitude spectrum grid ...")
        plot_fourier_grid(
            generators_dict,
            os.path.join(save_dir, f"{mode_tag}_spectrum_grid.png"),
            config["input_size"], cols=args.cols,
        )
        print("[1b] Plotting RGB vs Fourier comparison ...")
        plot_fourier_comparison(
            generators_dict,
            os.path.join(save_dir, f"{mode_tag}_comparison.png"),
            config["input_size"],
        )

    else:  # srm
        print("\n[1/5] Plotting SRM residual map grid ...")
        plot_srm_grid(
            generators_dict,
            os.path.join(save_dir, f"{mode_tag}_residual_grid.png"),
            config["input_size"], cols=args.cols,
        )
        print("[1b] Plotting RGB vs SRM residual comparison ...")
        plot_srm_comparison(
            generators_dict,
            os.path.join(save_dir, f"{mode_tag}_comparison.png"),
            config["input_size"],
        )

    # -----------------------------------------------------------------------
    # Step 2 -- extract features from single images
    # -----------------------------------------------------------------------
    print(f"[2/5] Extracting {preprocessing!r} features from single images ...")
    features_single = extract_features_by_class(patterns, **feat_kwargs)

    # -----------------------------------------------------------------------
    # Step 3 -- mean images (same preprocessing applied)
    # -----------------------------------------------------------------------
    patterns_means   = build_means_patterns(data_dir)
    generators_means = build_generators_dict(patterns_means)
    has_means        = any(generators_means.values())

    if has_means:
        print(f"[3/5] Plotting mean-image grid ({preprocessing!r}) ...")
        if preprocessing == "none":
            plot_generators_grid(
                generators_means,
                os.path.join(save_dir, f"{mode_tag}_means_grid.png"),
                cols=args.cols,
            )
        elif preprocessing == "fourier":
            plot_fourier_grid(
                generators_means,
                os.path.join(save_dir, f"{mode_tag}_means_spectrum_grid.png"),
                config["input_size"], cols=args.cols,
            )
        else:  # srm
            plot_srm_grid(
                generators_means,
                os.path.join(save_dir, f"{mode_tag}_means_residual_grid.png"),
                config["input_size"], cols=args.cols,
            )

        print(f"[4/5] Extracting {preprocessing!r} features from mean images ...")
        features_means = extract_features_by_class(patterns_means, **feat_kwargs)

        print("[5/5] Running paired t-SNE (single vs mean images) ...")
        plot_tsne_paired(
            features_single, features_means, COLORS,
            f"{prefix}_tsne_paired", save_dir, num_layers,
            alpha=args.tsne_alpha, s=args.tsne_s,
        )
    else:
        print("[3/5] Mean-images directory not found -- skipping paired t-SNE.")
        print("[4/5] -")
        print("[5/5] Running single-panel t-SNE ...")
        plot_tsne_single(
            features_single, COLORS,
            f"{prefix}_tsne", save_dir, num_layers,
            alpha=args.tsne_alpha, s=args.tsne_s,
        )

    print(f"\nDone. All plots saved to: {save_dir}/")


if __name__ == "__main__":
    main()
