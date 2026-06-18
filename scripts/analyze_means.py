"""
analyze_means.py — paired t-SNE feature analysis (single images vs their means).

Reproduces the core analysis of scripts/analysis_WILD.ipynb: for every source it
produces a side-by-side t-SNE plot per layer — LEFT = features of single images,
RIGHT = features of the corresponding mean images — to show how averaging
increases inter-class separability.

  • MEAN images   : read automatically from every subfolder of --means-dir.
  • SINGLE images : taken from SINGLE_IMAGE_GLOBS below (edit it by hand),
                    paired to the means by key (subfolder name).

Feature extraction mirrors scripts/generate_parameters.py exactly (same backbone
forward, same normalisation via constants.get_norm_stats).

If the means directory is missing/empty, it explains how to create it with
scripts/generate_means.py.

Usage:
    python scripts/analyze_means.py --model_name resnet50 \
        --means-dir <DATA_DIR>/means/500 --sample-size 100
"""
import argparse
import os
import random
from glob import glob

import numpy as np
import torch
import timm
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE

from muflow import constants as const

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Plots are saved next to this script, in scripts/.pictures/
PICTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pictures")

# ─────────────────────────────────────────────────────────────────────────────
# EDIT ME — single-image globs per source, paired by key to the means subfolders.
# The key must match a subfolder name inside --means-dir (e.g. 'ffhq').
# Collected with glob(recursive=True), so ** is supported. Example:
#   SINGLE_IMAGE_GLOBS = {
#       "ffhq":      "/path/to/ffhq/**/*.png",
#       "celeba_hq": "/path/to/celeba_hq/**/*.jpg",
#   }
SINGLE_IMAGE_GLOBS = {}
# ─────────────────────────────────────────────────────────────────────────────

_HOWTO_MEANS = (
    "\nAverage images are produced by scripts/generate_means.py. For each real "
    "dataset, set INPUT_GLOB in that script and run, e.g.:\n"
    "    python scripts/generate_means.py --name ffhq --mean-size 500\n"
    "This populates <output-root>/<mean-size>/<name>/ with the mean images."
)


def get_args():
    p = argparse.ArgumentParser(description="Paired t-SNE analysis (single vs means) per layer.")
    p.add_argument("--model_name", type=str, default="resnet50",
                   help="Backbone name (must have a configs/<model_name>.yaml).")
    p.add_argument("--means-dir", type=str, default=os.path.join(const.DATA_DIR, "means", "500"),
                   help="Directory containing one subfolder of average images per source.")
    p.add_argument("--sample-size", type=int, default=100,
                   help="Max number of images sampled per source, per side (default: 100).")
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Backbone + feature extraction (mirrors generate_parameters.py)
# ═══════════════════════════════════════════════════════════════════════════

def build_backbone(model_name, config):
    out_indices = config.get("out_indices", [1, 2, 3])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name in [const.BACKBONE_CAIT, const.BACKBONE_DEIT]:
        model = timm.create_model(model_name, pretrained=True, in_chans=3)
        backbone_type = "cait_deit"
    elif model_name in const.DINO_BACKBONES:
        model = timm.create_model(const.DINO_TIMM_NAMES[model_name], pretrained=True,
                                  img_size=config["input_size"])
        backbone_type = "dino"
    elif model_name in const.CLIP_BACKBONES:
        from muflow.model import CLIPVisualExtractor
        model = CLIPVisualExtractor(model_name, out_block_indices=out_indices)
        backbone_type = "clip"
    else:
        model = timm.create_model(model_name, pretrained=True, features_only=True,
                                  out_indices=out_indices, in_chans=3)
        backbone_type = "cnn"

    model.eval().to(device)
    return model, backbone_type, out_indices, device


def make_get_features(model, backbone_type, out_indices, config, device):
    """Return get_features_from_path(path), matching generate_parameters.py.

    Normalisation uses constants.get_norm_stats (CLIP stats for CLIP, else
    ImageNet) — the same stats the dataset applies at train/eval time.
    """
    norm_mean, norm_std = const.get_norm_stats(config["backbone_name"])
    mean = torch.tensor(norm_mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(norm_std).view(1, 3, 1, 1).to(device)
    input_size = config["input_size"]

    def get_features(img):
        img_t = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img_t = img_t / 255.0
        img_t = (img_t - mean) / std

        with torch.no_grad():
            if backbone_type == "cait_deit":
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
                    x = x.permute(0, 2, 1).reshape(N, C, input_size // 16, input_size // 16)
                    return [x]
                else:  # CaiT
                    x = model.patch_embed(img_t)
                    x = x + model.pos_embed
                    x = model.pos_drop(x)
                    for i in range(41):
                        x = model.blocks[i](x)
                    x = model.norm(x)
                    N, _, C = x.shape
                    x = x.permute(0, 2, 1).reshape(N, C, input_size // 16, input_size // 16)
                    return [x]
            elif backbone_type == "dino":
                return model.get_intermediate_layers(img_t, n=list(out_indices), reshape=True)
            else:  # clip / cnn
                return model(img_t)

    def get_features_from_path(path):
        img = np.array(Image.open(path).convert("RGB").resize(
            [input_size, input_size], Image.BILINEAR))
        return get_features(img)

    return get_features_from_path


def _pool(feats, pooling_type):
    """Spatial pooling of a (B, C, H, W) feature map → 1-D vector."""
    if pooling_type == "mean":
        return feats.mean(dim=(2, 3)).flatten().cpu().numpy()
    elif pooling_type == "max":
        return feats.flatten(2).max(-1)[0].cpu().numpy().squeeze(0)
    elif pooling_type == "mean_std":
        m = feats.mean(dim=(2, 3)).flatten().cpu().numpy()
        s = feats.std(dim=(2, 3)).flatten().cpu().numpy()
        return np.concatenate([m, s])
    else:  # flatten
        return feats.mean(1).flatten(1).cpu().numpy().squeeze(0)


def extract_features_by_class(patterns_dict, get_features_from_path,
                              sample_size, num_layers, pooling_type):
    """Extract pooled features per source. Returns {label: {layer_idx: ndarray}}."""
    features_by_class = {}
    for label, pattern in patterns_dict.items():
        files = glob(pattern, recursive=True)
        if not files:
            print(f"[!] No files found for {label} ({pattern}). Skipping.")
            continue
        sample = random.sample(files, min(sample_size, len(files)))
        layer_features = {i: [] for i in range(num_layers)}
        for img_path in sample:
            try:
                feats = get_features_from_path(img_path)
                for i in range(len(feats)):
                    layer_features[i].append(_pool(feats[i], pooling_type))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
        features_by_class[label] = {i: np.array(layer_features[i]) for i in range(num_layers)}
        print(f"  {label}: {len(layer_features[0])} images")
    return features_by_class


# ═══════════════════════════════════════════════════════════════════════════
# Paired t-SNE per layer (single | means)
# ═══════════════════════════════════════════════════════════════════════════

def _tsne_scatter(ax, features_by_class, layer_idx, colors_dict, alpha=0.7, s=60):
    """Run t-SNE on one side and scatter onto ax. Returns [(handle, label)]."""
    valid = [l for l in features_by_class
             if layer_idx in features_by_class[l] and len(features_by_class[l][layer_idx]) > 0]
    if not valid:
        return []

    all_feats = np.vstack([features_by_class[l][layer_idx] for l in valid])
    perplexity = min(30, max(2, len(all_feats) - 1))
    feats_2d = TSNE(n_components=2, random_state=42, perplexity=perplexity).fit_transform(all_feats)

    handles, start = [], 0
    for label in valid:
        n = len(features_by_class[label][layer_idx])
        pts = feats_2d[start:start + n]
        h = ax.scatter(pts[:, 0], pts[:, 1], label=label, c=[colors_dict.get(label, "gray")],
                       alpha=alpha, s=s, edgecolors="k", linewidths=0.3)
        handles.append((h, label))
        start += n

    ax.grid(True, linestyle="--", alpha=0.5)
    return handles


def plot_tsne_paired(features_single, features_means, colors_dict, num_layers, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for layer_idx in range(num_layers):
        print(f"t-SNE paired — layer {layer_idx}...")
        fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(28, 13),
                                         gridspec_kw={"wspace": 0.08})

        h_left = _tsne_scatter(ax_l, features_single, layer_idx, colors_dict)
        _tsne_scatter(ax_r, features_means, layer_idx, colors_dict)

        ax_l.set_title("Single images", fontsize=18, fontweight="bold")
        ax_r.set_title("Mean images", fontsize=18, fontweight="bold")
        fig.suptitle(f"t-SNE — layer {layer_idx}", fontsize=20, fontweight="bold")

        # Shared legend (union of labels seen on the left, fall back to colors_dict)
        if h_left:
            handles = [h for h, _ in h_left]
            labels = [l for _, l in h_left]
            fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)),
                       fontsize=10, framealpha=0.9, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout()
        out = os.path.join(save_dir, f"tsne_paired_layer{layer_idx}.png")
        plt.savefig(out, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out}")


def main():
    args = get_args()

    # ── Locate means ──────────────────────────────────────────────────────────
    if not os.path.isdir(args.means_dir):
        raise FileNotFoundError(f"Means directory not found: {args.means_dir}" + _HOWTO_MEANS)

    means_sources = sorted(
        d for d in os.listdir(args.means_dir)
        if os.path.isdir(os.path.join(args.means_dir, d))
    )
    if not means_sources:
        raise FileNotFoundError(f"No source subfolders under {args.means_dir}." + _HOWTO_MEANS)

    patterns_means = {name: os.path.join(args.means_dir, name, "*.png") for name in means_sources}
    print(f"Mean sources found in {args.means_dir}: {means_sources}")

    # ── Single-image globs (paired by key) ────────────────────────────────────
    if not SINGLE_IMAGE_GLOBS:
        raise ValueError(
            "SINGLE_IMAGE_GLOBS is empty. Edit scripts/analyze_means.py and add, for each "
            "source you want on the LEFT (single images), a glob keyed by the means "
            "subfolder name, e.g. {'ffhq': '/path/to/ffhq/**/*.png'}.")

    patterns_single = {name: g for name, g in SINGLE_IMAGE_GLOBS.items() if name in means_sources}
    missing_single = [n for n in means_sources if n not in SINGLE_IMAGE_GLOBS]
    if missing_single:
        print(f"[warn] No SINGLE_IMAGE_GLOBS entry for: {missing_single} "
              f"(these will appear only on the means side).")

    # ── Backbone ──────────────────────────────────────────────────────────────
    config_path = os.path.join(const.WORKING_DIR, "configs", f"{args.model_name}.yaml")
    config = yaml.safe_load(open(config_path, "r"))
    config.setdefault("backbone_name", args.model_name)
    pooling_type = config.get("pooling_type", "mean")
    print(f"Model: {args.model_name}  |  pooling: {pooling_type}")

    model, backbone_type, out_indices, device = build_backbone(args.model_name, config)
    num_layers = len(out_indices)
    get_features_from_path = make_get_features(model, backbone_type, out_indices, config, device)

    # ── Extract features ───────────────────────────────────────────────────────
    print("\nExtracting features from SINGLE images...")
    features_single = extract_features_by_class(
        patterns_single, get_features_from_path,
        sample_size=args.sample_size, num_layers=num_layers, pooling_type=pooling_type)

    print("\nExtracting features from MEAN images...")
    features_means = extract_features_by_class(
        patterns_means, get_features_from_path,
        sample_size=args.sample_size, num_layers=num_layers, pooling_type=pooling_type)

    # ── Shared color map over all sources ──────────────────────────────────────
    all_labels = sorted(set(features_single) | set(features_means))
    cmap = plt.get_cmap("tab20")
    colors_dict = {lab: cmap(i % 20) for i, lab in enumerate(all_labels)}

    print("\nPlotting paired t-SNE per layer...")
    plot_tsne_paired(features_single, features_means, colors_dict, num_layers, PICTURES_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
