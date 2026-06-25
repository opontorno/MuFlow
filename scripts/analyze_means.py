"""
analyze_means.py — Are the PATCH-CENTROID representations of MEAN images
discriminative? (gate for the patch-based redesign, branch `patches`)

Representation logic under test (the new µFlow representation):
    image  →  TOT native patches (crop = input_size, NO resize)
           →  backbone features per patch  →  spatial pool  →  patch vector
           →  MEAN over patches            →  image representation (centroid)

This script computes that representation for the *average* images of every source
and checks whether the resulting centroids are separable (real vs fake), per
backbone layer — both visually (t-SNE) and quantitatively (logistic-probe
accuracy / ROC-AUC via cross-validation).

If the mean-image centroids are NOT separable, the average-image GMM target must
be dropped in favour of a real-single-patch target.

Usage:
    python scripts/analyze_means.py --model_name resnet50 \
        --means-dir <DATA_DIR>/datasets_means/500 --num-patches 16 --sample-size 120
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from muflow import constants as const

ImageFile.LOAD_TRUNCATED_IMAGES = True

PICTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pictures")

# Sources considered REAL (compared after _norm: lowercase, '_'→' '); rest = fake.
REAL_SOURCES = {"ffhq", "celeba hq", "wdf real"}


def _norm(s):
    return s.lower().replace("_", " ").strip()


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", type=str, default="resnet50",
                   help="Backbone name (must have a configs/<model_name>.yaml).")
    p.add_argument("--means-dir", type=str,
                   default=os.path.join(const.DATA_DIR, "datasets_means", "500"),
                   help="Directory with one subfolder of average images per source.")
    p.add_argument("--num-patches", type=int, default=16,
                   help="TOT native patches averaged into one image representation.")
    p.add_argument("--sample-size", type=int, default=120,
                   help="Max mean images sampled per source.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Backbone + per-patch feature extraction (mirrors generate_parameters.py)
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


def make_patch_features(model, backbone_type, out_indices, config, device):
    """Return get_features(patch_array): a native RGB patch (H×W×3, already at
    input_size, NO resize) → list of feature maps, one per requested layer.
    Normalisation matches constants.get_norm_stats (CLIP stats / else ImageNet)."""
    norm_mean, norm_std = const.get_norm_stats(config["backbone_name"])
    mean = torch.tensor(norm_mean).view(1, 3, 1, 1).to(device)
    std = torch.tensor(norm_std).view(1, 3, 1, 1).to(device)
    input_size = config["input_size"]

    def get_features(patch):
        img_t = torch.from_numpy(np.array(patch)).permute(2, 0, 1).unsqueeze(0).float().to(device)
        img_t = (img_t / 255.0 - mean) / std
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
                    x = model.norm(x)[:, 2:, :]
                    N, _, C = x.shape
                    return [x.permute(0, 2, 1).reshape(N, C, input_size // 16, input_size // 16)]
                else:  # CaiT
                    x = model.patch_embed(img_t) + model.pos_embed
                    x = model.pos_drop(x)
                    for i in range(41):
                        x = model.blocks[i](x)
                    x = model.norm(x)
                    N, _, C = x.shape
                    return [x.permute(0, 2, 1).reshape(N, C, input_size // 16, input_size // 16)]
            elif backbone_type == "dino":
                return model.get_intermediate_layers(img_t, n=list(out_indices), reshape=True)
            else:  # clip / cnn
                return model(img_t)

    return get_features


def _pool(feats, pooling_type):
    """Spatial pooling of a (B, C, H, W) feature map → 1-D vector."""
    if pooling_type == "max":
        return feats.flatten(2).max(-1)[0].cpu().numpy().squeeze(0)
    elif pooling_type == "mean_std":
        m = feats.mean(dim=(2, 3)).flatten().cpu().numpy()
        s = feats.std(dim=(2, 3)).flatten().cpu().numpy()
        return np.concatenate([m, s])
    elif pooling_type == "flatten":
        return feats.mean(1).flatten(1).cpu().numpy().squeeze(0)
    else:  # mean
        return feats.mean(dim=(2, 3)).flatten().cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Patch-centroid representation
# ═══════════════════════════════════════════════════════════════════════════

def crop_patches(arr, P, k, rng):
    """k random native P×P crops (no resize). Falls back to a center crop if smaller."""
    H, W = arr.shape[:2]
    if H < P or W < P:
        s = min(H, W)
        y, x = (H - s) // 2, (W - s) // 2
        return [arr[y:y + s, x:x + s]]
    out = []
    for _ in range(k):
        y = rng.integers(0, H - P + 1)
        x = rng.integers(0, W - P + 1)
        out.append(arr[y:y + P, x:x + P])
    return out


def image_centroid(path, P, k, rng, get_feat, pooling, num_layers):
    """Patch-centroid representation of one image, per layer."""
    arr = np.array(Image.open(path).convert("RGB"))
    patches = crop_patches(arr, P, k, rng)
    per_layer = {i: [] for i in range(num_layers)}
    for patch in patches:
        if patch.shape[0] != P or patch.shape[1] != P:
            continue
        feats = get_feat(patch)
        for i in range(len(feats)):
            per_layer[i].append(_pool(feats[i], pooling))
    if not per_layer[0]:
        return None
    return {i: np.mean(np.stack(per_layer[i]), axis=0) for i in range(num_layers)}


def collect(means_dir, sources, P, k, sample_size, rng, get_feat, pooling, num_layers):
    """Return {layer: (X, y, labels)} with y=1 fake / 0 real, labels=source names."""
    by_layer_X = {i: [] for i in range(num_layers)}
    y, labels = [], []
    for src in sources:
        files = glob(os.path.join(means_dir, src, "*.png"))
        if not files:
            continue
        is_real = _norm(src) in REAL_SOURCES
        sample = random.sample(files, min(sample_size, len(files)))
        n_ok = 0
        for f in sample:
            rep = image_centroid(f, P, k, rng, get_feat, pooling, num_layers)
            if rep is None:
                continue
            for i in range(num_layers):
                by_layer_X[i].append(rep[i])
            y.append(0 if is_real else 1)
            labels.append(src)
            n_ok += 1
        print(f"  {src:<34} {'REAL' if is_real else 'fake'}  n={n_ok}")
    return {i: (np.array(by_layer_X[i]), np.array(y), np.array(labels)) for i in range(num_layers)}


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    random.seed(args.seed)
    os.makedirs(PICTURES_DIR, exist_ok=True)

    if not os.path.isdir(args.means_dir):
        raise FileNotFoundError(f"Means directory not found: {args.means_dir}")
    sources = sorted(d for d in os.listdir(args.means_dir)
                     if os.path.isdir(os.path.join(args.means_dir, d)))
    print(f"Sources: {sources}\n")

    config_path = os.path.join(const.WORKING_DIR, "configs", f"{args.model_name}.yaml")
    config = yaml.safe_load(open(config_path))
    config.setdefault("backbone_name", args.model_name)
    P = config["input_size"]
    pooling = config.get("pooling_type", "mean")

    model, backbone_type, out_indices, device = build_backbone(args.model_name, config)
    num_layers = len(out_indices)
    get_feat = make_patch_features(model, backbone_type, out_indices, config, device)

    rng = np.random.default_rng(args.seed)
    print(f"Computing patch-centroid representations "
          f"(patch={P}px, {args.num_patches} patches/img, pooling={pooling})...")
    per_layer = collect(args.means_dir, sources, P, args.num_patches,
                        args.sample_size, rng, get_feat, pooling, num_layers)

    # ── Quantitative separability (real vs fake) + t-SNE, per layer ────────────
    print("\n" + "=" * 64)
    print(f"Real-vs-fake separability of MEAN patch-centroids  ({args.model_name})")
    print("=" * 64)
    cmap = plt.get_cmap("tab20")
    color = {s: cmap(i % 20) for i, s in enumerate(sources)}

    for i in range(num_layers):
        X, y, labels = per_layer[i]
        if len(X) == 0 or len(set(y)) < 2:
            print(f"layer {i}: not enough data"); continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy").mean()
        auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
        print(f"layer {i} (out_idx={out_indices[i]}): probe ACC={acc:.3f}  AUC={auc:.3f}  "
              f"[N={len(X)}, dim={X.shape[1]}]")

        perp = min(30, max(2, len(X) - 1))
        emb = TSNE(n_components=2, random_state=args.seed, perplexity=perp).fit_transform(X)
        fig, ax = plt.subplots(figsize=(11, 9))
        for s in sources:
            m = labels == s
            if not m.any():
                continue
            is_real = _norm(s) in REAL_SOURCES
            kw = dict(edgecolors="k", linewidths=0.3) if is_real else {}
            ax.scatter(emb[m, 0], emb[m, 1], s=70, alpha=0.8,
                       marker="o" if is_real else "x",
                       c=[color[s]], label=("● " if is_real else "✕ ") + s, **kw)
        ax.set_title(f"Mean patch-centroids — {args.model_name} layer {i}  "
                     f"(probe ACC={acc:.2f}, AUC={auc:.2f})", fontweight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        out = os.path.join(PICTURES_DIR, f"patchmeans_{args.model_name}_layer{i}.png")
        fig.savefig(out, dpi=150); plt.close(fig)
        print(f"           → {out}")

    print("\nReals = circles, fakes = crosses. High probe AUC ⇒ mean patch-centroids "
          "are discriminative ⇒ average-image GMM target is viable.")


if __name__ == "__main__":
    main()
