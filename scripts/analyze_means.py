"""
analyze_means.py — Are the PATCH-CENTROID representations of MEAN images
discriminative? (gate for the patch-based redesign, branch `patches`)

Representation under test (the µFlow representation):
    image  →  k native patches (crop = input_size, NO resize)
           →  backbone features per patch  →  spatial pool
           →  MEAN over patches  →  image centroid

Computes that centroid for every mean image, then checks real/fake
separability per layer — quantitatively (logistic probe, 5-fold CV) and
visually (t-SNE scatter: circles = real, crosses = fake).

Usage:
    python scripts/analyze_means.py --model_name resnet50 \
        --means-dir <DATA_DIR>/datasets_means/500 --num-patches 16 --sample-size 120
"""
import argparse
import os
import random
from glob import glob

import numpy as np
import yaml
import matplotlib.pyplot as plt
from PIL import Image, ImageFile
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from muflow import constants as const
from muflow.model import build_backbone
from muflow.patch_utils import image_centroid

ImageFile.LOAD_TRUNCATED_IMAGES = True

PICTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pictures")

# Sources considered REAL (normalised: lowercase, '_'→' '); rest = fake.
REAL_SOURCES = {"ffhq", "celeba hq"}


def _norm(s: str) -> str:
    return s.lower().replace("_", " ").strip()


def get_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model_name", type=str, default="resnet50",
                   help="Backbone name (must have a configs/<model_name>.yaml).")
    p.add_argument("--means-dir", type=str,
                   default=os.path.join(const.DATA_DIR, "datasets_means", "500"),
                   help="Directory with one subfolder of average images per source.")
    p.add_argument("--num-patches", type=int, default=16,
                   help="Native patches averaged into one image centroid.")
    p.add_argument("--sample-size", type=int, default=120,
                   help="Max mean images sampled per source.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Data collection
# ════════════════════════════════════════════════════════════════════════════

def collect(means_dir, sources, P, k, seed, sample_size,
            backbone, backbone_type, out_indices, pooling, norm_mean, norm_std, device):
    """Return {layer_idx: (X, y, source_labels)} — y=1 fake / 0 real."""
    num_layers   = len(out_indices)
    by_layer_X   = {i: [] for i in range(num_layers)}
    y, labels    = [], []

    for src in sources:
        files = glob(os.path.join(means_dir, src, "*.png"))
        if not files:
            continue
        is_real = _norm(src) in REAL_SOURCES
        sample  = random.sample(files, min(sample_size, len(files)))
        n_ok    = 0
        for f in sample:
            img = Image.open(f).convert("RGB")
            rep = image_centroid(img, P, k, seed,
                                 backbone, backbone_type, out_indices, P,
                                 pooling, norm_mean, norm_std, device)
            for i in range(num_layers):
                by_layer_X[i].append(rep[i])
            y.append(0 if is_real else 1)
            labels.append(src)
            n_ok += 1
        print(f"  {src:<34} {'REAL' if is_real else 'fake'}  n={n_ok}")

    return {i: (np.array(by_layer_X[i]), np.array(y), np.array(labels))
            for i in range(num_layers)}


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

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
    P        = config["input_size"]
    pooling  = config.get("pooling_type", "mean")

    backbone, backbone_type, out_indices, device = build_backbone(args.model_name, config)
    num_layers = len(out_indices)
    norm_mean, norm_std = const.get_norm_stats(args.model_name)

    print(f"Computing patch-centroid representations "
          f"(patch={P}px, {args.num_patches} patches/img, pooling={pooling})...")
    per_layer = collect(args.means_dir, sources, P, args.num_patches, args.seed,
                        args.sample_size, backbone, backbone_type, out_indices,
                        pooling, norm_mean, norm_std, device)

    # ── Separability (logistic probe) + t-SNE, per layer ────────────────────
    print("\n" + "=" * 64)
    print(f"Real-vs-fake separability of MEAN patch-centroids  ({args.model_name})")
    print("=" * 64)
    cmap  = plt.get_cmap("tab20")
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
        emb  = TSNE(n_components=2, random_state=args.seed, perplexity=perp).fit_transform(X)

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
