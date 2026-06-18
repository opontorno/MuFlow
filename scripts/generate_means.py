"""
generate_means.py — Compute average ("mean") images for ONE dataset.

User utility to prepare the data MuFlow needs for training: in practice you only
need the average images of your REAL dataset (though this script works for any
set of images). Averaging many images of the same source amplifies its
consistent low-level traces, producing the discriminative representations the
GMM is fit on (see scripts/generate_parameters.py).

HOW TO USE
----------
1. Edit INPUT_GLOB below so it matches the images of ONE dataset.
   Images are collected with glob (recursive=True), so ** is supported, e.g.
       INPUT_GLOB = "/path/to/ffhq/**/*.png"
2. Run, choosing a name for the source and the averaging parameters:
       python scripts/generate_means.py --name ffhq --mean-size 500 --num-images 1000

Output layout (matches the MEANS_DIR read by generate_parameters.py):
    <output-root>/<mean-size>/<name>/0.png, 1.png, ...
"""
import argparse
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from muflow import constants as const


# ─────────────────────────────────────────────────────────────────────────────
# EDIT ME — glob matching the images of the dataset you want to average.
# Collected with glob(recursive=True), so ** is supported.
#   e.g. "/path/to/ffhq/**/*.png"
INPUT_GLOB = ""
# ─────────────────────────────────────────────────────────────────────────────


def get_args():
    p = argparse.ArgumentParser(
        description="Compute average images for one dataset (set INPUT_GLOB inside the file).")
    p.add_argument("--name", required=True,
                   help="Source name = output subfolder. For reals we use e.g. 'ffhq'; "
                        "for fakes it would be the generator name.")
    p.add_argument("--mean-size", type=int, default=500,
                   help="Number of images averaged into each mean image (default: 500).")
    p.add_argument("--num-images", type=int, default=1000,
                   help="Number of mean images to produce (default: 1000).")
    p.add_argument("--resize", type=int, default=None,
                   help="Optional square resize (px) applied before averaging (default: none).")
    p.add_argument("--output-root", type=str, default=os.path.join(const.DATA_DIR, "datasets_means"),
                   help="Root output dir; means saved under <output-root>/<mean-size>/<name>/.")
    return p.parse_args()


def collect_images(input_glob):
    """Collect image paths from the user-provided glob."""
    if not input_glob:
        raise ValueError(
            "INPUT_GLOB is empty. Edit scripts/generate_means.py and set INPUT_GLOB to a "
            "glob matching the dataset you want to average, e.g. '/path/to/ffhq/**/*.png'.")
    return np.unique(glob(input_glob, recursive=True))


def generate_means(paths, save_dir, num_images, mean_size, resize=None):
    """Produce `num_images` mean images, each the average of `mean_size` random images."""
    os.makedirs(save_dir, exist_ok=True)

    if len(paths) < mean_size:
        raise ValueError(
            f"Found only {len(paths)} images but --mean-size={mean_size}. "
            f"Need at least {mean_size}; lower --mean-size or check INPUT_GLOB.")

    # Use a pool of up to `num_images` images (mirrors the original behaviour);
    # each mean re-shuffles the pool and averages the first `mean_size`.
    paths = paths.copy()
    np.random.shuffle(paths)
    paths = paths[:num_images]

    resize_to = (resize, resize) if resize is not None else None
    if resize_to is None:
        images = np.array([np.array(Image.open(p).convert("RGB")) for p in paths])
    else:
        images = np.array([np.array(Image.open(p).convert("RGB").resize(resize_to)) for p in paths])

    for i in range(num_images):
        save_path = os.path.join(save_dir, f"{i}.png")
        if os.path.exists(save_path):
            continue
        np.random.shuffle(images)
        img_mean = images[:mean_size].mean(0)
        if img_mean.dtype != np.uint8:
            img_mean = np.clip(img_mean, 0, 255).astype(np.uint8)
        plt.imsave(save_path, img_mean)
        if (i + 1) % 50 == 0 or (i + 1) == num_images:
            print(f"  [{i + 1}/{num_images}] saved")


def main():
    args = get_args()
    paths = collect_images(INPUT_GLOB)
    if len(paths) == 0:
        print(f"No images matched INPUT_GLOB: {INPUT_GLOB!r}")
        return

    save_dir = os.path.join(args.output_root, str(args.mean_size), args.name)
    print(f"Dataset name : {args.name}")
    print(f"Images found : {len(paths)}")
    print(f"Mean size    : {args.mean_size}  |  Num means: {args.num_images}")
    print(f"Saving to    : {save_dir}")

    generate_means(paths, save_dir, args.num_images, args.mean_size, resize=args.resize)
    print("Done!")


if __name__ == "__main__":
    main()
