import argparse
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from muflow import constants as const

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config as cfg


def get_args():
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Compute average images for the real source.")
    p.add_argument("--input", nargs="+", default=cfg.PATH_REAL,
                   help="glob(s) of images to average; defaults to PATH_REAL")
    p.add_argument("--name", type=str, default=cfg.real_tag(cfg.PATH_REAL),
                   help="output subfolder name; defaults to the PATH_REAL tag")
    p.add_argument("--mean-size", type=int, default=cfg.MEAN_SIZE)
    p.add_argument("--num-images", type=int, default=cfg.NUM_MEANS)
    p.add_argument("--resize", type=int, default=None)
    p.add_argument("--output-root", type=str, default=os.path.join(const.DATA_DIR, "datasets_means"))
    return p.parse_args()


def collect_images(input_globs):
    """Collect unique image paths matching one or more globs."""
    files = []
    for g in input_globs:
        files.extend(glob(g, recursive=True))
    return np.unique(files)


def generate_means(paths, save_dir, num_images, mean_size, resize=None):
    """Produce mean images and write them to disk."""
    os.makedirs(save_dir, exist_ok=True)

    if len(paths) < mean_size:
        raise ValueError(
            f"Found only {len(paths)} images but --mean-size={mean_size}. "
            f"Need at least {mean_size}; lower --mean-size or check INPUT_GLOB.")

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
    paths = collect_images(args.input)
    if len(paths) == 0:
        print(f"No images matched: {args.input!r}")
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
