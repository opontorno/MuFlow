import argparse
import os
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from muflow import constants as const


INPUT_GLOB = ""


def get_args():
    """Parse command-line arguments.
    Returns: argparse.Namespace.
    """
    p = argparse.ArgumentParser(
        description="Compute average images for one dataset (set INPUT_GLOB inside the file).")
    p.add_argument("--name", required=True,
                   help="Source name = output subfolder.")
    p.add_argument("--mean-size", type=int, default=500,
                   help="Number of images averaged into each mean image.")
    p.add_argument("--num-images", type=int, default=1000,
                   help="Number of mean images to produce.")
    p.add_argument("--resize", type=int, default=None,
                   help="Optional square resize (px) applied before averaging.")
    p.add_argument("--output-root", type=str, default=os.path.join(const.DATA_DIR, "datasets_means"),
                   help="Root output dir; means saved under <output-root>/<mean-size>/<name>/.")
    return p.parse_args()


def collect_images(input_glob):
    """Collect image paths matching a glob.
    input_glob: glob pattern (recursive).
    Returns: unique array of file paths.
    """
    if not input_glob:
        raise ValueError(
            "INPUT_GLOB is empty. Edit scripts/generate_means.py and set INPUT_GLOB to a "
            "glob matching the dataset you want to average, e.g. '/path/to/ffhq/**/*.png'.")
    return np.unique(glob(input_glob, recursive=True))


def generate_means(paths, save_dir, num_images, mean_size, resize=None):
    """Produce mean images and write them to disk.
    paths: array of source image paths.
    save_dir: output directory.
    num_images: number of mean images to produce.
    mean_size: images averaged into each mean.
    resize: optional square resize (px) before averaging.
    """
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
