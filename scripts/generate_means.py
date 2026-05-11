import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pdb
from tqdm import tqdm, trange
from PIL import Image
import argparse
from muflow.constants import DATA_DIR

def get_args():
    parser = argparse.ArgumentParser(description="Generate mean images for WILD and real datasets")
    parser.add_argument('--num_images', type=int, default=1000, help='Number of mean images to generate per class/folder')
    parser.add_argument('--mean_size', type=int, default=500, help='Number of images to average for each mean image')
    parser.add_argument('--resize_means_dim', type=int, default=None, help='Dimension to resize mean images to (if None, no resizing)')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save generated mean images')
    parser.add_argument('--data_root', type=str, default=DATA_DIR,
                        help='Root data directory. Override to use a different dataset, e.g. the '
                             'pre-aligned dataset: --data_root /media/.../ad4dd/aligned')
    parser.add_argument('--wild_glob', type=str, default=None, help='Override glob path for WILD folders')
    parser.add_argument('--wild_base', type=str, default=None, help='Override base path for WILD images')
    parser.add_argument('--datasets_dfx_glob', type=str, default=None, help='Override glob path for datasets_DFX folders')
    parser.add_argument('--datasets_dfx_base', type=str, default=None, help='Override base path for datasets_DFX images')
    parser.add_argument('--celeba_hq_glob', type=str, default=None, help='Override glob path for CelebA-HQ dataset')
    parser.add_argument('--ffhq_glob', type=str, default=None, help='Override glob path for FFHQ real images')
    parser.add_argument('--real_folders', type=str, nargs='+', default=['ffhq', 'celeba_hq'], help='List of real folders to process')
    parser.add_argument('--input_dir', type=str, default=None,
                        help='Optional: path to a folder OR glob pattern containing images (.png/.jpg/.jpeg) to process in generic mode')
    args = parser.parse_args()

    # Resolve glob defaults from data_root (allows --data_root to change everything at once)
    root = args.data_root
    if args.wild_glob       is None: args.wild_glob       = os.path.join(root, 'WILD', '**', '*')
    if args.wild_base       is None: args.wild_base       = os.path.join(root, 'WILD')
    if args.datasets_dfx_glob is None: args.datasets_dfx_glob = os.path.join(root, 'datasets_DFX', '*')
    if args.datasets_dfx_base is None: args.datasets_dfx_base = os.path.join(root, 'datasets_DFX')
    if args.celeba_hq_glob    is None: args.celeba_hq_glob    = os.path.join(root, 'celeba_hq', '**', '**', '*.jpg')
    if args.ffhq_glob         is None: args.ffhq_glob         = os.path.join(root, 'ffhq', '*', '*.png')

    if args.output_dir is None:
        tag = 'aligned' if root != DATA_DIR else 'WILD'
        args.output_dir = os.path.join(DATA_DIR, f"{tag}_means/{args.mean_size}/")

    return args


def collect_images_from_dir(input_dir):
    extensions = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.jpeg', '*.JPEG']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
    return np.unique(image_paths)


def collect_images_from_input(input_path_or_glob):
    valid_ext = {'.png', '.jpg', '.jpeg'}

    if os.path.isdir(input_path_or_glob):
        return collect_images_from_dir(input_path_or_glob)

    matched_paths = glob.glob(input_path_or_glob, recursive=True)
    image_paths = []

    for path in matched_paths:
        if os.path.isdir(path):
            image_paths.extend(collect_images_from_dir(path))
        elif os.path.isfile(path):
            ext = os.path.splitext(path)[1].lower()
            if ext in valid_ext:
                image_paths.append(path)

    return np.unique(image_paths)


def generate_means_from_paths(pattern_list, save_dir, num_images, mean_size, resize_to=None):
    os.makedirs(save_dir, exist_ok=True)

    if len(pattern_list) < mean_size:
        print(f"Not enough images in '{save_dir}' to create mean images. Skipping.")
        return

    np.random.shuffle(pattern_list)

    if resize_to is None:
        images = np.array([np.array(Image.open(img).convert('RGB')) for img in pattern_list])
    else:
        images = np.array([np.array(Image.open(img).convert('RGB').resize(resize_to)) for img in pattern_list])

    bar = tqdm(range(num_images), desc=f"Generating means [{os.path.basename(save_dir)}]", leave=False)
    for i in bar:
        save_path = os.path.join(save_dir, f'{i}.png')
        if os.path.exists(save_path):
            bar.set_postfix_str(f"Skipped {i}")
            continue

        np.random.shuffle(images)
        img_mean = images[:mean_size].mean(0)

        if img_mean.dtype != np.uint8:
            img_mean = np.clip(img_mean, 0, 255).astype(np.uint8)

        plt.imsave(save_path, img_mean)
        bar.set_postfix_str(f"Saved {i}")

def main():
    args = get_args()
    num_images = args.num_images
    mean_size = args.mean_size
    output_dir = args.output_dir

    if args.input_dir is not None:
        if args.output_dir is not None:
            generic_save_dir = args.output_dir
        else:
            generic_folder_name = os.path.basename(os.path.normpath(args.input_dir))
            generic_save_dir = os.path.join(output_dir, generic_folder_name)

        pattern_list = collect_images_from_input(args.input_dir)
        if len(pattern_list) == 0:
            print(f"No PNG/JPG images found for input path/glob: {args.input_dir}")
            return

        print(f"Processing generic folder: {args.input_dir}")
        generate_means_from_paths(
            pattern_list=pattern_list,
            save_dir=generic_save_dir,
            num_images=num_images,
            mean_size=mean_size,
            resize_to=args.resize_means_dim
        )
        return

    ff4all_folders = glob.glob(args.wild_glob)
    fake_folders = [fold.split('/')[-1] for fold in ff4all_folders if os.path.isdir(fold)]
    
    # Add datasets_DFX folders
    datasets_dfx_folders = glob.glob(args.datasets_dfx_glob)
    dfx_fake_folders = [fold.split('/')[-1] for fold in datasets_dfx_folders if os.path.isdir(fold)]
    fake_folders.extend(dfx_fake_folders)
    
    real_folders = args.real_folders

    print("Processing fake folders...")
    for folder in tqdm(fake_folders, desc="Fake Folders", leave=True):
        if not os.path.exists(os.path.join(output_dir, folder)):
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
        try:
            # Try WILD path first
            pattern = os.path.join(args.wild_base, '**', folder, '*.png')
            pattern_list = np.unique(glob.glob(pattern, recursive=True))
            
            # If not found in WILD, try datasets_DFX
            if len(pattern_list) == 0:
                pattern = os.path.join(args.datasets_dfx_base, folder, '*.png')
                pattern_list = np.unique(glob.glob(pattern, recursive=True))
            np.random.shuffle(pattern_list)
            if len(pattern_list) < mean_size:
                print(f"Not enough images in folder '{folder}' to create mean images. Skipping.")
                continue
            generate_means_from_paths(
                pattern_list=pattern_list,
                save_dir=os.path.join(output_dir, folder),
                num_images=num_images,
                mean_size=mean_size,
                resize_to=args.resize_means_dim
            )
        except Exception as e:
            print(f"Error processing folder '{folder}': {e}")

    print("Processing real folders...")
    for folder in tqdm(real_folders, desc="Real Folders", leave=True):
        if not os.path.exists(os.path.join(output_dir, folder)):
            os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

        if folder == 'celeba_hq':
            pattern_list = np.unique(glob.glob(args.celeba_hq_glob, recursive=True))
        elif folder == 'ffhq':
            pattern_list = np.unique(glob.glob(args.ffhq_glob))
        else:
            print(f"Unknown real folder: {folder}, skipping.")
            continue

        if len(pattern_list) == 0:
            print(f"No images found for real folder {folder}")
            continue

        np.random.shuffle(pattern_list)
        pattern_list = pattern_list[:num_images]
        if len(pattern_list) < mean_size:
            print(f"Not enough images in real folder '{folder}' to create mean images. Skipping.")
            continue

        generate_means_from_paths(
            pattern_list=pattern_list,
            save_dir=os.path.join(output_dir, folder),
            num_images=num_images,
            mean_size=mean_size,
            resize_to=args.resize_means_dim
        )

if __name__ == '__main__':
    main()