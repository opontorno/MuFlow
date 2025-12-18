import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pdb
from tqdm import tqdm, trange
from PIL import Image
import argparse

def get_args():
    parser = argparse.ArgumentParser(description="Generate mean images for FF4ALL and real datasets")
    parser.add_argument('--num_images', type=int, default=1000, help='Number of mean images to generate per class/folder')
    parser.add_argument('--mean_size', type=int, default=500, help='Number of images to average for each mean image')
    parser.add_argument('--output_dir', type=str, default=None, help='Directory to save generated mean images')
    parser.add_argument('--ff4all_glob', type=str, default='/media/orazio_mattia_group/ad4dd/FF4ALL/**/*', help='Glob path for FF4ALL folders')
    parser.add_argument('--ff4all_base', type=str, default='/media/orazio_mattia_group/ad4dd/FF4ALL', help='Base path for FF4ALL images')
    parser.add_argument('--celeba_hq_glob', type=str, default='/media/orazio_mattia_group/ad4dd/celeba_hq/**/**/*.jpg', help='Glob path for CelebA-HQ dataset')
    parser.add_argument('--ffhq_glob', type=str, default='/media/orazio_mattia_group/ad4dd/ffhq/*/*.png', help='Glob path for FFHQ real images')
    parser.add_argument('--real_folders', type=str, nargs='+', default=['ffhq'], help='List of real folders to process')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"/media/orazio_mattia_group/ad4dd/FF4ALL_means/{args.mean_size}/"

    return args

def main():
    args = get_args()
    num_images = args.num_images
    mean_size = args.mean_size
    output_dir = args.output_dir

    ff4all_folders = glob.glob(args.ff4all_glob)
    fake_folders = [fold.split('/')[-1] for fold in ff4all_folders if os.path.isdir(fold)]
    real_folders = args.real_folders

    # print("Processing fake folders...")
    # for folder in tqdm(fake_folders, desc="Fake Folders", leave=True):
    #     if not os.path.exists(os.path.join(output_dir, folder)):
    #         os.makedirs(os.path.join(output_dir, folder), exist_ok=True)
    #     try:
    #         pattern = os.path.join(args.ff4all_base, '**', folder, '*.png')
    #         pattern_list = np.unique(glob.glob(pattern, recursive=True))
    #         np.random.shuffle(pattern_list)
    #         if len(pattern_list) < mean_size:
    #             print(f"Not enough images in folder '{folder}' to create mean images. Skipping.")
    #             continue
    #         images = np.array([plt.imread(img) for img in pattern_list])

    #         bar = tqdm(range(num_images), desc=f"Generating means [{folder}]", leave=False)
    #         for i in bar:
    #             save_path = os.path.join(output_dir, folder, f'{i}.png')
    #             if os.path.exists(save_path):
    #                 bar.set_postfix_str(f"Skipped {i}")
    #                 continue

    #             np.random.shuffle(images)
    #             img_mean = images[:mean_size].mean(0)

    #             plt.imsave(save_path, img_mean)
    #             bar.set_postfix_str(f"Saved {i}")
    #     except Exception as e:
    #         print(f"Error processing folder '{folder}': {e}")

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

        images = np.array([np.array(Image.open(img).resize((256,256))) for img in pattern_list])

        bar = tqdm(range(num_images), desc=f"Generating means [{folder}]", leave=False)
        for i in bar:
            save_path = os.path.join(output_dir, folder, f'{i}.png')
            if os.path.exists(save_path):
                bar.set_postfix_str(f"Skipped {i}")
                continue

            np.random.shuffle(images)
            img_mean = images[:mean_size].mean(0)

            plt.imsave(save_path, img_mean.astype(np.uint8))
            bar.set_postfix_str(f"Saved {i}")

if __name__ == '__main__':
    main()