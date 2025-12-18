import os
import numpy as np
import cv2
from tqdm import tqdm
import glob
import argparse
import pdb

def get_parser():

    parser = argparse.ArgumentParser(description='Calculate Fourier Spectrum of Images')

    parser.add_argument('--size', required=True, type=int, help='Size of the output spectrum')

    args = parser.parse_args()
    return args

def calculate_fourier_spectrum(image_path, size):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    resized_image = cv2.resize(image, size)
    f = np.fft.fft2(resized_image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    phase_spectrum = np.angle(fshift)
    return magnitude_spectrum, phase_spectrum

def process_images(image_paths, size, processed_dirs):    
    for i, image_path in enumerate(tqdm(image_paths)):
        output_dir = os.path.dirname(image_path).replace('dataset_sota', 'dataset_sota_spectrum_{}'.format(size[0]))
        if output_dir not in processed_dirs:
            processed_dirs.append(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir, exist_ok=True)
        
        filename = os.path.basename(image_path)

        if filename.endswith('.png'):
            filename = filename[:-4]
            magnitude_output_path = os.path.join(output_dir, f'magnitude_{filename}.npy')
            # phase_output_path = os.path.join(output_dir, f'phase_{filename}.npy')

            if os.path.exists(magnitude_output_path):# and os.path.exists(phase_output_path):
                continue

            magnitude_spectrum, phase_spectrum = calculate_fourier_spectrum(image_path, size)
            
            np.save(magnitude_output_path, magnitude_spectrum)
            # np.save(phase_output_path, phase_spectrum)

if __name__ == "__main__":

    # input_pattern = '/media/orazio_mattia_group/ad4dd/dataset/train/FF++/**/**/c23/frames/**/*.png'
    train_input_pattern = '/media/orazio_mattia_group/ad4dd/dataset_sota/train/**/**/*.png'
    test_input_pattern = '/media/orazio_mattia_group/ad4dd/dataset_sota/test/forenSynths/**/**/*.png'
    
    args = get_parser()
    resize_size = (args.size, args.size)

    image_paths = np.unique(np.array(glob.glob(train_input_pattern, recursive=True)))
    
    processed_dirs = []
    process_images(image_paths, resize_size, processed_dirs)

    image_paths = np.unique(np.array(glob.glob(test_input_pattern, recursive=True)))
    
    processed_dirs = []
    process_images(image_paths, resize_size, processed_dirs)