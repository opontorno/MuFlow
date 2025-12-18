import numpy as np
import matplotlib.pyplot as plt
import glob
import os
import pdb
from tqdm import tqdm
from PIL import Image


num_images = 1000
mean_size = 500

output_dir = f"/media/orazio_mattia_group/ad4dd/datasets_sota_mean/{mean_size}/forensynth/"

models = os.listdir('/media/orazio_mattia_group/ad4dd/datasets_sota/test/forenSynths/')
models = [model for model in models if not model.endswith('.zip')]
folder = 'fake'

for model in models:
    os.makedirs(os.path.join(output_dir, model), exist_ok=True)

    pattern_list = np.unique(glob.glob(f'/media/orazio_mattia_group/ad4dd/datasets_sota/test/forenSynths/{model}/**/1_{folder}/*.png'))
    if len(pattern_list) == 0:
        pattern_list = np.unique(glob.glob(f'/media/orazio_mattia_group/ad4dd/datasets_sota/test/forenSynths/{model}/1_{folder}/*.png'))

    np.random.shuffle(pattern_list)
    pattern_list = pattern_list[:1000]

    images = np.array([np.array(Image.open(img).resize((256,256))) for img in pattern_list])

    for i in tqdm(range(num_images), desc=model):
        if os.path.exists(os.path.join(output_dir, model, f'{i}.png')):
            continue

        np.random.shuffle(images)
        img_mean = images[:mean_size].mean(0)
        
        if not os.path.exists(os.path.join(output_dir, model, f'{i}.png')):
            plt.imsave(os.path.join(output_dir, model, f'{i}.png'), img_mean.astype(np.uint8))