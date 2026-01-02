import torch, os
import torch.nn as nn
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import torch
from PIL import Image
from PIL import ImageFile
import numpy as np
import random
import timm
import yaml
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture

# === Hyperparameters ===
model_name = "densenet121"

config_path = f"/home/mlitrico/AD4DD/FastFlow/configs/{model_name}.yaml" 
config = yaml.safe_load(open(config_path, "r"))
print("Model config: ", config)


# === Model Setup ===
model = timm.create_model(model_name, pretrained=True, features_only=True, in_chans=3, out_indices=[1, 2, 3])
model.eval()

channels = model.feature_info.channels()
print("Channels: ", channels)
scales = model.feature_info.reduction()
print("Scales: ", scales)

# === Get features ===
def get_features(img):
    img = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float()
    with torch.no_grad():
        features = model(img)
    return features

def get_features_from_path(path): 
    img = np.array(Image.open(path).convert("RGB").resize([config['input_size'],config['input_size']]))
    feature = get_features(img)  # (C, H, W)
    return feature  # (C,)

# def get_features_from_path(path):
#     img = plt.imread(path)[:,:,:3]
#     return get_features(img)


def calculate_fourier_spectrum(image):
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
    #phase_spectrum = np.angle(fshift)
    return magnitude_spectrum


common_path = "/media/orazio_mattia_group/ad4dd/FF4ALL_means/500"
generators = os.listdir(common_path)

patterns_means = {}
for gen in generators:
    patterns_means[gen] = os.path.join(common_path, gen, '*.png')

flux_1_means = glob(patterns_means['Flux.1'])
ffhq_means = glob(patterns_means['ffhq'])
stable_diffusion_35_means = glob(patterns_means['Stable DIffusion 3.5'])
starry_ai_means = glob(patterns_means['Starry AI'])
stylegan3_means = glob(patterns_means['StyleGAN3'])
stable_diffusion_xl_means = glob(patterns_means['Stable Diffusion XL'])
attend_and_excite_means = glob(patterns_means['Stable Diffusion Attend and Excite'])
midjourney_means = glob(patterns_means['Midjourney'])
stable_cascade_means = glob(patterns_means['Stable Cascade'])
flux_1_1_pro_means = glob(patterns_means['Flux.1.1 Pro'])
deep_ai_means = glob(patterns_means['Deep AI'])
stylegan2_means = glob(patterns_means['StyleGAN2'])
celeba_hq_means = glob(patterns_means['celeba_hq'])
hotpot_ai_means = glob(patterns_means['Hotpot AI'])
tencent_hunyuan_means = glob(patterns_means['Tencent Hunyuan'])
dalle_3_means = glob(patterns_means['Dall-E 3'])
stylegan_means = glob(patterns_means['StyleGAN'])
nvidia_sana_pag_means = glob(patterns_means['Nvidia Sana PAG'])

reals = 'ffhq'
assert reals in ['ffhq', 'celeba_hq', 'ffhq+celeba_hq']

if reals == 'ffhq':
    real_sample = ffhq_means
elif reals == 'celeba_hq':
    real_sample = celeba_hq_means
else:
    real_sample = ffhq_means+celeba_hq_means
print(f"Number of real samples: {len(real_sample)}")
print("Extracting features...")
real_features = []
for img_path in real_sample:
    features = get_features_from_path(img_path)
    feats_ = []
    for feats in features:
        feats_.append(feats.mean([2, 3]).flatten().cpu().numpy())
    real_features.append(feats_)
    
gmm = {"real": []}

clf = GaussianMixture()
for i in range(len(real_features[0])):
    real_features_ = np.stack([feats[i] for feats in real_features])
    print(real_features_.shape)
    
    clf.fit(real_features_)
    gmm["real"].append([clf.means_, clf.covariances_])

np.save(f"/home/mlitrico/AD4DD/parameters/gmm_parameters_{model_name}_{reals}_{config['input_size']}.npy", gmm)