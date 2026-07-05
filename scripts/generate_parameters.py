import os
import argparse
import numpy as np
import yaml
from glob import glob
from PIL import Image
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

from muflow import constants as const
from muflow.model import build_backbone
from muflow.patch_utils import image_centroid

PREFIX = ""
MEANS_DIR = os.path.join(const.DATA_DIR, "datasets_means", "500")

parser = argparse.ArgumentParser(description='Generate GMM parameters for FastFlow')
parser.add_argument('-model', '--model_name', type=str, required=True)
parser.add_argument('-reals', '--reals', type=str, default=const.real_tag(const.PATH_REAL),
                    help="real source name (mean-images subfolder); defaults to the PATH_REAL tag")
args = parser.parse_args()

model_name = args.model_name
reals      = args.reals

config_path = os.path.join(const.WORKING_DIR, "configs", f"{model_name}.yaml")
config = yaml.safe_load(open(config_path))
config.setdefault("backbone_name", model_name)
print("Model config:", config)
print(f"Reals dataset: {reals}")

out_indices  = config.get("out_indices", [1, 2, 3])
pooling_type = config.get("pooling_type", "mean")
n_components = config.get("gmm_n_components", 1)
input_size   = config["input_size"]
num_layers   = len(out_indices)

print(f"out_indices  : {out_indices}")
print(f"pooling_type : {pooling_type}")
print(f"GMM components: {n_components}")

backbone, backbone_type, out_indices, device = build_backbone(model_name, config)
norm_mean, norm_std = const.get_norm_stats(model_name)

MEAN_SIZE = os.path.basename(MEANS_DIR.rstrip("/"))
_HOWTO = (
    f"\nCompute average images first:\n"
    f"    python scripts/generate_means.py --mean_size {MEAN_SIZE}\n"
)

if not os.path.isdir(MEANS_DIR):
    raise FileNotFoundError(f"Average-images directory not found: {MEANS_DIR}" + _HOWTO)

generators = os.listdir(MEANS_DIR)
patterns_means = {gen: os.path.join(MEANS_DIR, gen, "*.png") for gen in generators}

required = ["ffhq", "celeba_hq"] if reals == "ffhq+celeba_hq" else [reals]
missing  = [r for r in required if r not in patterns_means]
if missing:
    raise FileNotFoundError(
        f"No average-images subfolder(s) for {missing} under {MEANS_DIR}. "
        f"Available: {sorted(patterns_means.keys())}" + _HOWTO)

real_sample = []
for r in required:
    real_sample += glob(patterns_means[r])
if not real_sample:
    raise FileNotFoundError(
        f"No *.png average images found for reals='{reals}' under {MEANS_DIR}." + _HOWTO)

print(f"\nReal samples  : {len(real_sample)}")
print(f"Patch-centroid: {const.PATCH_NUM_REPR} patches of {input_size}px (seed {const.PATCH_SEED})")
print("Extracting features...")

real_features = []
for path in tqdm(real_sample):
    img = Image.open(path).convert("RGB")
    real_features.append(
        image_centroid(img, input_size, const.PATCH_NUM_REPR, const.PATCH_SEED,
                       backbone, backbone_type, out_indices, input_size,
                       pooling_type, norm_mean, norm_std, device)
    )

print("Feature extraction complete!")

gmm = {"real": []}
print("\nFitting Gaussian Mixture Models...")
REG_COVAR_SCHEDULE = [1e-6, 1e-4, 1e-2, 1e-1, 1.0]

for i in range(num_layers):
    X = np.stack([feats[i] for feats in real_features]).astype(np.float64)
    n_samples, n_features = X.shape
    print(f"  Layer {i} (out_indices[{i}]={out_indices[i]}): shape {X.shape}")

    fitted = False
    for reg in REG_COVAR_SCHEDULE:
        try:
            clf = GaussianMixture(n_components=n_components, reg_covar=reg, random_state=42)
            clf.fit(X)
            if reg > REG_COVAR_SCHEDULE[0]:
                print(f"    ⚠  Fitted with reg_covar={reg} "
                      f"(n={n_samples} < d={n_features}, covariance is rank-deficient)")
            fitted = True
            break
        except ValueError:
            continue

    if not fitted:
        raise RuntimeError(
            f"GMM fit failed for layer {i} even with reg_covar={REG_COVAR_SCHEDULE[-1]}. "
            f"Consider pooling_type='mean'.")

    gmm["real"].append([clf.means_, clf.covariances_])

output_path = os.path.join(
    const.WORKING_DIR, "parameters",
    f"{PREFIX}{n_components}-gmm_parameters_{config['backbone_name']}"
    f"_indices_{out_indices}_{reals}_{input_size}_{pooling_type}.npy"
)
print(f"\nSaving GMM parameters to: {output_path}")
np.save(output_path, gmm)
print("Done!")
