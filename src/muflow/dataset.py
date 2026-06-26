import os
from glob import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import random
import numpy as np
import pandas as pd

from muflow import constants as c
from muflow.attacks import RobustnessAttacks
from muflow.patch_utils import random_patches, repr_patches, make_patch_transform


CSV_PATH = os.path.join(c.WORKING_DIR, 'data', 'dataset_split_rand.csv')


def filter_files_by_csv_split(image_files, is_train, is_val=False):
    """Filter image files based on the train/val/test CSV split."""
    guidance = pd.read_csv(CSV_PATH)
    if is_train:
        split_name = 'val' if is_val else 'train'
        guidance = guidance[guidance['split'] == split_name]
    else:
        guidance = guidance[guidance['split'] == 'test']
    allowed = guidance['path'].to_list()
    mask = np.isin(image_files, allowed)
    return image_files[mask]


class Dataset:
    def __init__(self,
    reals_name,
    input_size=256,
    is_train=True,
    is_val=False,
    attack_type='none',
    attack_params=None,
    num_train_patches=c.PATCH_NUM_TRAIN,
    num_repr_patches=c.PATCH_NUM_REPR,
    debug=False,
    norm_mean=None,
    norm_std=None,
    ):
        self.reals_name = reals_name
        self.is_train = is_train
        self.is_val = is_val
        self.input_size = input_size
        self.attack_type = attack_type
        self.attack_params = attack_params if attack_params is not None else {}
        self.num_train_patches = num_train_patches
        self.num_repr_patches = num_repr_patches
        self.debug = debug
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def create_dataset(self):
        if self.reals_name == 'ffhq':
            root_dir = [f"{c.DATA_DIR}/datasets/ffhq/*"] if self.is_train \
                        else [f"{c.DATA_DIR}/datasets/ffhq/*"] + [f"{c.DATA_DIR}/datasets/WILD/**/**"] + [f"{c.DATA_DIR}/datasets/datasets_DFX/**"] + [f"{c.DATA_DIR}/datasets/celeba_hq/val/**/*"]
            file_pattern = "*.*g"
        elif self.reals_name == 'celeba_hq':
            root_dir = [f"{c.DATA_DIR}/datasets/celeba_hq/train/*"] if self.is_train \
                        else [f"{c.DATA_DIR}/datasets/celeba_hq/val/*", f"{c.DATA_DIR}/datasets/WILD/*"] + [f"{c.DATA_DIR}/datasets/datasets_DFX/"] + [f"{c.DATA_DIR}/datasets/ffhq/"]
            file_pattern = "**/*.*g"
        else:
            raise ValueError(f"Unsupported reals source: {self.reals_name!r}. Choose 'ffhq' or 'celeba_hq'.")

        if not self.is_train:
            print(f'Attack type: {self.attack_type}')
            if self.attack_type != 'none':
                print(f'Attack params: {self.attack_params}')

        return DeepFakeDataset(
            root_dir=root_dir,
            file_pattern=file_pattern,
            input_size=self.input_size,
            is_train=self.is_train,
            is_val=self.is_val,
            reals_name=self.reals_name,
            attack_type=self.attack_type,
            attack_params=self.attack_params,
            num_train_patches=self.num_train_patches,
            num_repr_patches=self.num_repr_patches,
            debug=self.debug,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )


class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, file_pattern, input_size=256, is_train=True, is_val=False, reals_name='ffhq',
                 attack_type='none', attack_params=None, seed=124,
                 num_train_patches=c.PATCH_NUM_TRAIN, num_repr_patches=c.PATCH_NUM_REPR,
                 debug=False, norm_mean=None, norm_std=None):
        self.debug = debug
        self.is_train = is_train
        self.is_val = is_val

        # Patch side = backbone input size (int). No resize is ever applied.
        self.P = input_size if isinstance(input_size, int) else input_size[0]
        self.num_train_patches = num_train_patches
        self.num_repr_patches = num_repr_patches
        self.seed_repr = c.PATCH_SEED

        random.seed(seed)
        np.random.seed(seed)

        # Content-preserving degradations: applied to the WHOLE image at test time
        # (before patch extraction), for the robustness evaluation. Never in training.
        self.attack = RobustnessAttacks(
            attack_type=attack_type if not is_train else 'none',
            **(attack_params if attack_params is not None else {})
        )

        self.transform = make_patch_transform(norm_mean, norm_std)

        self.image_files = [np.unique(np.array(glob(os.path.join(r, file_pattern), recursive=True))) for r in root_dir]
        self.image_files = np.concatenate(self.image_files)

        # Filter files based on CSV split
        self.image_files = filter_files_by_csv_split(self.image_files, is_train, is_val)
        self.classes = np.unique([f.split("/")[-2] for f in self.image_files if (f.split("/")[-3] != "ffhq" and f.split("/")[-4] != "celeba_hq")])
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}
        ood_reals = 'celeba_hq' if reals_name == 'ffhq' else 'ffhq'
        self.class_to_idx[np.str_(ood_reals)] = 99  # Out-of-distribution real class

        self.labels = []
        for image_file in self.image_files:
            if reals_name in image_file:
                self.labels.append(0)
            elif ood_reals in image_file:
                self.labels.append(99)  # Out-of-distribution real class
            else:
                class_name = image_file.split("/")[-2]
                self.labels.append(self.class_to_idx[class_name])

        if not self.is_train:
            min_count = min(sum(1 for label in self.labels if (label != 0 and label != 99)), self.labels.count(0))
            balanced_items = []
            for label in sorted(set(self.labels)):  # Sort labels for consistency
                label_items = [(img, lbl) for img, lbl in zip(self.image_files, self.labels) if lbl == label]
                k = min_count if label == 0 else min_count // len(self.classes)

                random.seed(seed + label)  # Different seed per label
                balanced_items.extend(random.choices(label_items, k=k))

            self.image_files, self.labels = zip(*balanced_items)
            self.image_files = np.array(self.image_files)
            self.labels = np.array(self.labels)

        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(self.image_files))

        self.image_files = np.array(self.image_files)[idx]
        self.labels = np.array(self.labels)[idx]

        if self.debug and self.is_train:
            if not self.is_val:
                print("-" * 40)
                print("-- DEBUG MODE: Using only 100 samples for training ---")
                print("-" * 40)
            self.image_files = self.image_files[:100]
            self.labels = self.labels[:100]

    def _patches(self, image):
        """Return a (k, 3, P, P) tensor of normalised patches for one image."""
        if self.is_train and not self.is_val:
            plist = random_patches(image, self.P, self.num_train_patches)        # random, per epoch
        else:
            plist = repr_patches(image, self.P, self.num_repr_patches, self.seed_repr)  # deterministic
        return torch.stack([self.transform(p).float() for p in plist])

    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]

        image = Image.open(image_file).convert("RGB")
        if not self.is_train:                 # robustness degradation on the whole image
            image = self.attack.apply(image)

        patches = self._patches(image)        # (k, 3, P, P)

        if self.is_train:                     # train & val: patches only (no label)
            return patches
        return patches, label                 # test: patches + label

    def __len__(self):
        return len(self.image_files)
