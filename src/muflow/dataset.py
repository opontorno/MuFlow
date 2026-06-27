import os
from glob import glob
import torch
from PIL import Image
import random
import numpy as np
import pandas as pd

from muflow import constants as c
from muflow.attacks import RobustnessAttacks
from muflow.patch_utils import random_patches, repr_patches, make_patch_transform


CSV_PATH = os.path.join(c.WORKING_DIR, 'data', 'dataset_split_rand.csv')


def filter_files_by_csv_split(image_files, is_train, is_val=False):
    """Keep only the files belonging to the requested CSV split.
    image_files: array of file paths.
    is_train: True for the train/val splits, False for the test split.
    is_val: True selects 'val', False selects 'train' (only when is_train).
    Returns: filtered array of file paths.
    """
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
        """
        reals_name: real source ('ffhq' or 'celeba_hq').
        input_size: patch side.
        is_train: build the train/val dataset if True, else the test dataset.
        is_val: select the validation split (only meaningful with is_train).
        attack_type: robustness degradation applied at test time.
        attack_params: parameters for the attack.
        num_train_patches: random patches per training image.
        num_repr_patches: deterministic patches per image at val/test.
        debug: cap the training set to 100 samples.
        norm_mean: normalization mean.
        norm_std: normalization std.
        """
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
        """Build and return the configured DeepFakeDataset.
        Returns: DeepFakeDataset instance.
        """
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
        """
        root_dir: list of glob roots to collect images from.
        file_pattern: glob pattern appended to each root.
        input_size: patch side (no resize is applied).
        is_train: train/val mode if True, test mode otherwise.
        is_val: select the validation split.
        reals_name: in-distribution real source.
        attack_type: test-time degradation.
        attack_params: parameters for the attack.
        seed: RNG seed for shuffling and balancing.
        num_train_patches: random patches per training image.
        num_repr_patches: deterministic patches per image at val/test.
        debug: cap the training set to 100 samples.
        norm_mean: normalization mean.
        norm_std: normalization std.
        """
        self.debug = debug
        self.is_train = is_train
        self.is_val = is_val

        self.P = input_size if isinstance(input_size, int) else input_size[0]
        self.num_train_patches = num_train_patches
        self.num_repr_patches = num_repr_patches
        self.seed_repr = c.PATCH_SEED

        random.seed(seed)
        np.random.seed(seed)

        self.attack = RobustnessAttacks(
            attack_type=attack_type if not is_train else 'none',
            **(attack_params if attack_params is not None else {})
        )

        self.transform = make_patch_transform(norm_mean, norm_std)

        self.image_files = [np.unique(np.array(glob(os.path.join(r, file_pattern), recursive=True))) for r in root_dir]
        self.image_files = np.concatenate(self.image_files)

        self.image_files = filter_files_by_csv_split(self.image_files, is_train, is_val)
        self.classes = np.unique([f.split("/")[-2] for f in self.image_files if (f.split("/")[-3] != "ffhq" and f.split("/")[-4] != "celeba_hq")])
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}
        ood_reals = 'celeba_hq' if reals_name == 'ffhq' else 'ffhq'
        self.class_to_idx[np.str_(ood_reals)] = 99

        self.labels = []
        for image_file in self.image_files:
            if reals_name in image_file:
                self.labels.append(0)
            elif ood_reals in image_file:
                self.labels.append(99)
            else:
                class_name = image_file.split("/")[-2]
                self.labels.append(self.class_to_idx[class_name])

        if not self.is_train:
            min_count = min(sum(1 for label in self.labels if (label != 0 and label != 99)), self.labels.count(0))
            balanced_items = []
            for label in sorted(set(self.labels)):
                label_items = [(img, lbl) for img, lbl in zip(self.image_files, self.labels) if lbl == label]
                k = min_count if label == 0 else min_count // len(self.classes)

                random.seed(seed + label)
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
        """Extract and transform the patches representing one image.
        image: PIL.Image.
        Returns: (k, 3, P, P) tensor — random patches for train, deterministic otherwise.
        """
        if self.is_train and not self.is_val:
            plist = random_patches(image, self.P, self.num_train_patches)
        else:
            plist = repr_patches(image, self.P, self.num_repr_patches, self.seed_repr)
        return torch.stack([self.transform(p).float() for p in plist])

    def __getitem__(self, index):
        """Return one dataset item.
        index: sample index.
        Returns: (k, 3, P, P) patches for train/val, or (patches, label) for test.
        """
        image_file = self.image_files[index]
        label = self.labels[index]

        image = Image.open(image_file).convert("RGB")
        if not self.is_train:
            image = self.attack.apply(image)

        patches = self._patches(image)

        if self.is_train:
            return patches
        return patches, label

    def __len__(self):
        return len(self.image_files)
