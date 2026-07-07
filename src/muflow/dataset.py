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


def _glob_many(patterns, kind):
    """Glob a list of patterns; raise if a non-empty pattern list yields no files."""
    if not patterns:
        return []
    all_files = set()
    for g in patterns:
        matched = glob(g, recursive=True)
        if not matched:
            raise FileNotFoundError(
                f"No {kind} images found for glob: {g!r}\n"
                f"Check DATA_DIR and the {kind} paths in config.py.")
        all_files.update(matched)
    return sorted(all_files)


def filter_files_by_csv_split(image_files, is_train, is_val=False):
    """Keep only the files belonging to the requested CSV split."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(
            f"{CSV_PATH} not found. Run scripts/generate_csv.py first "
            "(after configuring PATH_REAL/PATH_REAL_OOD/PATH_FAKE in config.py).")
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
    real_paths,
    fake_paths=None,
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
        self.real_paths = real_paths
        self.fake_paths = fake_paths if fake_paths is not None else []
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
        """Build and return the configured DeepFakeDataset."""
        if not self.is_train:
            print(f'Attack type: {self.attack_type}')
            if self.attack_type != 'none':
                print(f'Attack params: {self.attack_params}')

        return DeepFakeDataset(
            real_paths=self.real_paths,
            fake_paths=self.fake_paths,
            input_size=self.input_size,
            is_train=self.is_train,
            is_val=self.is_val,
            attack_type=self.attack_type,
            attack_params=self.attack_params,
            num_train_patches=self.num_train_patches,
            num_repr_patches=self.num_repr_patches,
            debug=self.debug,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )


class DeepFakeDataset(Dataset):
    def __init__(self, real_paths, fake_paths=None, input_size=256, is_train=True, is_val=False,
                 attack_type='none', attack_params=None, seed=124,
                 num_train_patches=c.PATCH_NUM_TRAIN, num_repr_patches=c.PATCH_NUM_REPR,
                 debug=False, norm_mean=None, norm_std=None):
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

        real_files = _glob_many(real_paths, "real")
        fake_files = _glob_many(fake_paths, "fake")
        real_set = set(real_files)

        self.image_files = np.array(real_files + fake_files)
        self.image_files = filter_files_by_csv_split(self.image_files, is_train, is_val)

        if len(self.image_files) == 0:
            split_name = ('val' if is_val else 'train') if is_train else 'test'
            raise RuntimeError(
                f"No images left after filtering by the '{split_name}' split in {CSV_PATH}. "
                "Re-run scripts/generate_csv.py after configuring config.py's PATH_* patterns.")

        self.classes = np.unique([f.split("/")[-2] for f in self.image_files if f not in real_set])
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}

        self.labels = []
        for image_file in self.image_files:
            if image_file in real_set:
                self.labels.append(0)
            else:
                class_name = image_file.split("/")[-2]
                self.labels.append(self.class_to_idx[class_name])

        if not self.is_train:
            min_count = min(sum(1 for label in self.labels if label != 0), self.labels.count(0))
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
        """Extract and transform the patches representing one image."""
        if self.is_train and not self.is_val:
            plist = random_patches(image, self.P, self.num_train_patches)
        else:
            plist = repr_patches(image, self.P, self.num_repr_patches, self.seed_repr)
        return torch.stack([self.transform(p).float() for p in plist])

    def __getitem__(self, index):
        """Return one dataset item."""
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
