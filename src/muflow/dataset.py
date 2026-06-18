import os
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import numpy as np
import pandas as pd

from muflow import constants as c
from muflow.attacks import RobustnessAttacks


CSV_PATH = os.path.join(c.WORKING_DIR, 'data', 'dataset_split_rand.csv')


def create_image_transform(input_size, is_train=False,
                           use_augs=True, apply_affine_aug=None, affine_prob=0.5,
                           norm_mean=None, norm_std=None):
    """
    Helper function to create image transform pipeline.

    Pipeline:
        1. RandomAffine (translate + scale, NO rotation, fill=0) — only if use_augs=True and is_train,
           unless apply_affine_aug overrides explicitly (used to disable on val set).
        2. Resize (downsample AFTER spatial transforms to minimise interpolation damage).
        3. ToTensor + Normalize.
    """
    _affine = (is_train and use_augs) if apply_affine_aug is None else apply_affine_aug
    mean = norm_mean if norm_mean is not None else [0.485, 0.456, 0.406]
    std  = norm_std  if norm_std  is not None else [0.229, 0.224, 0.225]

    pipeline = []

    if _affine:
        pipeline.append(
            transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=0,
                    translate=(0.20, 0.20),
                    scale=(0.8, 1.0),
                    fill=0,
                    interpolation=transforms.InterpolationMode.BILINEAR
                )],
                p=affine_prob,
            )
        )

    pipeline.append(transforms.Resize(input_size))
    pipeline.append(transforms.ToTensor())
    pipeline.append(transforms.Normalize(mean, std))

    return transforms.Compose(pipeline)


def filter_files_by_csv_split(image_files, is_train, is_val=False):
    """
    Helper function to filter image files based on CSV split.
    
    Args:
        image_files: Array of image file paths
        is_train: Whether this is training data
        is_val: Whether this is validation data (only used if is_train=True)
    
    Returns:
        Filtered array of image file paths
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
    input_size=(224, 224),
    is_train=True,
    is_val=False,
    attack_type='none',
    attack_params=None,
    use_augs=True,
    affine_prob=0.5,
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
        self.use_augs = use_augs
        self.affine_prob = affine_prob
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
            use_augs=self.use_augs,
            debug=self.debug,
            affine_prob=self.affine_prob,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )


class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, file_pattern, input_size=(224, 224), is_train=True, is_val=False, reals_name='ffhq',
                 attack_type='none', attack_params=None, seed=124, use_augs=True,
                 debug=False, affine_prob=0.5, norm_mean=None, norm_std=None):
        self.debug = debug
        self.is_train = is_train
        self.is_val = is_val

        random.seed(seed)
        np.random.seed(seed)

        self.attack = RobustnessAttacks(
            attack_type=attack_type if not is_train else 'none',  # Attacchi solo in test
            **(attack_params if attack_params is not None else {})
        )
        
        # Val set (is_train=True, is_val=True) is used for threshold computation:
        # disable RandomAffineAug so the loss distribution matches the test set
        # (no augmentation at test time). With flatten pooling the spatial
        # perturbation inflates val std enormously, making the threshold too wide.
        self.image_transform = create_image_transform(
            input_size, is_train, use_augs,
            apply_affine_aug=False if is_val else None,
            affine_prob=affine_prob,
            norm_mean=norm_mean,
            norm_std=norm_std,
        )

        file_pattern = file_pattern 
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
                print("-"*40)
                print("-- DEBUG MODE: Using only 100 samples for training ---")
                print("-"*40)
            self.image_files = self.image_files[:100]
            self.labels = self.labels[:100]
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]

        image = Image.open(image_file).convert("RGB")

        if not self.is_train:
            image = self.attack.apply(image)

        image = self.image_transform(image).float()
        
        if self.is_train:
            return image
        else:
            return image, label

    def __len__(self):
        return len(self.image_files)

