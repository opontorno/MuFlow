import os
from glob import glob
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import io
import numpy as np
import cv2
from muflow import constants as c
from collections import Counter
from random import choices
import pandas as pd
from tqdm import tqdm

from muflow.fourier_utils import FourierMagnitudeTransform, ToTensorNoScale
from muflow.srm_utils import SRMResidualTransform


DATA_DIR = "/media/orazio_mattia_group/ad4dd"
CSV_PATH = '/media/orazio_mattia_group/ad4dd/dataset_split_rand.csv'


def create_image_transform(input_size, preprocessing="none", is_train=False, use_augs=True):
    """
    Helper function to create image transform pipeline.

    Pipeline structure:
        1. Resize
        2. Augmentations (if is_train and use_augs)
        3. Preprocessing transform (fourier | srm | none)
        4. ToTensor
        5. Normalize (if preprocessing == 'none')

    Args:
        input_size   : Target image size.
        preprocessing: One of 'none', 'fourier', 'srm'.
        is_train     : Whether this is for training (enables augmentations).
        use_augs     : Whether to use data augmentations (only if is_train=True).

    Returns:
        torchvision.transforms.Compose object
    """
    pipeline = []
    
    # 1. Always start with Resize
    pipeline.append(transforms.Resize(input_size))
    
    # 2. Add augmentations if training and use_augs is enabled
    if is_train and use_augs:
        AUGMENTATION_POOL = [
            transforms.RandomApply([transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03)], p=0.5),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.RandomResizedCrop(input_size, scale=(0.92, 1.0), ratio=(0.95, 1.05))], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.5),
            transforms.RandomApply([transforms.RandomRotation(degrees=5)], p=0.5),
        ]
        pipeline.append(RandomApplyAugmentations(AUGMENTATION_POOL, min_augs=1, max_augs=2))
    
    # 3. Add preprocessing transform if enabled
    if preprocessing == "fourier":
        pipeline.append(FourierMagnitudeTransform())
        pipeline.append(ToTensorNoScale())
    elif preprocessing == "srm":
        # Training: seed=None → random 3 filters per image (augmentation)
        # Eval:     seed=0   → fixed 3 filters every call (reproducibility)
        pipeline.append(SRMResidualTransform(seed=None if is_train else 0))
        pipeline.append(ToTensorNoScale())
    else:
        # Standard RGB pipeline (converts [0,255] → [0,1] → ImageNet-normalised)
        pipeline.append(transforms.ToTensor())
        pipeline.append(transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]))
    
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


class RandomApplyAugmentations(torch.nn.Module):
    """
    Randomly applies a random subset of augmentations from a given list.
    """
    def __init__(self, augmentations, min_augs=1, max_augs=None):
        super().__init__()
        self.augmentations = augmentations
        self.min_augs = min_augs
        self.max_augs = max_augs if max_augs is not None else len(augmentations)

    def forward(self, img):
        num_augs = random.randint(self.min_augs, self.max_augs)
        augs = random.sample(self.augmentations, num_augs)
        for aug in augs:
            img = aug(img)
        return img


class Dataset:
    def __init__(self,
    dataset_name,
    reals_name,
    input_size=(224, 224),
    is_train=True,
    is_val=False,
    preprocessing="none",
    test_name="forenSynth",
    attack_type='none',
    attack_params=None,
    use_augs=True,
    debug=False
    ):
        """
        Factory class to create dataset instances based on the dataset name.

        Args:
            dataset_name (str): Name of the dataset to create.
            is_train (bool): Flag indicating if the dataset is for training or testing.
        """
        self.dataset_name = dataset_name
        self.reals_name = reals_name
        self.test_name = test_name  
        self.is_train = is_train
        self.is_val = is_val
        self.input_size = input_size
        self.preprocessing = preprocessing
        self.attack_type = attack_type
        self.attack_params = attack_params if attack_params is not None else {}
        self.use_augs = use_augs
        self.debug = debug
        
    def create_dataset(self):
        if self.dataset_name == "FF++":            # TODO: sistemare patterns
            root_dir = f"{c.DATA_DIR}/dataset/train/FF++/real" if self.is_train else f"{c.DATA_DIR}/dataset/train/FF++/"
            file_pattern = "**/c23/frames_spectrum_256/**/magnitude*.npy" if self.preprocessing == "fourier" else "**/c23/frames/**/*.png"

            return DeepFakeDataset(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=True,
                is_train=self.is_train,
                preprocessing=self.preprocessing,
                attack_type=self.attack_type,
                attack_params=self.attack_params,
                use_augs=self.use_augs,
            )

        elif self.dataset_name == 'WILD':
            if self.reals_name == 'ffhq':
                root_dir = [f"{c.DATA_DIR}/ffhq/*"] if self.is_train \
                            else [f"{c.DATA_DIR}/ffhq/*"] + [f"{c.DATA_DIR}/WILD/**/**"] + [f"{c.DATA_DIR}/datasets_DFX/**"] + [f"{c.DATA_DIR}/celeba_hq/val/**/*"]
                file_pattern = "*.*g" 
            elif self.reals_name == 'celeba_hq':
                root_dir = [f"{c.DATA_DIR}/celeba_hq/train/*"] if self.is_train \
                            else [f"{c.DATA_DIR}/celeba_hq/val/*", f"{c.DATA_DIR}/WILD/*"] + [f"{c.DATA_DIR}/datasets_DFX/"] + [f"{c.DATA_DIR}/ffhq/"]
                file_pattern = "**/*.*g" 

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
                preprocessing=self.preprocessing,
                attack_type=self.attack_type,
                attack_params=self.attack_params,
                use_augs=self.use_augs,
                debug=self.debug
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")


class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, file_pattern, input_size=(224, 224), is_train=True, is_val=False, reals_name='ffhq',
                 preprocessing="none", attack_type='none', attack_params=None, seed=124, use_augs=True, debug=False):
        """
        Args:
            root_dir (str): Path to the root folder containing image data.
            input_size (tuple): Size for resizing images.
            is_train (bool): Flag to indicate if the dataset is for training or testing.
        """

        self.debug = debug
        self.is_train = is_train
        self.is_val = is_val
        self.preprocessing = preprocessing

        random.seed(seed)
        np.random.seed(seed)

        self.attack = RobustnessAttacks(
            attack_type=attack_type if not is_train else 'none',  # Attacchi solo in test
            **(attack_params if attack_params is not None else {})
        )
        
        self.image_transform = create_image_transform(input_size, preprocessing, is_train, use_augs)

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

        # self.labels = [0 if ("ffhq" in image_file or "celeba_hq" in image_file) else self.class_to_idx[image_file.split("/")[-2]] for image_file in self.image_files]
        
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

        # Load image as RGB (preprocessing transform applied in pipeline)
        image = Image.open(image_file).convert("RGB")
        
        # Apply robustness attacks only in test mode and only for plain RGB input
        if not self.is_train and self.preprocessing == "none":
            image = self.attack.apply(image)
        
        # Apply transforms (includes Fourier/SRM if configured)
        image = self.image_transform(image).float()
        
        if self.is_train:
            return image
        else:
            return image, label

    def __len__(self):
        return len(self.image_files)


class RobustnessAttacks:
    """Class to apply various robustness attacks to images"""
    
    def __init__(self, attack_type='none', **attack_params):
        """
        Args:
            attack_type: attack type ('none', 'jpeg', 'gaussian_blur', 'rotation', 
                        'gaussian_noise', 'salt_pepper', 'resize', 'crop')
            attack_params: specific parameters for the attack
        """
        self.attack_type = attack_type
        self.attack_params = attack_params
        
    def apply(self, image):
        """Apply attack to image"""
        if self.attack_type == 'none':
            return image
        elif self.attack_type == 'jpeg':
            return self.jpeg_compression(image)
        elif self.attack_type == 'gaussian_blur':
            return self.gaussian_blur(image)
        elif self.attack_type == 'rotation':
            return self.rotation(image)
        elif self.attack_type == 'gaussian_noise':
            return self.gaussian_noise(image)
        elif self.attack_type == 'salt_pepper':
            return self.salt_pepper_noise(image)
        elif self.attack_type == 'resize':
            return self.resize_attack(image)
        elif self.attack_type == 'crop':
            return self.center_crop(image)
        elif self.attack_type == 'horizontal_flip':
            return self.horizontal_flip(image)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")
    
    def jpeg_compression(self, image):
        """JPEG compression con quality factor specificato"""
        quality = self.attack_params.get('quality', 75)
        
        # Converti PIL in array se necessario
        if isinstance(image, Image.Image):
            # Save to buffer with JPEG compression
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        else:
            raise ValueError("JPEG compression requires PIL Image")
    
    def gaussian_blur(self, image):
        """Gaussian blur"""
        kernel_size = self.attack_params.get('kernel_size', 5)
        sigma = self.attack_params.get('sigma', 1.0)
        
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
            return Image.fromarray(blurred)
        else:
            raise ValueError("Gaussian blur requires PIL Image")
    
    def rotation(self, image):
        """Rotate the image"""
        angle = self.attack_params.get('angle', 10)
        
        if isinstance(image, Image.Image):
            return image.rotate(angle, resample=Image.BILINEAR, expand=False)
        else:
            raise ValueError("Rotation requires PIL Image")
    
    def gaussian_noise(self, image):
        """Add Gaussian noise"""
        mean = self.attack_params.get('mean', 0)
        std = self.attack_params.get('std', 0.1)
        
        if isinstance(image, Image.Image):
            img_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(mean, std, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 1)
            return Image.fromarray((noisy * 255).astype(np.uint8))
        else:
            raise ValueError("Gaussian noise requires PIL Image")
    
    def salt_pepper_noise(self, image):
        """Add salt-and-pepper noise"""
        amount = self.attack_params.get('amount', 0.05)
        
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            # Salt
            num_salt = np.ceil(amount * img_array.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape[:2]]
            img_array[coords[0], coords[1], :] = 255
            
            # Pepper
            num_pepper = np.ceil(amount * img_array.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape[:2]]
            img_array[coords[0], coords[1], :] = 0
            
            return Image.fromarray(img_array)
        else:
            raise ValueError("Salt-pepper noise requires PIL Image")
    
    def resize_attack(self, image):
        """Resize to smaller size and then restore"""
        scale_factor = self.attack_params.get('scale_factor', 0.5)
        
        if isinstance(image, Image.Image):
            orig_size = image.size
            new_size = (int(orig_size[0] * scale_factor), int(orig_size[1] * scale_factor))
            resized = image.resize(new_size, Image.BILINEAR)
            return resized.resize(orig_size, Image.BILINEAR)
        else:
            raise ValueError("Resize attack requires PIL Image")
    
    def center_crop(self, image):
        """Center crop the image"""
        crop_ratio = self.attack_params.get('crop_ratio', 0.8)
        
        if isinstance(image, Image.Image):
            width, height = image.size
            new_width = int(width * crop_ratio)
            new_height = int(height * crop_ratio)
            
            left = (width - new_width) // 2
            top = (height - new_height) // 2
            right = left + new_width
            bottom = top + new_height
            
            cropped = image.crop((left, top, right, bottom))
            return cropped.resize((width, height), Image.BILINEAR)
        else:
            raise ValueError("Center crop requires PIL Image")

    def horizontal_flip(self, image):
        """Horizontal flip of the image"""
        if isinstance(image, Image.Image):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            raise ValueError("Horizontal flip requires PIL Image")


class PairDataset(DeepFakeDataset):
    """
    Dataset that returns pairs of images for contrastive learning.
    
    This dataset extends DeepFakeDataset to return pairs of images
    from the same class (both real). Used for training projection layers
    with contrastive learning.
    
    Returns:
        Tuple of two images (image1, image2) from the same class
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize PairDataset with same arguments as DeepFakeDataset"""
        super(PairDataset, self).__init__(*args, **kwargs)
        
        # Filter only real images (label == 0) for training
        if self.is_train:
            real_mask = np.array(self.labels) == 0
            self.image_files = self.image_files[real_mask]
            self.labels = self.labels[real_mask]
            
            if len(self.image_files) == 0:
                raise ValueError("No real images found in training set for PairDataset")
    
    def __getitem__(self, index):
        """
        Get a pair of images.
        
        Returns both the indexed image and another random image from the same class (real).
        This creates positive pairs for contrastive learning.
        
        Args:
            index: Index of the first image
            
        Returns:
            Tuple (image1, image2): Two images from real class
        """
        # Get first image
        image_file1 = self.image_files[index]
        image1 = Image.open(image_file1).convert("RGB")
        
        # Get second image (different from first, same class)
        # Sample another random real image
        other_index = index
        while other_index == index:
            other_index = np.random.randint(0, len(self.image_files))
        
        image_file2 = self.image_files[other_index]
        image2 = Image.open(image_file2).convert("RGB")
        
        # Apply transforms
        image1 = self.image_transform(image1).float()
        image2 = self.image_transform(image2).float()
        
        return image1, image2
    
    def __len__(self):
        return len(self.image_files)
