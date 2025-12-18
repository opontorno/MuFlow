import os
from glob import glob
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import random
import io
import numpy as np
import pdb
import cv2
import constants as c
from collections import Counter
from random import choices
import pandas as pd

DATA_DIR = "/media/orazio_mattia_group/ad4dd"
CSV_PATH = '/media/orazio_mattia_group/ad4dd/dataset_split.csv'

class Dataset:
    def __init__(self, 
    dataset_name, 
    reals_name,
    input_size=(224,224), 
    is_train=True,  
    use_fourier=True, 
    test_name="forenSynth",
    reals=None,
    attack_type='none',
    attack_params=None
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
        self.input_size = input_size
        self.use_fourier = use_fourier
        self.attack_type = attack_type
        self.attack_params = attack_params if attack_params is not None else {}

    def create_dataset(self):
        if self.dataset_name == "FF++":            # TODO: sistemare patterns
            root_dir = f"{c.DATA_DIR}/dataset/train/FF++/real" if self.is_train else f"{c.DATA_DIR}/dataset/train/FF++/"
            file_pattern = "**/c23/frames_spectrum_256/**/magnitude*.npy" if self.use_fourier else "**/c23/frames/**/*.png"

            return DeepFakeDataset(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=True,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
                attack_type=self.attack_type,
                attack_params=self.attack_params
            )

        elif self.dataset_name == 'FF4ALL':
            if self.reals_name == 'ffhq':
                test_folders = ['014000', '022000']
                # root_dir = [f"{c.DATA_DIR}/ffhq/*"] if self.is_train \
                root_dir = [f"{c.DATA_DIR}/ffhq/{fol}" for fol in os.listdir(f"{c.DATA_DIR}/ffhq") if fol not in test_folders] if self.is_train \
                            else [f"{c.DATA_DIR}/ffhq/{fol}" for fol in test_folders] + [f"{c.DATA_DIR}/FF4ALL/**/**"]
                file_pattern = "*.png" 
            elif self.reals_name == 'celeba_hq':
                root_dir = [f"{c.DATA_DIR}/celeba_hq/train/*"] if self.is_train else [f"{c.DATA_DIR}/celeba_hq/val/*", f"{c.DATA_DIR}/FF4ALL/*"]
                file_pattern = "**/*.*g" 
            #########################################################check effectivness#############################
            elif self.reals_name == 'ffhq+celeba_hq':
                root_dir = [f"{c.DATA_DIR}/ffhq", f"{c.DATA_DIR}/celeba_hq/*"] if self.is_train else [f"{c.DATA_DIR}/ffhq", f"{c.DATA_DIR}/celeba_hq/*", f"{c.DATA_DIR}/FF4ALL/*"]
                file_pattern = "**/*" 
            ###################################################################################################

            if not self.is_train:
                print(f'Attack type: {self.attack_type}')
                if self.attack_type != 'none':
                    print(f'Attack params: {self.attack_params}')
            
            return DeepFakeDataset(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=True,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
                attack_type=self.attack_type,
                attack_params=self.attack_params
            )
            
        elif self.dataset_name == 'progan':
            if self.test_name == "forenSynth":
                test_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/test/forenSynths/**/**/" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/test/forenSynths/**/**/"
            elif self.test_name == "DiffusionForensics":
                test_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/test/DiffusionForensics/**/**/**" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/test/DiffusionForensics/**/**/**"
            
            train_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/train/**/0_real/" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/train/**/0_real/"
            root_dir = train_dir if self.is_train else test_dir
            
            file_pattern = "*.npy" if self.use_fourier else "*.[pj][pn]g"
        
            return DeepFakeDatasetSota(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=False,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

class DeepFakeDataset(Dataset):
    def __init__(self, root_dir, file_pattern, input_size=(224, 224), is_train=True, 
                 use_valid=False, use_fourier=False, attack_type='none', attack_params=None, seed=124):
        """
        Args:
            root_dir (str): Path to the root folder containing image data.
            input_size (tuple): Size for resizing images.
            is_train (bool): Flag to indicate if the dataset is for training or testing.
        """

        random.seed(seed)
        np.random.seed(seed)

        self.attack = RobustnessAttacks(
            attack_type=attack_type if not is_train else 'none',  # Attacchi solo in test
            **(attack_params if attack_params is not None else {})
        )
        
        if use_fourier:
            self.image_transform = transforms.Compose([transforms.ToTensor()])
        else:
            if is_train:
                self.image_transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.RandomHorizontalFlip(p=0.5),
                    # transforms.RandomResizedCrop(input_size, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
            else:
                self.image_transform = transforms.Compose([
                    transforms.Resize(input_size),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])

        file_pattern = file_pattern 
        self.image_files = [np.unique(np.array(glob(os.path.join(r, file_pattern), recursive=True))) for r in root_dir]
        self.image_files = np.concatenate(self.image_files)

        self.is_train = is_train
        self.use_fourier = use_fourier
        self.classes = np.unique([f.split("/")[-2] for f in self.image_files if (f.split("/")[-3] != "ffhq" and f.split("/")[-4] != "celeba_hq")])
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}
        
        self.labels = [0 if ("ffhq" in image_file or "celeba_hq" in image_file) else self.class_to_idx[image_file.split("/")[-2]] for image_file in self.image_files]
        
        if not self.is_train:      
            min_count = min(sum(1 for label in self.labels if label != 0), self.labels.count(0))
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
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]

        # image = np.load(image_file) if self.use_fourier else Image.open(image_file).convert("RGB")

        if self.use_fourier:
            image = np.load(image_file)
        else:
            image = Image.open(image_file).convert("RGB")
            image = self.attack.apply(image)
        image = self.image_transform(image).float()
        
        if self.is_train:
            return image
        else:
            return image, label

    def __len__(self):
        return len(self.image_files)


class Dataset_celeba:
    def __init__(self, 
    dataset_name, 
    reals_name,
    input_size=(224,224), 
    is_train=True,  
    use_fourier=True, 
    test_name="forenSynth",
    reals=None,
    attack_type='none',
    attack_params=None
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
        self.input_size = input_size
        self.use_fourier = use_fourier
        self.attack_type = attack_type
        self.attack_params = attack_params if attack_params is not None else {}

    def create_dataset(self):
        if self.dataset_name == "FF++":            # TODO: sistemare patterns
            root_dir = f"{c.DATA_DIR}/dataset/train/FF++/real" if self.is_train else f"{c.DATA_DIR}/dataset/train/FF++/"
            file_pattern = "**/c23/frames_spectrum_256/**/magnitude*.npy" if self.use_fourier else "**/c23/frames/**/*.png"

            return DeepFakeDataset(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=True,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
                attack_type=self.attack_type,
                attack_params=self.attack_params
            )

        elif self.dataset_name == 'FF4ALL':
            if self.reals_name == 'ffhq':
                test_folders = ['014000', '022000']
                root_dir = [f"{c.DATA_DIR}/ffhq/{fol}" for fol in os.listdir(f"{c.DATA_DIR}/ffhq") if fol not in test_folders] if self.is_train \
                            else [f"{c.DATA_DIR}/celeba_hq/val/*/**"] + [f"{c.DATA_DIR}/FF4ALL/**/**"] + [f"{c.DATA_DIR}/ffhq/{fol}" for fol in test_folders] 
                file_pattern = "*.*g"
            elif self.reals_name == 'celeba_hq':
                root_dir = [f"{c.DATA_DIR}/celeba_hq/train/*"] if self.is_train else [f"{c.DATA_DIR}/celeba_hq/val/*", f"{c.DATA_DIR}/FF4ALL/*"]
                file_pattern = "**/*.*g" 
            #########################################################check effectivness#############################
            elif self.reals_name == 'ffhq+celeba_hq':
                root_dir = [f"{c.DATA_DIR}/ffhq", f"{c.DATA_DIR}/celeba_hq/*"] if self.is_train else [f"{c.DATA_DIR}/ffhq", f"{c.DATA_DIR}/celeba_hq/*", f"{c.DATA_DIR}/FF4ALL/*"]
                file_pattern = "**/*" 
            ###################################################################################################

            if not self.is_train:
                print(f'Attack type: {self.attack_type}')
                if self.attack_type != 'none':
                    print(f'Attack params: {self.attack_params}')
            
            return DeepFakeDataset_w_celeba(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=True,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
                attack_type=self.attack_type,
                attack_params=self.attack_params
            )
            
        elif self.dataset_name == 'progan':
            if self.test_name == "forenSynth":
                test_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/test/forenSynths/**/**/" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/test/forenSynths/**/**/"
            elif self.test_name == "DiffusionForensics":
                test_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/test/DiffusionForensics/**/**/**" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/test/DiffusionForensics/**/**/**"
            
            train_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/train/**/0_real/" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/train/**/0_real/"
            root_dir = train_dir if self.is_train else test_dir
            
            file_pattern = "*.npy" if self.use_fourier else "*.[pj][pn]g"
        
            return DeepFakeDatasetSota(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=False,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")

class DeepFakeDataset_w_celeba(Dataset):
    def __init__(self, root_dir, file_pattern, input_size=(224, 224), is_train=True, 
                 use_valid=False, use_fourier=False, attack_type='none', attack_params=None, seed=124):
        """
        Args:
            root_dir (str): Path to the root folder containing image data.
            input_size (tuple): Size for resizing images.
            is_train (bool): Flag to indicate if the dataset is for training or testing.
        """

        random.seed(seed)
        np.random.seed(seed)

        self.attack = RobustnessAttacks(
            attack_type=attack_type if not is_train else 'none',  # Attacchi solo in test
            **(attack_params if attack_params is not None else {})
        )
        
        self.image_transform = transforms.Compose([transforms.ToTensor()]) if use_fourier else transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Collect all image paths recursively
        file_pattern = file_pattern 
        self.image_files = [np.unique(np.array(glob(os.path.join(r, file_pattern), recursive=True))) for r in root_dir]
        self.image_files = np.concatenate(self.image_files)

        self.is_train = is_train
        self.use_fourier = use_fourier
        self.classes = np.unique([f.split("/")[-2] for f in self.image_files if (f.split("/")[-3] != "ffhq" and f.split("/")[-4] != "celeba_hq")]+["celeba_hq"])
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}
        
        # self.labels = [0 if ("ffhq" in image_file or "celeba_hq" in image_file) else self.class_to_idx[image_file.split("/")[-2]] for image_file in self.image_files]

        self.labels = []
        for image_file in self.image_files:
            if "ffhq" in image_file:
                self.labels.append(0)
            elif "celeba_hq" in image_file:
                self.labels.append(0)
                # self.labels.append(self.class_to_idx["celeba_hq"])
            else:
                class_name = image_file.split("/")[-2]
                self.labels.append(self.class_to_idx[class_name])


        #Balancing test set
        # if not self.is_train:      
        #     min_count = min(Counter(self.labels).values())
        #     self.image_files, self.labels = zip(*[item for label in set(self.labels) for item in choices([(img, lbl) for img, lbl in zip(self.image_files, self.labels) if lbl == label], k=(min_count if label == 0 else min_count//len(self.classes)))])
        if not self.is_train:      
            min_count = min(sum(1 for label in self.labels if label != 0), self.labels.count(0))
            balanced_items = []
            for label in sorted(set(self.labels)):  # Sort labels for consistency
                label_items = [(img, lbl) for img, lbl in zip(self.image_files, self.labels) if lbl == label]
                k = min_count if label == 0 else min_count // len(self.classes)
                # Re-seed before choices to ensure reproducibility
                random.seed(seed + label)  # Different seed per label
                balanced_items.extend(random.choices(label_items, k=k))
            
            self.image_files, self.labels = zip(*balanced_items)
            self.image_files = np.array(self.image_files)
            self.labels = np.array(self.labels)
        

        # idx = np.random.permutation(len(self.image_files))
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(self.image_files))

        self.image_files = np.array(self.image_files)[idx]
        self.labels = np.array(self.labels)[idx]
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]

        # image = np.load(image_file) if self.use_fourier else Image.open(image_file).convert("RGB")

        if self.use_fourier:
            image = np.load(image_file)
        else:
            image = Image.open(image_file).convert("RGB")
            image = self.attack.apply(image)
        image = self.image_transform(image).float()
        
        if self.is_train:
            return image
        else:
            return image, label

    def __len__(self):
        return len(self.image_files)
    
    

class DeepFakeDatasetSota(Dataset):
    def __init__(self, root_dir, file_pattern, input_size=(224, 224), is_train=True, use_valid=False, use_fourier=False):
        """
        Args:
            root_dir (str): Path to the root folder containing image data.
            input_size (tuple): Size for resizing images.
            is_train (bool): Flag to indicate if the dataset is for training or testing.
        """
        self.image_transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize(384), transforms.ToTensor(), transforms.Normalize([0.0], [1.0]),]) if use_fourier else transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Collect all image paths recursively
        file_pattern = file_pattern 
        self.image_files = np.unique(np.array(glob(os.path.join(root_dir, file_pattern), recursive=True)))
        random.seed(seed)  # Ensures reproducibility
        random.shuffle(self.image_files)
        # random.shuffle(self.image_files)
        # self.image_files = self.image_files[:100]
        
        # Split into 90% train and 10% test if "FF++" is in the root directory path
        if use_valid:
            random.seed(seed)  # Ensures reproducibility
            random.shuffle(self.image_files)
            split_idx = int(0.9 * len(self.image_files))
            train_files = self.image_files[:split_idx]
            test_files = self.image_files[split_idx:]
            self.image_files = train_files if is_train else test_files

        self.is_train = is_train
        self.use_fourier = use_fourier
        self.classes = np.unique([f.split("/")[7] for f in self.image_files if not is_train])
        
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}

    def __getitem__(self, index):
        image_file = self.image_files[index]
        
        image = np.load(image_file) if self.use_fourier else Image.open(image_file).convert("RGB").resize([256,256])
        
        image = self.image_transform(image).float()
        
        if self.is_train:
            return image
        else:
            
            label = 0 if "0_real" in image_file else self.class_to_idx[image_file.split("/")[7]]
            return image, label

    def __len__(self):
        return len(self.image_files)


class Dataset_multi_class:
    def __init__(self, 
    dataset_name, 
    reals_name,
    input_size=(224,224), 
    is_train=True,  
    is_val=False,
    use_fourier=True, 
    test_name="forenSynth",
    reals=None,
    attack_type='none',
    attack_params=None
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
        self.use_fourier = use_fourier
        self.attack_type = attack_type
        self.attack_params = attack_params if attack_params is not None else {}

        self.train_fakes = ['StyleGAN', 'Flux.1.1 Pro']

    def create_dataset(self):
        if self.dataset_name == "FF++":            # TODO: sistemare patterns
            root_dir = f"{c.DATA_DIR}/dataset/train/FF++/real" if self.is_train else f"{c.DATA_DIR}/dataset/train/FF++/"
            file_pattern = "**/c23/frames_spectrum_256/**/magnitude*.npy" if self.use_fourier else "**/c23/frames/**/*.png"

            return DeepFakeDataset(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=True,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
                attack_type=self.attack_type,
                attack_params=self.attack_params
            )

        elif self.dataset_name == 'FF4ALL':
            if self.reals_name == 'ffhq':
                test_folders = ['014000', '022000']
                train_fake_folsers = self.train_fakes
                root_dir = [f"{DATA_DIR}/ffhq/{fol}" for fol in os.listdir(f"{DATA_DIR}/ffhq") if fol not in test_folders] + [f"{DATA_DIR}/FF4ALL/**/{fol}" for fol in train_fake_folsers] if self.is_train \
                            else [f"{DATA_DIR}/ffhq/{fol}" for fol in test_folders] + [f"{DATA_DIR}/FF4ALL/**/{fol}" for fol in os.listdir(f"{DATA_DIR}/FF4ALL/Closed Set")+os.listdir(f"{DATA_DIR}/FF4ALL/Open Set") if fol not in train_fake_folsers]
                file_pattern = "*.png" 
            elif self.reals_name == 'celeba_hq':
                train_fake_folsers = self.train_fakes
                root_dir = [f"{DATA_DIR}/celeba_hq/train/*"] + [f"{DATA_DIR}/FF4ALL/**/{fol}" for fol in train_fake_folsers] if self.is_train \
                            else [f"{DATA_DIR}/celeba_hq/val/*"] + [f"{DATA_DIR}/FF4ALL/**/{fol}" for fol in os.listdir(f"{DATA_DIR}/FF4ALL/Closed Set")+os.listdir(f"{DATA_DIR}/FF4ALL/Open Set") if fol not in train_fake_folsers]
                file_pattern = "**/*.*g" 
            #########################################################check effectivness#############################
            elif self.reals_name == 'ffhq+celeba_hq':
                root_dir = [f"{DATA_DIR}/ffhq", f"{DATA_DIR}/celeba_hq/*"] if self.is_train else [f"{DATA_DIR}/ffhq", f"{DATA_DIR}/celeba_hq/*", f"{DATA_DIR}/FF4ALL/*"]
                file_pattern = "**/*" 
            ###################################################################################################

            print(f'train fake folders: {self.train_fakes}')
            
            return DeepFakeDataset_multi_class(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                is_train=self.is_train,
                is_val=self.is_val,
                use_fourier=self.use_fourier,
                attack_type=self.attack_type,
                attack_params=self.attack_params
            )
            
        elif self.dataset_name == 'progan':
            if self.test_name == "forenSynth":
                test_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/test/forenSynths/**/**/" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/test/forenSynths/**/**/"
            elif self.test_name == "DiffusionForensics":
                test_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/test/DiffusionForensics/**/**/**" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/test/DiffusionForensics/**/**/**"
            
            train_dir = f"{c.DATA_DIR}/datasets_sota_spectrum_256/train/**/0_real/" if self.use_fourier else f"{c.DATA_DIR}/datasets_sota/train/**/0_real/"
            root_dir = train_dir if self.is_train else test_dir
            
            file_pattern = "*.npy" if self.use_fourier else "*.[pj][pn]g"
        
            return DeepFakeDatasetSota(
                root_dir=root_dir,
                file_pattern=file_pattern,
                input_size=self.input_size,
                use_valid=False,
                is_train=self.is_train,
                use_fourier=self.use_fourier,
            )

        else:
            raise ValueError(f"Unsupported dataset: {self.dataset_name}")


class DeepFakeDataset_multi_class(Dataset):
    def __init__(self, root_dir, file_pattern, input_size=(224, 224), is_train=True, 
                 is_val=False, use_fourier=True, attack_type='none', attack_params=None, seed=124):
        """
        Args:
            root_dir (str): Path to the root folder containing image data.
            input_size (tuple): Size for resizing images.
            is_train (bool): Flag to indicate if the dataset is for training or testing.
        """

        random.seed(seed)
        np.random.seed(seed)

        self.attack = RobustnessAttacks(
            attack_type=attack_type if not is_train else 'none',  # Attacchi solo in test
            **(attack_params if attack_params is not None else {})
        )

        self.image_transform = transforms.Compose([transforms.ToTensor()]) if use_fourier else transforms.Compose([
            transforms.Resize(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # Collect all image paths recursively
        self.image_files = [np.unique(np.array(glob(os.path.join(r, file_pattern), recursive=True))) for r in root_dir]
        self.image_files = np.concatenate(self.image_files)

        if is_train:
            guidance = pd.read_csv(CSV_PATH)
            guidance = guidance[guidance['split']==('val' if is_val else 'train')]
            allowed = guidance['path'].to_list()

            mask = np.isin(self.image_files, allowed)
            self.image_files = self.image_files[mask]

        self.use_fourier = use_fourier
        self.classes = np.unique([f.split("/")[-2] for f in self.image_files if (f.split("/")[-3] != "ffhq" and f.split("/")[-4] != "celeba_hq")])
        self.class_to_idx = {cls: idx + 1 for idx, cls in enumerate(self.classes)}
        self.labels = [0 if ("ffhq" in image_file or "celeba_hq" in image_file) else self.class_to_idx[image_file.split("/")[-2]] for image_file in self.image_files]
        # self.labels = [0 if ("ffhq" in image_file or "celeba_hq" in image_file) else 1 for image_file in self.image_files]

        # self.labels = [0 if ("ffhq" in image_file or "celeba_hq" in image_file) else 1 for image_file in self.image_files]
        #Balancing test set
        min_count = min(sum(1 for label in self.labels if label != 0), self.labels.count(0))
        self.image_files, self.labels = zip(*[item for label in set(self.labels) for item in choices([(img, lbl) for img, lbl in zip(self.image_files, self.labels) if lbl == label], k=(min_count if label == 0 else min_count//len(self.classes)))])
        # min_count = min(Counter(self.labels).values())
        # self.image_files, self.labels = zip(*[item for label in set(self.labels) for item in choices([(img, lbl) for img, lbl in zip(self.image_files, self.labels) if lbl == label], k=(min_count))])
        
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(self.image_files))

        self.image_files = np.array(self.image_files)[idx]
        self.labels = np.array(self.labels)[idx]
        
    def __getitem__(self, index):
        image_file = self.image_files[index]
        label = self.labels[index]

        if self.use_fourier:
            image = np.load(image_file)
        else:
            image = Image.open(image_file).convert("RGB")
            image = self.attack.apply(image)
        
        image = self.image_transform(image).float()
        
        return image, label

    def __len__(self):
        return len(self.image_files)


class RobustnessAttacks:
    """Classe per applicare vari attacchi di robustezza alle immagini"""
    
    def __init__(self, attack_type='none', **attack_params):
        """
        Args:
            attack_type: tipo di attacco ('none', 'jpeg', 'gaussian_blur', 'rotation', 
                        'gaussian_noise', 'salt_pepper', 'resize', 'crop')
            attack_params: parametri specifici per l'attacco
        """
        self.attack_type = attack_type
        self.attack_params = attack_params
        
    def apply(self, image):
        """Applica l'attacco all'immagine"""
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
            # Salva in buffer con compressione JPEG
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        else:
            raise ValueError("JPEG compression richiede PIL Image")
    
    def gaussian_blur(self, image):
        """Gaussian blur"""
        kernel_size = self.attack_params.get('kernel_size', 5)
        sigma = self.attack_params.get('sigma', 1.0)
        
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
            return Image.fromarray(blurred)
        else:
            raise ValueError("Gaussian blur richiede PIL Image")
    
    def rotation(self, image):
        """Rotazione dell'immagine"""
        angle = self.attack_params.get('angle', 10)
        
        if isinstance(image, Image.Image):
            return image.rotate(angle, resample=Image.BILINEAR, expand=False)
        else:
            raise ValueError("Rotation richiede PIL Image")
    
    def gaussian_noise(self, image):
        """Aggiunge rumore gaussiano"""
        mean = self.attack_params.get('mean', 0)
        std = self.attack_params.get('std', 0.1)
        
        if isinstance(image, Image.Image):
            img_array = np.array(image).astype(np.float32) / 255.0
            noise = np.random.normal(mean, std, img_array.shape)
            noisy = np.clip(img_array + noise, 0, 1)
            return Image.fromarray((noisy * 255).astype(np.uint8))
        else:
            raise ValueError("Gaussian noise richiede PIL Image")
    
    def salt_pepper_noise(self, image):
        """Aggiunge rumore salt-and-pepper"""
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
            raise ValueError("Salt-pepper noise richiede PIL Image")
    
    def resize_attack(self, image):
        """Resize a dimensione minore e poi ripristina"""
        scale_factor = self.attack_params.get('scale_factor', 0.5)
        
        if isinstance(image, Image.Image):
            orig_size = image.size
            new_size = (int(orig_size[0] * scale_factor), int(orig_size[1] * scale_factor))
            resized = image.resize(new_size, Image.BILINEAR)
            return resized.resize(orig_size, Image.BILINEAR)
        else:
            raise ValueError("Resize attack richiede PIL Image")
    
    def center_crop(self, image):
        """Center crop dell'immagine"""
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
            raise ValueError("Center crop richiede PIL Image")

    def horizontal_flip(self, image):
        """Flip orizzontale dell'immagine"""
        if isinstance(image, Image.Image):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            raise ValueError("Horizontal flip richiede PIL Image")