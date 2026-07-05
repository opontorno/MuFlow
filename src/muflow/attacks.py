import io
import random

import cv2
import numpy as np
from PIL import Image


class RobustnessAttacks:
    def __init__(self, attack_type='none', **attack_params):
        self.attack_type = attack_type
        self.attack_params = attack_params

    def apply(self, image):
        """Apply the configured attack to an image."""
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
        elif self.attack_type == 'random_crop':
            return self.random_crop(image)
        elif self.attack_type == 'horizontal_flip':
            return self.horizontal_flip(image)
        else:
            raise ValueError(f"Unknown attack type: {self.attack_type}")

    def jpeg_compression(self, image):
        """JPEG-recompress an image."""
        quality = self.attack_params.get('quality', 75)
        if isinstance(image, Image.Image):
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            return Image.open(buffer).convert('RGB')
        else:
            raise ValueError("JPEG compression requires PIL Image")

    def gaussian_blur(self, image):
        """Gaussian-blur an image."""
        kernel_size = self.attack_params.get('kernel_size', 5)
        sigma = self.attack_params.get('sigma', 1.0)
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            blurred = cv2.GaussianBlur(img_array, (kernel_size, kernel_size), sigma)
            return Image.fromarray(blurred)
        else:
            raise ValueError("Gaussian blur requires PIL Image")

    def rotation(self, image):
        """Rotate an image."""
        angle = self.attack_params.get('angle', 10)
        if isinstance(image, Image.Image):
            return image.rotate(angle, resample=Image.BILINEAR, expand=False)
        else:
            raise ValueError("Rotation requires PIL Image")

    def gaussian_noise(self, image):
        """Add Gaussian noise to an image."""
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
        """Add salt-and-pepper noise to an image."""
        amount = self.attack_params.get('amount', 0.05)
        if isinstance(image, Image.Image):
            img_array = np.array(image)
            num_salt = np.ceil(amount * img_array.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in img_array.shape[:2]]
            img_array[coords[0], coords[1], :] = 255
            num_pepper = np.ceil(amount * img_array.size * 0.5)
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in img_array.shape[:2]]
            img_array[coords[0], coords[1], :] = 0
            return Image.fromarray(img_array)
        else:
            raise ValueError("Salt-pepper noise requires PIL Image")

    def resize_attack(self, image):
        """Down- then up-scale an image to lose high-frequency detail."""
        scale_factor = self.attack_params.get('scale_factor', 0.5)
        if isinstance(image, Image.Image):
            orig_size = image.size
            new_size = (int(orig_size[0] * scale_factor), int(orig_size[1] * scale_factor))
            resized = image.resize(new_size, Image.BILINEAR)
            return resized.resize(orig_size, Image.BILINEAR)
        else:
            raise ValueError("Resize attack requires PIL Image")

    def center_crop(self, image):
        """Center-crop then resize back to original size."""
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

    def random_crop(self, image):
        """Random-crop then resize back to original size."""
        crop_ratio = self.attack_params.get('crop_ratio', 0.9)
        if isinstance(image, Image.Image):
            width, height = image.size
            crop_w = int(width * crop_ratio)
            crop_h = int(height * crop_ratio)
            left = random.randint(0, width - crop_w)
            top = random.randint(0, height - crop_h)
            cropped = image.crop((left, top, left + crop_w, top + crop_h))
            return cropped.resize((width, height), Image.BILINEAR)
        else:
            raise ValueError("Random crop requires PIL Image")

    def horizontal_flip(self, image):
        """Horizontally flip an image."""
        if isinstance(image, Image.Image):
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        else:
            raise ValueError("Horizontal flip requires PIL Image")
