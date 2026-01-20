"""
Fourier Transform Utilities for MuFlow Project

This module provides a unified implementation of Fourier magnitude spectrum calculation
used across the entire project (analysis, training, parameter generation).

The transformation converts RGB images to Fourier magnitude spectra:
    RGB → Grayscale (luminance) → FFT 2D → Magnitude spectrum → Replicate to 3 channels
"""

import numpy as np
import torch
from PIL import Image


def calculate_fourier_magnitude_rgb(image):
    """
    Calculate Fourier magnitude spectrum from RGB image via grayscale conversion.
    
    This function implements the standard approach for converting RGB images to
    Fourier magnitude spectra suitable for CNN input:
    
    1. Convert RGB to grayscale using standard luminance weights (Y = 0.299*R + 0.587*G + 0.114*B)
    2. Compute 2D FFT on grayscale image
    3. Shift zero frequency to center
    4. Compute magnitude spectrum with logarithmic scale
    5. Normalize to [0, 255] range
    6. Replicate to 3 channels for CNN compatibility
    
    Args:
        image: numpy array of shape (H, W, 3) with RGB channels, or PIL Image
        
    Returns:
        magnitude_rgb: numpy array of shape (H, W, 3) with replicated magnitude spectrum,
                      dtype uint8, range [0, 255]
    
    Example:
        >>> from PIL import Image
        >>> img = Image.open('path/to/image.jpg').convert('RGB')
        >>> fourier_spectrum = calculate_fourier_magnitude_rgb(img)
        >>> # fourier_spectrum.shape == (H, W, 3)
    """
    # Convert PIL Image to numpy if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure we have a 3-channel RGB image
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H, W, 3), got {image.shape}")
    
    # Convert RGB to grayscale using standard luminance weights
    # Y = 0.299*R + 0.587*G + 0.114*B (ITU-R BT.601 standard)
    gray = 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]
    
    # Compute 2D FFT on grayscale image
    f = np.fft.fft2(gray)
    
    # Shift zero frequency component to center
    fshift = np.fft.fftshift(f)
    
    # Compute magnitude spectrum with logarithmic scale
    # Log scale helps compress the dynamic range
    magnitude = 20 * np.log(np.abs(fshift) + 1)
    
    # Normalize to [0, 1] range
    magnitude = (magnitude - magnitude.min()) / (magnitude.max() - magnitude.min() + 1e-8)
    
    # Replicate to 3 channels for CNN input (H, W) → (H, W, 3)
    # This maintains compatibility with standard RGB-trained networks
    magnitude_rgb = np.stack([magnitude, magnitude, magnitude], axis=-1)
    
    return magnitude_rgb.astype(np.float32)


class FourierMagnitudeTransform:
    """
    PyTorch/torchvision-compatible transform for Fourier magnitude spectrum.
    
    This transform can be used in torchvision.transforms.Compose pipelines.
    It converts RGB images to Fourier magnitude spectra using the same
    implementation as calculate_fourier_magnitude_rgb().
    
    Example:
        >>> from torchvision import transforms
        >>> transform = transforms.Compose([
        ...     transforms.Resize(256),
        ...     FourierMagnitudeTransform(),
        ...     ToTensorNoScale(),  # Use custom ToTensor that doesn't divide by 255
        ... ])
        >>> img = Image.open('path/to/image.jpg')
        >>> fourier_tensor = transform(img)
    """
    
    def __init__(self):
        """Initialize the transform."""
        pass
    
    def __call__(self, img):
        """
        Apply Fourier magnitude transform to an image.
        
        Args:
            img: PIL Image or numpy array of shape (H, W, 3)
        
        Returns:
            numpy array of shape (H, W, 3) with Fourier magnitude spectrum (float32, [0, 1])
        """
        return calculate_fourier_magnitude_rgb(img)
    
    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensorNoScale:
    """
    Convert numpy array to PyTorch tensor WITHOUT dividing by 255.
    
    This is useful when the input is already normalized to [0, 1] range,
    such as the output of FourierMagnitudeTransform.
    
    Standard transforms.ToTensor() divides by 255 assuming uint8 input [0, 255],
    but this transform keeps the values as-is.
    
    Example:
        >>> transform = transforms.Compose([
        ...     FourierMagnitudeTransform(),  # Output: [0, 1] float32
        ...     ToTensorNoScale(),             # Convert to tensor without scaling
        ... ])
    """
    
    def __call__(self, pic):
        """
        Convert numpy array to tensor without scaling.
        
        Args:
            pic: numpy array of shape (H, W, C)
        
        Returns:
            torch.Tensor of shape (C, H, W)
        """
        if isinstance(pic, np.ndarray):
            # numpy array: (H, W, C) → tensor: (C, H, W)
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()
        else:
            raise TypeError(f'pic should be ndarray. Got {type(pic)}')
    
    def __repr__(self):
        return self.__class__.__name__ + '()'
