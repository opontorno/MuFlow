"""
Projection layers for FastFlow model.

This module provides different types of projection layers that can be used
to transform feature maps before feeding them to the normalizing flow.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseProjection(nn.Module, ABC):
    """
    Abstract base class for projection layers.
    All projection layers should inherit from this class.
    """
    
    def __init__(self, in_channels, **kwargs):
        super(BaseProjection, self).__init__()
        self.in_channels = in_channels
    
    @abstractmethod
    def forward(self, x):
        """
        Forward pass of the projection layer.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Output tensor of shape (B, C, H, W)
        """
        pass


class ConvProjection(BaseProjection):
    """
    Convolutional projection using 1x1 convolutions.
    
    Architecture: Conv2d(1x1) -> ReLU -> Conv2d(1x1)
    Maintains spatial dimensions while transforming channels.
    
    Args:
        in_channels: Number of input channels
        hidden_ratio: Ratio for hidden dimension (default: 0.5)
    """
    
    def __init__(self, in_channels, hidden_ratio=0.5, **kwargs):
        super(ConvProjection, self).__init__(in_channels)
        
        hidden_dim = int(in_channels * hidden_ratio)
        
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1, bias=True)
        )
    
    def forward(self, x):
        """
        Apply 1x1 convolutions directly.
        
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        return self.projection(x)


class MLPProjection(BaseProjection):
    """
    MLP-based projection with spatial reshape.
    
    Architecture: Reshape -> Linear -> ReLU -> Linear -> Reshape back
    Processes each spatial location independently using fully connected layers.
    
    Args:
        in_channels: Number of input channels
        hidden_ratio: Ratio for hidden dimension (default: 0.5)
    """
    
    def __init__(self, in_channels, hidden_ratio=0.5, **kwargs):
        super(MLPProjection, self).__init__(in_channels)
        
        hidden_dim = int(in_channels * hidden_ratio)
        
        self.projection = nn.Sequential(
            nn.Linear(in_channels, hidden_dim, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_channels, bias=True)
        )
    
    def forward(self, x):
        """
        Reshape, apply MLP, reshape back.
        
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        B, C, H, W = x.shape
        
        # Reshape to (B, H*W, C)
        x_reshaped = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Apply MLP
        x_projected = self.projection(x_reshaped)
        
        # Reshape back to (B, C, H, W)
        x_out = x_projected.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        return x_out


class AutoEncoderProjection(BaseProjection):
    """
    AutoEncoder-based projection.
    
    Classic autoencoder with encoder-bottleneck-decoder structure.
    Learns a compressed representation and reconstructs the features.
    
    Architecture:
        Encoder: Conv2d(3x3) -> ReLU -> Conv2d(3x3) -> ReLU
        Bottleneck: Conv2d(1x1) to reduce channels
        Decoder: ConvTranspose2d(3x3) -> ReLU -> ConvTranspose2d(3x3)
    
    Args:
        in_channels: Number of input channels
        bottleneck_ratio: Ratio for bottleneck dimension (default: 0.25)
        hidden_ratio: Ratio for hidden layers (default: 0.5)
    """
    
    def __init__(self, in_channels, bottleneck_ratio=0.25, hidden_ratio=0.5, **kwargs):
        super(AutoEncoderProjection, self).__init__(in_channels)
        
        hidden_dim = int(in_channels * hidden_ratio)
        bottleneck_dim = int(in_channels * bottleneck_ratio)
        
        # Encoder: progressively reduce channels
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, bottleneck_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: progressively increase channels back
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_dim, hidden_dim, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1, bias=True)
        )
    
    def forward(self, x):
        """
        Encode to bottleneck and decode back.
        
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W)
        """
        # Encode
        encoded = self.encoder(x)
        
        # Decode
        decoded = self.decoder(encoded)
        
        return decoded


class IdentityProjection(BaseProjection):
    """
    Identity projection (no transformation).
    Useful for ablation studies or when projection is not needed.
    """
    
    def __init__(self, in_channels, **kwargs):
        super(IdentityProjection, self).__init__(in_channels)
    
    def forward(self, x):
        """Return input unchanged."""
        return x


# Factory function to create projection layers
def create_projection_layer(projection_type, in_channels, **kwargs):
    """
    Factory function to create projection layers.
    
    Args:
        projection_type: Type of projection ('conv', 'mlp', 'autoencoder', 'identity')
        in_channels: Number of input channels
        **kwargs: Additional arguments passed to the projection layer
        
    Returns:
        Instance of the requested projection layer
        
    Raises:
        ValueError: If projection_type is not recognized
    """
    projection_map = {
        'conv': ConvProjection,
        'mlp': MLPProjection,
        'autoencoder': AutoEncoderProjection,
        'identity': IdentityProjection
    }
    
    if projection_type not in projection_map:
        raise ValueError(
            f"Unknown projection type: {projection_type}. "
            f"Available types: {list(projection_map.keys())}"
        )
    
    return projection_map[projection_type](in_channels, **kwargs)
