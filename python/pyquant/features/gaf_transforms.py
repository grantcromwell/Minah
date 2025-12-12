import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
import math

class GAFTransform(nn.Module):
    """
    Gramian Angular Field (GAF) transformation module for time series data.
    Converts 1D time series into 2D images using Gramian Angular Summation/Difference Fields.
    """
    def __init__(self, method: str = 'summation', scale: float = 1.0):
        """
        Initialize GAF transform.
        
        Args:
            method: 'summation' for GASF or 'difference' for GADF
            scale: Scaling factor for the output values
        """
        super().__init__()
        self.method = method.lower()
        self.scale = scale
        
        if self.method not in ['summation', 'difference']:
            raise ValueError("method must be either 'summation' (GASF) or 'difference' (GADF)")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply GAF transformation to input time series.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            
        Returns:
            GAF images of shape (batch_size, n_features, seq_len, seq_len)
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D input (batch, seq_len, n_features), got {x.dim()}D")
            
        batch_size, seq_len, n_features = x.shape
        
        # Rescale to [-1, 1] for angular encoding
        x_min = x.min(dim=1, keepdim=True)[0]
        x_max = x.max(dim=1, keepdim=True)[0]
        x_scaled = 2 * (x - x_min) / (x_max - x_min + 1e-8) - 1
        
        # Clamp to avoid numerical instability with arccos
        x_scaled = torch.clamp(x_scaled, -0.9999, 0.9999)
        
        # Convert to angular space
        phi = torch.arccos(x_scaled)  # (batch_size, seq_len, n_features)
        
        # Create outer product of angles
        phi_i = phi.unsqueeze(2)  # (batch, seq_len, 1, n_features)
        phi_j = phi.unsqueeze(1)  # (batch, 1, seq_len, n_features)
        
        # Compute GAF matrix
        if self.method == 'summation':
            # GASF: Gramian Angular Summation Field
            gaf = torch.cos(phi_i + phi_j)
        else:
            # GADF: Gramian Angular Difference Field
            gaf = torch.sin(phi_i - phi_j)
        
        # Scale the output
        gaf = gaf * self.scale
        
        # Reorder dimensions to (batch_size, n_features, seq_len, seq_len)
        gaf = gaf.permute(0, 3, 1, 2)
        
        return gaf

class MultiChannelGAF(nn.Module):
    """
    Multi-channel GAF transformation that processes multiple time series
    and stacks their GAF representations as channels.
    """
    def __init__(self, method: str = 'summation', scale: float = 1.0):
        super().__init__()
        self.gaf = GAFTransform(method=method, scale=scale)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply multi-channel GAF transformation.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, n_features)
            
        Returns:
            Stacked GAF images of shape (batch_size, n_features, seq_len, seq_len)
        """
        return self.gaf(x)

class GAFCNNBlock(nn.Module):
    """
    CNN block for processing GAF images.
    Can be used as a feature extractor before the transformer.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN block.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor after CNN processing
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.activation(x)
        
        # Pooling and dropout
        x = self.pool(x)
        x = self.dropout(x)
        
        return x

class GAFEncoder(nn.Module):
    """
    Complete GAF encoder that combines GAF transformation and CNN feature extraction.
    """
    def __init__(
        self,
        in_features: int,
        gaf_method: str = 'summation',
        hidden_dims: list = [64, 128, 256],
        kernel_size: int = 3,
        dropout: float = 0.1,
        output_dim: Optional[int] = None
    ):
        super().__init__()
        self.in_features = in_features
        self.gaf = MultiChannelGAF(method=gaf_method)
        
        # Create CNN blocks
        self.cnn_blocks = nn.ModuleList()
        in_channels = in_features  # Each feature becomes a channel
        
        for hidden_dim in hidden_dims:
            self.cnn_blocks.append(
                GAFCNNBlock(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=kernel_size,
                    dropout=dropout
                )
            )
            in_channels = hidden_dim
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Optional output projection
        self.output_proj = None
        if output_dim is not None:
            self.output_proj = nn.Linear(hidden_dims[-1], output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the GAF encoder.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, in_features)
            
        Returns:
            Encoded features of shape (batch_size, output_dim) or (batch_size, hidden_dims[-1])
        """
        # Apply GAF transformation
        # x shape: (batch_size, seq_len, in_features) -> (batch_size, in_features, seq_len, seq_len)
        x_gaf = self.gaf(x)
        
        # Process through CNN blocks
        for cnn_block in self.cnn_blocks:
            x_gaf = cnn_block(x_gaf)
        
        # Global average pooling
        # x_gaf shape: (batch_size, hidden_dims[-1], h, w) -> (batch_size, hidden_dims[-1])
        x_pooled = self.global_pool(x_gaf).squeeze(-1).squeeze(-1)
        
        # Optional projection
        if self.output_proj is not None:
            x_pooled = self.output_proj(x_pooled)
        
        return x_pooled

class HybridTransformerModel(nn.Module):
    """
    Hybrid model that combines GAF+CNN feature extraction with a Transformer encoder.
    """
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        output_dim: int = 3,
        gaf_hidden_dims: list = [64, 128],
        use_gaf: bool = True,
        use_transformer: bool = True,
        gaf_method: str = 'summation'
    ):
        super().__init__()
        self.use_gaf = use_gaf
        self.use_transformer = use_transformer
        
        # GAF-based feature extractor
        if use_gaf:
            self.gaf_encoder = GAFEncoder(
                in_features=input_dim,
                gaf_method=gaf_method,
                hidden_dims=gaf_hidden_dims,
                output_dim=d_model // 2 if use_transformer else d_model
            )
        
        # Raw feature embedding (bypassing GAF)
        self.feature_proj = nn.Linear(input_dim, d_model // 2 if use_gaf and use_transformer else d_model)
        
        # Transformer encoder
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output heads
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mo
