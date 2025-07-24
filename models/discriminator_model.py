"""
PatchGAN Discriminator for RGB to Multispectral Image Super-Resolution

This module implements a PatchGAN discriminator that evaluates the realism
of generated multispectral images conditioned on RGB inputs.

"""

import torch
import torch.nn as nn
from typing import Tuple


class PatchGANDiscriminator(nn.Module):
  
    def __init__(self, 
                 in_channels: int, 
                 features_d: int = 64,
                 n_layers: int = 3,
                 norm_layer: nn.Module = nn.BatchNorm2d,
                 use_sigmoid: bool = False):
        super().__init__()
        
        if not (isinstance(in_channels, int) and in_channels > 0):
            raise ValueError(f"in_channels must be positive integer, got {in_channels}")
        if not (isinstance(features_d, int) and features_d > 0):
            raise ValueError(f"features_d must be positive integer, got {features_d}")
        if not (isinstance(n_layers, int) and n_layers >= 1):
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        
        self.in_channels = in_channels
        self.features_d = features_d
        self.n_layers = n_layers
        self.use_sigmoid = use_sigmoid
        
        # Build discriminator layers
        layers = []
        
        # First layer: no normalization
        layers.append(nn.Conv2d(in_channels, features_d, kernel_size=4, 
                               stride=2, padding=1, bias=True))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Intermediate layers
        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)  # Cap at 8x base features
            
            layers.extend([
                nn.Conv2d(features_d * nf_mult_prev, features_d * nf_mult,
                         kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(features_d * nf_mult),
                nn.LeakyReLU(0.2, inplace=True)
            ])
        
        # Final layer before output
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        layers.extend([
            nn.Conv2d(features_d * nf_mult_prev, features_d * nf_mult,
                     kernel_size=4, stride=1, padding=1, bias=False),
            norm_layer(features_d * nf_mult),
            nn.LeakyReLU(0.2, inplace=True)
        ])
        
        # Output layer
        layers.append(nn.Conv2d(features_d * nf_mult, 1, kernel_size=4,
                               stride=1, padding=1, bias=True))
        
        if use_sigmoid:
            layers.append(nn.Sigmoid())
        
        self.net = nn.Sequential(*layers)
        
        # Initialize weights
        self.initialize_weights()

    def forward(self, input_rgb: torch.Tensor, 
                target_multispectral: torch.Tensor) -> torch.Tensor:
       
        # Concatenate RGB and multispectral images along channel dimension
        x = torch.cat([input_rgb, target_multispectral], dim=1)
        return self.net(x)

    def get_num_parameters(self) -> int:
       
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_output_shape(self, input_shape: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        
        with torch.no_grad():
            dummy_rgb = torch.zeros(input_shape[0], 3, input_shape[2], input_shape[3])
            dummy_ms = torch.zeros(input_shape[0], self.in_channels - 3, 
                                 input_shape[2], input_shape[3])
            output = self.forward(dummy_rgb, dummy_ms)
            return output.shape

    def initialize_weights(self):
       
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight.data, 1.0, 0.02)
                nn.init.constant_(m.bias.data, 0.0)


class MultiScaleDiscriminator(nn.Module):
  
    
    def __init__(self, 
                 in_channels: int,
                 features_d: int = 64,
                 n_scales: int = 3,
                 n_layers: int = 3):
        super().__init__()
        
        self.n_scales = n_scales
        self.discriminators = nn.ModuleList()
        
        for i in range(n_scales):
            disc = PatchGANDiscriminator(
                in_channels=in_channels,
                features_d=features_d,
                n_layers=n_layers - i  # Fewer layers for lower scales
            )
            self.discriminators.append(disc)
        
        # Downsampling layers for multi-scale input
        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=1, 
                                      count_include_pad=False)

    def forward(self, input_rgb: torch.Tensor, 
                target_multispectral: torch.Tensor) -> list:
      
        results = []
        rgb_scaled = input_rgb
        ms_scaled = target_multispectral
        
        for i, discriminator in enumerate(self.discriminators):
            if i > 0:
                rgb_scaled = self.downsample(rgb_scaled)
                ms_scaled = self.downsample(ms_scaled)
            
            output = discriminator(rgb_scaled, ms_scaled)
            results.append(output)
        
        return results

    def get_num_parameters(self) -> int:
        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)