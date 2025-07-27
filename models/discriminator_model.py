"""
PatchGAN Discriminator for RGB to Multispectral Image Super-Resolution
"""

import torch
import torch.nn as nn
from typing import Tuple


import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=34, features_d=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, features_d, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d, features_d * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 2, features_d * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 4, features_d * 8, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=1, padding=1, bias=False),
        )

    def forward(self, input_rgb, target_hs):
        x = torch.cat([input_rgb, target_hs], dim=1)
        return self.net(x)

        
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
