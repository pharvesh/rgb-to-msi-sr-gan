"""
Residual Attention U-Net Generator for RGB to Multispectral Image Super-Resolution

This module implements a U-Net architecture with residual connections and attention
for generating high-quality multispectral images from RGB inputs.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ResidualConvBlock(nn.Module):
    
    def __init__(self, in_channels: int, out_channels: int, 
                 mid_channels: Optional[int] = None):
        super().__init__()
        
        if mid_channels is None:
            mid_channels = out_channels

        # Main convolutional path
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, kernel_size=3, 
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
       
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Residual connection
        out = self.relu(out)
        
        return out


class AttentionGate(nn.Module):
    
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
     
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi_input = self.relu(g1 + x1)
        alpha = self.psi(psi_input)
        return x * alpha


class DownBlock(nn.Module):
   
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualConvBlock(in_channels, out_channels)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpAttentionBlock(nn.Module):
   
    
    def __init__(self, dec_in_ch: int, enc_skip_ch: int, out_ch: int, 
                 bilinear: bool = True, F_int_div: int = 2):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            g_channels = dec_in_ch
            conv_in_channels = enc_skip_ch + g_channels
        else:
            self.up = nn.ConvTranspose2d(dec_in_ch, dec_in_ch // 2, 
                                       kernel_size=2, stride=2)
            g_channels = dec_in_ch // 2
            conv_in_channels = enc_skip_ch + g_channels

        self.att = AttentionGate(F_g=g_channels, F_l=enc_skip_ch, 
                               F_int=enc_skip_ch // F_int_div)
        self.conv = ResidualConvBlock(conv_in_channels, out_ch)

    def forward(self, x_prev_dec: torch.Tensor, 
                x_skip_enc: torch.Tensor) -> torch.Tensor:
       
        g = self.up(x_prev_dec)

        # Handle spatial dimension mismatch
        diffY = x_skip_enc.size()[2] - g.size()[2]
        diffX = x_skip_enc.size()[3] - g.size()[3]
        if diffY != 0 or diffX != 0:
            g = F.pad(g, [diffX // 2, diffX - diffX // 2,
                         diffY // 2, diffY - diffY // 2])

        # Apply attention gate
        x_att = self.att(g, x_skip_enc)

        # Concatenate and convolve
        x_cat = torch.cat([x_att, g], dim=1)
        return self.conv(x_cat)


class OutputConv(nn.Module):
   
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ResAttentionUNet(nn.Module):
   
    
    def __init__(self, n_channels: int, n_classes: int, 
                 base_features: int = 64, bilinear: bool = True):
        super().__init__()
        
        # Input validation
        if not (isinstance(n_channels, int) and n_channels > 0):
            raise ValueError(f"n_channels must be positive integer, got {n_channels}")
        if not (isinstance(n_classes, int) and n_classes > 0):
            raise ValueError(f"n_classes must be positive integer, got {n_classes}")
        if not (isinstance(base_features, int) and base_features > 0):
            raise ValueError(f"base_features must be positive integer, got {base_features}")

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        
        bf = base_features
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = ResidualConvBlock(n_channels, bf)
        self.down1 = DownBlock(bf, bf * 2)
        self.down2 = DownBlock(bf * 2, bf * 4)
        self.down3 = DownBlock(bf * 4, bf * 8)
        self.down4 = DownBlock(bf * 8, bf * 16 // factor)

        # Decoder with Attention Gates
        self.up1 = UpAttentionBlock(bf * 16 // factor, bf * 8, 
                                  bf * 8 // factor, bilinear=bilinear)
        self.up2 = UpAttentionBlock(bf * 8 // factor, bf * 4, 
                                  bf * 4 // factor, bilinear=bilinear)
        self.up3 = UpAttentionBlock(bf * 4 // factor, bf * 2, 
                                  bf * 2 // factor, bilinear=bilinear)
        self.up4 = UpAttentionBlock(bf * 2 // factor, bf, 
                                  bf, bilinear=bilinear)
        
        # Output layer
        self.outc = OutputConv(bf, n_classes)
        self.tanh = nn.Tanh()  # Output activation for [-1, 1] range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Encoder path
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # Decoder path with attention
        u = self.up1(x5, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)
        
        # Output
        logits = self.outc(u)
        output = self.tanh(logits)
        
        return output

    def get_num_parameters(self) -> int:
       
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def initialize_weights(self):
      
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)