"""
Residual Attention U-Net Generator for RGB to Multispectral Image Super-Resolution.
This module implements a U-Net architecture with residual connections and attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ResConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid = mid_channels or out_channels
        self.conv1 = nn.Conv2d(in_channels, mid, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid)
        self.conv2 = nn.Conv2d(mid, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
                                  nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
                                  nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        alpha = self.psi(psi)
        return x * alpha

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2),
                                           ResConvBlock(in_ch, out_ch))

    def forward(self, x):
        return self.maxpool_conv(x)

class UpAttention(nn.Module):
    def __init__(self, dec_in_ch, enc_skip_ch, out_ch, bilinear=True, F_int_div=2):
        super().__init__()
        self.bilinear = bilinear
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            g_ch = dec_in_ch
        else:
            self.up = nn.ConvTranspose2d(dec_in_ch, dec_in_ch // 2, kernel_size=2, stride=2)
            g_ch = dec_in_ch // 2
        self.att = AttentionGate(F_g=g_ch, F_l=enc_skip_ch, F_int=enc_skip_ch // F_int_div)
        conv_in = enc_skip_ch + g_ch
        self.conv = ResConvBlock(conv_in, out_ch)

    def forward(self, x_prev_dec, x_skip_enc):
        g = self.up(x_prev_dec)
        dy = x_skip_enc.size(2) - g.size(2)
        dx = x_skip_enc.size(3) - g.size(3)
        if dy != 0 or dx != 0:
            g = F.pad(g, [dx // 2, dx - dx // 2, dy // 2, dy - dy // 2])
        x_att = self.att(g, x_skip_enc)
        x_cat = torch.cat([x_att, g], dim=1)
        return self.conv(x_cat)

class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class ResAttentionUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=31, base_features=64, bilinear=True):
        super().__init__()
        self.inc = ResConvBlock(n_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        self.down4 = Down(base_features * 8, base_features * 16 // (2 if bilinear else 1))

        bf = base_features
        factor = 2 if bilinear else 1
        self.up1 = UpAttention(bf * 16 // factor, bf * 8, bf * 8 // factor, bilinear)
        self.up2 = UpAttention(bf * 8 // factor, bf * 4, bf * 4 // factor, bilinear)
        self.up3 = UpAttention(bf * 4 // factor, bf * 2, bf * 2 // factor, bilinear)
        self.up4 = UpAttention(bf * 2 // factor, bf, bf, bilinear)

        self.outc = OutConv(bf, n_classes)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        u = self.up1(x5, x4)
        u = self.up2(u, x3)
        u = self.up3(u, x2)
        u = self.up4(u, x1)

        return self.tanh(self.outc(u))
