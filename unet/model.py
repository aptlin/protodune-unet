""" Full assembly of the parts to form the complete network """

import torch.nn as nn
from .fragments import DoubleConv, Up, Down, OutConv


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, feature_dim=64, is_bilinear=False):
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = is_bilinear

        self.inc = DoubleConv(n_channels, feature_dim)
        self.down1 = Down(feature_dim, 2 * feature_dim)
        self.down2 = Down(2 * feature_dim, 4 * feature_dim)
        self.down3 = Down(4 * feature_dim, 8 * feature_dim)
        self.down4 = Down(8 * feature_dim, 8 * feature_dim)
        self.up1 = Up(16 * feature_dim, 4 * feature_dim, is_bilinear)
        self.up2 = Up(8 * feature_dim, 2 * feature_dim, is_bilinear)
        self.up3 = Up(4 * feature_dim, feature_dim, is_bilinear)
        self.up4 = Up(2 * feature_dim, feature_dim, is_bilinear)
        self.outc = OutConv(feature_dim, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
