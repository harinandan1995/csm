import torch
from src.nnutils.blocks import *


class UNet(nn.Module):
    """
    Module to use UNet with different layers
    """

    def __init__(self, in_channels=3, out_channels=1, features=None, batch_norm=False):
        """
        Construct the architecture of UNet

        :param in_channels: scalar (default 3 for GRB image)
        :param out_channels: scalar (default 1)
        :param features: list at most 5 elements (default [32,64,128,256,512])
                   if the list has more than 5 elements, we only keep the first 5
                   if the list has less than 5 elements, we extend the later layers with double features
        ：param batch_norm: bool, whether we have batch normalization (default: false)
        """
        super(UNet, self).__init__()

        if features is None:
            features = [32, 64, 128, 256, 512]
        if len(features) > 5:
            features = features[:5]
        elif len(features) < 5:
            while len(features) != 5:
                features.append(features[-1] * 2)

        self.encoder1 = double_conv(in_channels, features[0], batch_norm=batch_norm)
        self.encoder2 = double_conv(features[0], features[1], batch_norm=batch_norm)
        self.encoder3 = double_conv(features[1], features[2], batch_norm=batch_norm)
        self.encoder4 = double_conv(features[2], features[3], batch_norm=batch_norm)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = double_conv(features[3], features[4], batch_norm=batch_norm)

        self.decoder4 = double_conv(features[4] + features[3], features[3], batch_norm=batch_norm)
        self.decoder3 = double_conv(features[3] + features[2], features[2], batch_norm=batch_norm)
        self.decoder2 = double_conv(features[2] + features[1], features[1], batch_norm=batch_norm)
        self.decoder1 = double_conv(features[1] + features[0], features[0], batch_norm=batch_norm)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(
            in_channels=features[0], out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        """
        pass the data forward the UNet

        :param x: [D X W X H X C] or [D X W X H X 1] tensor
        :return :  [D X W X H X 1] tensor
        """
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upsample(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upsample(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upsample(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upsample(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)
