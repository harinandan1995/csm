import torch
from src.nnutils.blocks import *


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(UNet, self).__init__()

        features = init_features
        self.encoder1 = double_conv(in_channels, features)
        self.encoder2 = double_conv(features, features * 2)
        self.encoder3 = double_conv(features * 2, features * 4)
        self.encoder4 = double_conv(features * 4, features * 8)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = double_conv(features * 8, features * 16)

        self.decoder4 = double_conv(features * 16, features * 8)
        self.decoder3 = double_conv(features * 8, features * 4)
        self.decoder2 = double_conv(features * 4, features * 2)
        self.decoder1 = double_conv(features * 2, features)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upsample(bottleneck)
        #dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upsample(dec4)
        #dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upsample(dec3)
        #dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upsample(dec2)
        #dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv(dec1)
