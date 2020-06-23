import torch.nn as nn
import torchvision


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def double_conv(in_planes, out_planes, mid_planes=None, batch_norm=False):
    """double convolution layers and keep dimensions"""
    if batch_norm is False:
        if mid_planes is None:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, 3, padding=1),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, mid_planes, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_planes, out_planes, 3, padding=1),
                nn.ReLU(inplace=True)
            )
    else:
        if mid_planes is None:
            return nn.Sequential(
                nn.Conv2d(in_planes, out_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=out_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes, out_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=out_planes),
                nn.ReLU(inplace=True)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_planes, mid_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=mid_planes),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_planes, out_planes, 3, padding=1),
                nn.BatchNorm2d(num_features=out_planes),
                nn.ReLU(inplace=True)
            )


class ResNetConv(nn.Module):
    """resnet18 architecture with n blocks"""

    def __init__(self, n_blocks=4):
        super(ResNetConv, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.n_blocks = n_blocks

    def forward(self, x):
        n_blocks = self.n_blocks
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        if n_blocks >= 1:
            x = self.resnet.layer1(x)
        if n_blocks >= 2:
            x = self.resnet.layer2(x)
        if n_blocks >= 3:
            x = self.resnet.layer3(x)
        if n_blocks >= 4:
            x = self.resnet.layer4(x)

        return x
