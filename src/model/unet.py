import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn

from src.nnutils.blocks import conv2d, upconv2d, net_init


class UNet(nn.Module):
    """
    UNet model with skip connections
    """
    
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.modules.normalization.GroupNorm):
        """
        :param input_nc: Number of input channels
        :param output_nc: Number of output channels
        :param num_downs: Number of down samplings. If |num_downs| == 7, image of size 128x128 will become of size 1x1
        :param ngf: Number of features of first downsampling. For each downsampling the number of features increase by 2 times.
        """

        super(UNet, self).__init__()

        if num_downs >= 5:
            ngf_max = ngf*8
        else:
            ngf_max = ngf*pow(2, num_downs - 2)

        # construct unet structure
        all_blocks = []
        self.inner_most_block = unet_block = UnetSkipConnectionConcatBlock(ngf_max, ngf_max, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
        all_blocks.append(unet_block)


        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionConcatBlock(ngf_max, ngf_max, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            all_blocks.append(unet_block)

        for i in range(min(3, num_downs - 2)):
            unet_block = UnetSkipConnectionConcatBlock(ngf_max // pow(2,i+1), ngf_max // pow(2,i), input_nc=None, submodule=unet_block, norm_layer=norm_layer)
            all_blocks.append(unet_block)

        unet_block = UnetSkipConnectionConcatBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)
        all_blocks.append(unet_block)

        self.model = unet_block
        self.all_blocks = all_blocks
        net_init(self.model)

    def forward(self, input):
        
        return self.model(input)

    def get_inner_most(self, ):
        
        return self.inner_most_block

    def get_all_block(self, ):
        
        return self.all_blocks


class UnetSkipConnectionConcatBlock(nn.Module):
    """
    Defines the submodule with skip connection.
    X -------------------identity---------------------- X
    |-- downsampling -- |submodule| -- upsampling --|
    """
    
    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None, outermost=False, 
                innermost=False, norm_layer=nn.modules.normalization.GroupNorm):
        """
        :param outer_nc: Number of output channels
        :param input_nv: Number of input channels
        :param inner_nc: Number of intermediate channels
        :param submodule: Submodule to be used after down and before up sampling
        :param outermost: True or False. True if this is the outermost connection in the UNet
        :param innermost: True of False. True if this is the innermost connection in the UNet
        """
        
        super(UnetSkipConnectionConcatBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost

        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
    
        if outermost:
            self.down = [downconv]
            self.up = [upconv2d(inner_nc * 2, inner_nc), nn.Conv2d(inner_nc, outer_nc, kernel_size=3, stride=1, padding=1, bias=True)]
        elif innermost:
            self.down = [conv2d(False, input_nc, inner_nc, kernel_size=4, stride=2)]
            self.up = [upconv2d(inner_nc, outer_nc)]
        else:
            self.down = [conv2d(False, input_nc, inner_nc, kernel_size=4, stride=2)]
            self.up = [upconv2d(inner_nc * 2, outer_nc)]
            
        self.up = nn.Sequential(*self.up)
        self.down = nn.Sequential(*self.down)
        self.submodule = submodule

    def forward(self, x):
        x_inp = x
        self.x_enc = self.down(x_inp)
        if self.submodule is not None:
            out = self.submodule(self.x_enc)
        else:
            out = self.x_enc
        self.x_dec = self.up(out)
        if self.outermost:
            return self.x_dec
        else:
            return torch.cat([x_inp, self.x_dec], 1)
