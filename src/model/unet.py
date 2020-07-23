import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn


def upconv2d(in_planes, out_planes, mode='bilinear'):
    
    if mode == 'nearest':
        print('Using NN upsample!!')
    upconv = nn.Sequential(
        nn.Upsample(scale_factor=2, mode=mode),
        nn.ReflectionPad2d(1),
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0),
        nn.LeakyReLU(0.2,inplace=True)
    )
    
    return upconv


def conv2d(batch_norm, in_planes, out_planes, kernel_size=3, stride=1):
    
    if batch_norm:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.BatchNorm2d(out_planes),
            nn.LeakyReLU(0.2,inplace=True)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, bias=True),
            nn.LeakyReLU(0.2,inplace=True)
        )


def bilinear_init(kernel_size=4):
    # Following Caffe's BilinearUpsamplingFiller
    # https://github.com/BVLC/caffe/pull/2213/files
    import numpy as np
    width = kernel_size
    height = kernel_size
    f = int(np.ceil(width / 2.))
    cc = (2 * f - 1 - f % 2) / (2.*f)
    weights = torch.zeros((height, width))
    for y in range(height):
        for x in range(width):
            weights[y, x] = (1 - np.abs(x / f - cc)) * (1 - np.abs(y / f - cc))

    return weights


def net_init(net):
    for m in net.modules():
        if isinstance(m, nn.Linear):
            #n = m.out_features
            #m.weight.data.normal_(0, 0.02 / n) #this modified initialization seems to work better, but it's very hacky
            #n = m.in_features
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #xavier
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv2d): #or isinstance(m, nn.ConvTranspose2d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n)) #this modified initialization seems to work better, but it's very hacky
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.ConvTranspose2d):
            # Initialize Deconv with bilinear weights.
            base_weights = bilinear_init(m.weight.data.size(-1))
            base_weights = base_weights.unsqueeze(0).unsqueeze(0)
            m.weight.data = base_weights.repeat(m.weight.data.size(0), m.weight.data.size(1), 1, 1)
            if m.bias is not None:
                m.bias.data.zero_()

        if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
            #n = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.in_channels
            #m.weight.data.normal_(0, math.sqrt(2. / n))
            m.weight.data.normal_(0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()


#------ UNet style generator ------#
#----------------------------------#

# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
# concats additional features at the bottleneck
class UNet(nn.Module):
    
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.modules.normalization.GroupNorm):
        
        super(UNet, self).__init__()

        if num_downs >= 5:
            ngf_max = ngf*8
        else:
            ngf_max = ngf*pow(2, num_downs - 2)

        # construct unet structure
        all_blocks = []
        unet_block = UnetSkipConnectionConcatBlock(ngf_max, ngf_max, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)
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


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionConcatBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.modules.normalization.GroupNorm):
        super(UnetSkipConnectionConcatBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost

        # if submodule is None:
        #     pdb.set_trace()
        use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.01, False)
        uprelu = nn.ReLU(False)

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
