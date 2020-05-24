# model input: bottleneck layer of resnet 18+ CONV(FILTERS=64, kernel=4x4, S=2,) + 2 FC Layers
# model architecture:
# camera predictor:
#  - 2 FC Layer with leaky ReLU activation (see below) (different from the one in the encoder)
# - 100 input features (per default)
# - nn.Sequential(
#            nn.Linear(nc_inp, nc_out),
#            nn.LeakyReLU(0.1,inplace=True)
#        )
# - separate predictors for probability, scale, translation and quaternions (zero initialized)
# - biases: init
#   -         base_rotation = torch.FloatTensor([0.9239, 0, 0.3827 , 0]).unsqueeze(0).unsqueeze(0) ##pi/4
#   -        base_bias = torch.FloatTensor([ 0.7071,  0.7071,   0,   0]).unsqueeze(0).unsqueeze(0)

# model output: set of cameras with associated probabilities (also single camera output possible)
# - set of cameras
# loss function: re-projection loss (input mask - projected_mask squared); mask is a label


import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO: check whether encoded features are passed to the model


class CameraPredictor(nn.Module):
    def __init__(self, device, img_feats, encoder=None):
        super(CameraPredictor, self).__init__()
        if not encoder:
            _resnet = torch.hub.load('pytorch/vision:v0.6.0', 'resnet18', pretrained=True)
            self.encoder = nn.Sequential(*([*_resnet.children()][:-1]))

            for param in self.encoder.parameters():
                param.requires_grad = False
            self.avg_pool = None
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
            self.encoder = encoder

        self.fc = nn.Linear(img_feats, 256)
        self.fc2 = nn.Linear(256, 8)

    def forward(self, x):
        x = self.encoder(x)
        if self.avg_pool:
            x = self.avg_pool(x)

        x = x.squeeze(-1).squeeze(-1)  # convert NXCx1x1 tensor to a NXC vector
        x = self.fc(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)

        # predicted cam_params
        scale = x[..., 0]
        translate = x[..., 1:3 + 1]
        rotate = x[..., 4:]

        return scale, translate, rotate


class MultiCameraPredictor(nn.Module):
    """Module for predicting a set of camera poses and a corresponding probabilities."""

    def __init__(self):
        pass

    def forward(self):
        pass
